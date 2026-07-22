from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.compare_retrieval_models import top1_calibration
from scripts.retrieval_metrics import ranks_from_scores, ranking_metrics
from scripts.score_artifact_utils import (
    aggregate_scores_by_source,
    aligned_metadata,
    candidate_mask,
    load_aligned_scores,
    normalized_score_features,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select a base-anchored, group-robust convex rank fusion on dev and evaluate once on test."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--scores", action="append", required=True)
    parser.add_argument("--base", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--weight-step", type=float, default=0.1)
    parser.add_argument("--minimum-base-weight", type=float, default=0.5)
    parser.add_argument("--minimum-group-sources", type=int, default=8)
    parser.add_argument("--group-tolerance", type=float, default=0.02)
    parser.add_argument("--selection-quantile", type=float, default=0.10)
    parser.add_argument("--bootstrap-runs", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260723)
    return parser.parse_args()


def macro_metrics(scores: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    rows = [
        ranking_metrics(scores[labels == label], labels[labels == label])
        for label in np.unique(labels)
    ]
    return {
        metric: float(np.mean([row[metric] for row in rows]))
        for metric in rows[0]
    }


def reciprocal_rank(scores: np.ndarray, labels: np.ndarray) -> np.ndarray:
    return 1.0 / ranks_from_scores(scores, labels)


def profile_deltas(
    baseline: np.ndarray,
    candidate: np.ndarray,
    labels: np.ndarray,
) -> np.ndarray:
    delta = reciprocal_rank(candidate, labels) - reciprocal_rank(baseline, labels)
    return np.asarray([
        float(delta[labels == label].mean()) for label in np.unique(labels)
    ])


def paired_profile_bootstrap(
    baseline: np.ndarray,
    candidate: np.ndarray,
    labels: np.ndarray,
    runs: int,
    seed: int,
) -> dict[str, float]:
    delta = profile_deltas(baseline, candidate, labels)
    rng = np.random.default_rng(seed)
    draws = np.asarray([
        delta[rng.integers(0, len(delta), len(delta))].mean()
        for _ in range(runs)
    ])
    return {
        "mrr_delta": float(delta.mean()),
        "ci_low": float(np.quantile(draws, 0.025)),
        "ci_high": float(np.quantile(draws, 0.975)),
    }


def simplex_weights(
    names: list[str],
    base: str,
    step: float,
    minimum_base_weight: float,
) -> list[dict[str, float]]:
    units = round(1.0 / step)
    if not np.isclose(units * step, 1.0):
        raise ValueError("weight-step must divide 1.0 exactly")
    base_index = names.index(base)
    candidates = []
    for values in itertools.product(range(units + 1), repeat=len(names)):
        if sum(values) != units:
            continue
        weights = np.asarray(values, dtype="float64") / units
        if weights[base_index] + 1e-12 < minimum_base_weight:
            continue
        candidates.append(dict(zip(names, weights.tolist())))
    return candidates


def combine(
    weights: dict[str, float],
    rank_views: dict[str, np.ndarray],
    languages: np.ndarray,
    profiles: np.ndarray,
) -> np.ndarray:
    scores = sum(weights[name] * rank_views[name] for name in weights)
    scores = scores.copy()
    for row, language in enumerate(languages.astype(str)):
        scores[row, ~candidate_mask(profiles, language)] = -1e9
    return scores


def subgroup_report(
    baseline: np.ndarray,
    candidate: np.ndarray,
    labels: np.ndarray,
    frame: pd.DataFrame,
    minimum_sources: int,
) -> tuple[dict[str, object], float]:
    report: dict[str, object] = {}
    minimum_delta = float("inf")
    for column in ("language", "corpus"):
        field = {}
        values = frame[column].astype(str).to_numpy()
        for value in sorted(set(values)):
            mask = values == value
            if int(mask.sum()) < minimum_sources:
                continue
            base_mrr = macro_metrics(baseline[mask], labels[mask])["mrr"]
            candidate_mrr = macro_metrics(candidate[mask], labels[mask])["mrr"]
            delta = candidate_mrr - base_mrr
            field[value] = {
                "n_sources": int(mask.sum()),
                "base_mrr": base_mrr,
                "candidate_mrr": candidate_mrr,
                "delta": delta,
            }
            minimum_delta = min(minimum_delta, delta)
        report[column] = field
    if minimum_delta == float("inf"):
        minimum_delta = 0.0
    return report, minimum_delta


def main() -> None:
    args = parse_args()
    matrices, reference = load_aligned_scores(args.scores)
    if args.base not in matrices:
        raise ValueError(f"base view not found: {args.base}")
    if not 2 <= len(matrices) <= 4:
        raise ValueError("robust convex fusion expects two to four deliberately screened views")

    frame = aligned_metadata(args.input, reference)
    source_frame, labels, source_matrices = aggregate_scores_by_source(
        frame, reference["y_true"].astype(int), matrices
    )
    profiles = reference["profiles"].astype(str)
    languages = source_frame["language"].astype(str).to_numpy()
    splits = source_frame["split"].astype(str).to_numpy()
    dev = np.flatnonzero(splits == "dev")
    test = np.flatnonzero(splits == "test")
    if not len(dev) or not len(test):
        raise ValueError("separate dev and locked test sources are required")

    rank_views = {
        name: normalized_score_features(scores, languages, profiles)[1]
        for name, scores in source_matrices.items()
    }
    base_weights = {name: float(name == args.base) for name in matrices}
    base_scores = combine(base_weights, rank_views, languages, profiles)
    base_dev = base_scores[dev]
    base_test = base_scores[test]
    dev_labels = labels[dev]
    test_labels = labels[test]

    rng = np.random.default_rng(args.seed)
    n_dev_profiles = len(np.unique(dev_labels))
    bootstrap_indices = rng.integers(
        0, n_dev_profiles, size=(args.bootstrap_runs, n_dev_profiles)
    )
    candidate_rows = []
    candidate_scores: dict[str, np.ndarray] = {}
    for index, weights in enumerate(
        simplex_weights(
            list(matrices), args.base, args.weight_step, args.minimum_base_weight
        )
    ):
        scores = combine(weights, rank_views, languages, profiles)
        dev_scores = scores[dev]
        delta = profile_deltas(base_dev, dev_scores, dev_labels)
        lower_bound = float(
            np.quantile(delta[bootstrap_indices].mean(axis=1), args.selection_quantile)
        )
        groups, minimum_group_delta = subgroup_report(
            base_dev,
            dev_scores,
            dev_labels,
            source_frame.iloc[dev].reset_index(drop=True),
            args.minimum_group_sources,
        )
        feasible = minimum_group_delta >= -args.group_tolerance
        name = f"candidate_{index:03d}"
        candidate_scores[name] = scores
        candidate_rows.append({
            "candidate": name,
            "weights": weights,
            "dev_mrr": macro_metrics(dev_scores, dev_labels)["mrr"],
            "dev_mrr_delta": float(delta.mean()),
            "selection_lower_bound": lower_bound,
            "minimum_group_delta": minimum_group_delta,
            "group_feasible": feasible,
            "dev_subgroups": groups,
        })

    feasible_rows = [row for row in candidate_rows if row["group_feasible"]]
    robust_row = max(
        feasible_rows,
        key=lambda row: (
            row["selection_lower_bound"], row["dev_mrr_delta"], row["dev_mrr"]
        ),
    )
    exploratory_row = max(
        feasible_rows,
        key=lambda row: (row["dev_mrr"], row["selection_lower_bound"]),
    )
    robust_scores = candidate_scores[robust_row["candidate"]]
    exploratory_scores = candidate_scores[exploratory_row["candidate"]]

    test_metrics = {
        args.base: macro_metrics(base_test, test_labels),
        "robust_rank_fusion": macro_metrics(robust_scores[test], test_labels),
        "dev_mean_challenger": macro_metrics(exploratory_scores[test], test_labels),
    }
    bootstrap = paired_profile_bootstrap(
        base_test,
        robust_scores[test],
        test_labels,
        args.bootstrap_runs,
        args.seed + 1,
    )
    calibration = {
        args.base: top1_calibration(base_dev, dev_labels, base_test, test_labels),
        "robust_rank_fusion": top1_calibration(
            robust_scores[dev], dev_labels, robust_scores[test], test_labels
        ),
    }
    test_subgroups, minimum_test_group_delta = subgroup_report(
        base_test,
        robust_scores[test],
        test_labels,
        source_frame.iloc[test].reset_index(drop=True),
        args.minimum_group_sources,
    )
    fusion_is_distinct = any(
        name != args.base and weight > 0
        for name, weight in robust_row["weights"].items()
    )
    adopted = (
        fusion_is_distinct
        and bootstrap["ci_low"] > 0
        and calibration["robust_rank_fusion"]["ece"]
        <= calibration[args.base]["ece"] + 0.01
        and calibration["robust_rank_fusion"]["top1_precision_at_50pct_coverage"]
        >= calibration[args.base]["top1_precision_at_50pct_coverage"]
        and minimum_test_group_delta >= 0
    )

    report = {
        "protocol": {
            "selection_split": "dev",
            "locked_evaluation_split": "test",
            "evaluation_unit": "independent_source",
            "bootstrap_unit": "author-language profile",
            "score_transform": "query-wise within-language candidate percentile",
            "weight_constraint": "non-negative simplex",
            "minimum_base_weight": args.minimum_base_weight,
            "dev_group_tolerance": args.group_tolerance,
            "selection_objective": f"{args.selection_quantile:.2f} bootstrap quantile of paired MRR delta",
            "topic_views_allowed": False,
        },
        "base": args.base,
        "views": list(matrices),
        "robust_selection": robust_row,
        "exploratory_dev_mean_selection": exploratory_row,
        "test_metrics": test_metrics,
        "paired_profile_bootstrap": bootstrap,
        "calibration": calibration,
        "test_subgroups": test_subgroups,
        "minimum_test_group_delta": minimum_test_group_delta,
        "fusion_adopted": adopted,
        "decision": "robust_rank_fusion" if adopted else args.base,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "robust_reranker_metrics.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    pd.DataFrame([
        {
            "candidate": row["candidate"],
            **{f"weight_{name}": row["weights"][name] for name in matrices},
            "dev_mrr": row["dev_mrr"],
            "dev_mrr_delta": row["dev_mrr_delta"],
            "selection_lower_bound": row["selection_lower_bound"],
            "minimum_group_delta": row["minimum_group_delta"],
            "group_feasible": row["group_feasible"],
        }
        for row in candidate_rows
    ]).sort_values(
        ["group_feasible", "selection_lower_bound", "dev_mrr"],
        ascending=False,
    ).to_csv(args.output_dir / "robust_reranker_candidates.csv", index=False)
    np.savez_compressed(
        args.output_dir / "robust_reranker_source_scores.npz",
        source_ids=source_frame["source_key"].astype(str).to_numpy(),
        splits=splits,
        labels=labels,
        profiles=profiles,
        base_scores=base_scores.astype("float32"),
        robust_rank_fusion=robust_scores.astype("float32"),
        dev_mean_challenger=exploratory_scores.astype("float32"),
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
