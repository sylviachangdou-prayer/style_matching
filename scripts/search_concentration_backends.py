#!/usr/bin/env python3
"""Dev-select a low-hubness scoring backend without retraining the encoder."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluate_hubness_reranking import delta_interval, exposure, intervals
from scripts.evaluate_similarity_backends import (
    align_embeddings,
    assemble_scores,
    point_metrics,
    source_key,
)
from scripts.style_embedding_recall import balanced_train, profile_key


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--embedding-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-cap", type=int, default=300)
    parser.add_argument("--bootstrap-runs", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260903)
    parser.add_argument("--watch-profile", action="append", default=[])
    return parser.parse_args()


def specifications() -> list[tuple[str, object]]:
    shrinkages = (0.03, 0.05, 0.1, 0.15, 0.2, 0.3)
    local_shrinkages = (0.05, 0.1, 0.15)
    return [
        ("cosine", None),
        *(("whitened_cosine", value) for value in shrinkages),
        *(("author_balanced_whitened_cosine", value) for value in shrinkages),
        *(("whitened_csls", (value, k)) for value in local_shrinkages for k in (5, 10, 20)),
        *(("author_balanced_whitened_csls", (value, k)) for value in local_shrinkages for k in (5, 10, 20)),
        *(("cosine_whitened_blend", (value, alpha)) for value in local_shrinkages for alpha in (0.25, 0.5, 0.75)),
    ]


def candidate_id(method: str, parameter: object) -> str:
    return f"{method}:{parameter!r}"


def worst_decile_recall_at_3(scores: np.ndarray, labels: np.ndarray) -> float:
    order = np.argsort(scores, axis=1)[:, ::-1]
    ranks = np.asarray([
        np.flatnonzero(row == true)[0] + 1 for row, true in zip(order, labels)
    ])
    by_profile = [float(np.mean(ranks[labels == label] <= 3)) for label in np.unique(labels)]
    return float(np.quantile(by_profile, 0.1))


def evaluate(
    scores: np.ndarray,
    labels: np.ndarray,
    profiles: np.ndarray,
    groups: np.ndarray,
) -> tuple[dict[str, float], pd.DataFrame]:
    concentration, table = exposure(scores, labels, profiles, groups=groups, top_k=3)
    return {
        **point_metrics(scores, labels),
        "worst_decile_profile_recall_at_3": worst_decile_recall_at_3(scores, labels),
        **concentration,
    }, table


def subgroup_table(
    method: str,
    frame: pd.DataFrame,
    scores: np.ndarray,
    labels: np.ndarray,
) -> pd.DataFrame:
    rows = []
    for column in ("language", "corpus"):
        for value, subset in frame.groupby(column, sort=True):
            positions = subset.index.to_numpy()
            rows.append({
                "method": method,
                "group_type": column,
                "group": str(value),
                "n_queries": int(len(positions)),
                "n_true_profiles": int(len(np.unique(labels[positions]))),
                **point_metrics(scores[positions], labels[positions]),
            })
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.read_parquet(args.input).copy()
    frame["profile_key"] = profile_key(frame)
    train = balanced_train(frame, args.train_cap, args.seed)
    train["profile_key"] = profile_key(train)
    evaluation = frame[frame["split"].isin(["dev", "test"])].copy().reset_index(drop=True)
    profiles = np.asarray(sorted(train["profile_key"].unique()), dtype=str)
    profile_lookup = {profile: position for position, profile in enumerate(profiles)}
    evaluation = evaluation[evaluation["profile_key"].isin(profile_lookup)].reset_index(drop=True)
    labels = np.asarray([profile_lookup[value] for value in evaluation["profile_key"]])
    train_embeddings = align_embeddings(
        train,
        args.embedding_dir / "style_embedding_train_chunk_ids.npy",
        args.embedding_dir / "style_embedding_train_embeddings.npy",
    )
    eval_embeddings = align_embeddings(
        evaluation,
        args.embedding_dir / "style_embedding_eval_chunk_ids.npy",
        args.embedding_dir / "style_embedding_eval_embeddings.npy",
    )
    dev = evaluation["split"].eq("dev").to_numpy()
    test = evaluation["split"].eq("test").to_numpy()
    dev_frame = evaluation[dev].reset_index(drop=True)
    test_frame = evaluation[test].reset_index(drop=True)
    dev_groups = source_key(dev_frame).to_numpy()
    test_groups = source_key(test_frame).to_numpy()
    fit_cache: dict[str, dict] = {}

    baseline_scores = assemble_scores(
        train, train_embeddings, evaluation, eval_embeddings,
        profiles, "cosine", None, fit_cache,
    )
    baseline_dev, _ = evaluate(
        baseline_scores[dev], labels[dev], profiles, dev_groups
    )
    rows = []
    selected: tuple[str, object, np.ndarray, dict[str, float]] | None = None
    selected_key: tuple[float, float] | None = None
    for method, parameter in specifications():
        scores = baseline_scores if method == "cosine" else assemble_scores(
            train, train_embeddings, evaluation, eval_embeddings,
            profiles, method, parameter, fit_cache,
        )
        metrics, _ = evaluate(scores[dev], labels[dev], profiles, dev_groups)
        metrics["mrr_delta_vs_cosine"] = metrics["mrr"] - baseline_dev["mrr"]
        metrics["recall_at_3_delta_vs_cosine"] = (
            metrics["recall_at_3"] - baseline_dev["recall_at_3"]
        )
        metrics["worst_decile_delta_vs_cosine"] = (
            metrics["worst_decile_profile_recall_at_3"]
            - baseline_dev["worst_decile_profile_recall_at_3"]
        )
        ratios = [
            metrics[name] / max(baseline_dev[name], 1e-12)
            for name in ("false_top3_hhi", "false_top3_gini", "maximum_false_top3_share")
        ]
        metrics["concentration_index"] = float(np.mean(ratios))
        eligible = (
            metrics["mrr_delta_vs_cosine"] >= -0.01
            and metrics["recall_at_3_delta_vs_cosine"] >= -0.01
            and metrics["worst_decile_delta_vs_cosine"] >= -0.02
            and metrics["false_top3_hhi"] < baseline_dev["false_top3_hhi"]
            and metrics["false_top3_gini"] < baseline_dev["false_top3_gini"]
            and metrics["maximum_false_top3_share"] < baseline_dev["maximum_false_top3_share"]
        )
        row = {
            "candidate": candidate_id(method, parameter),
            "method": method,
            "parameter": repr(parameter),
            **metrics,
            "dev_eligible": eligible,
        }
        rows.append(row)
        key = (metrics["concentration_index"], -metrics["mrr"])
        if eligible and (selected_key is None or key < selected_key):
            selected = (method, parameter, scores.copy(), metrics)
            selected_key = key

    dev_results = pd.DataFrame(rows).sort_values(
        ["dev_eligible", "concentration_index", "mrr"],
        ascending=[False, True, False],
    )
    dev_results.to_csv(args.output_dir / "concentration_dev_search.csv", index=False)
    if selected is None:
        report = {
            "status": "no_dev_candidate_met_quality_and_concentration_constraints",
            "seed": args.seed,
            "production_change_authorized": False,
        }
        (args.output_dir / "concentration_selection.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )
        print(json.dumps(report, indent=2))
        print(f"RETURN: {args.output_dir / 'concentration_selection.json'}")
        return

    method, parameter, selected_scores, selected_dev = selected
    test_labels = labels[test]
    comparison_rows = []
    exposure_rows = []
    subgroup_rows = []
    for position, (name, scores) in enumerate((
        ("cosine", baseline_scores[test]),
        (candidate_id(method, parameter), selected_scores[test]),
    )):
        estimates = intervals(
            scores, test_labels, args.bootstrap_runs, args.seed + 100 * position
        )
        metrics, table = evaluate(scores, test_labels, profiles, test_groups)
        table.insert(0, "method", name)
        exposure_rows.append(table)
        comparison_rows.append({
            "method": name,
            **metrics,
            **{f"{metric}_ci_low": values["ci_low"] for metric, values in estimates.items()},
            **{f"{metric}_ci_high": values["ci_high"] for metric, values in estimates.items()},
        })
        subgroup_rows.append(subgroup_table(name, test_frame, scores, test_labels))

    comparison = pd.DataFrame(comparison_rows)
    baseline_test = baseline_scores[test]
    candidate_test = selected_scores[test]
    deltas = {
        metric: delta_interval(
            baseline_test, candidate_test, test_labels, metric,
            args.bootstrap_runs, args.seed + 1000 + position,
        )
        for position, metric in enumerate(("mrr", "recall_at_3"))
    }
    subgroup = pd.concat(subgroup_rows, ignore_index=True)
    baseline_subgroups = subgroup[subgroup["method"].eq("cosine")].set_index(
        ["group_type", "group"]
    )
    candidate_subgroups = subgroup[~subgroup["method"].eq("cosine")].set_index(
        ["group_type", "group"]
    )
    supported = candidate_subgroups[candidate_subgroups["n_true_profiles"].ge(10)]
    subgroup_pass = all(
        float(row["mrr"] - baseline_subgroups.loc[key, "mrr"]) >= -0.02
        and float(row["recall_at_3"] - baseline_subgroups.loc[key, "recall_at_3"]) >= -0.02
        for key, row in supported.iterrows()
    )
    baseline_test_metrics = comparison.iloc[0]
    candidate_test_metrics = comparison.iloc[1]
    diagnostic_gate = (
        deltas["mrr"]["ci_low"] >= -0.01
        and deltas["recall_at_3"]["delta"] >= -0.01
        and candidate_test_metrics["false_top3_hhi"] < baseline_test_metrics["false_top3_hhi"]
        and candidate_test_metrics["false_top3_gini"] < baseline_test_metrics["false_top3_gini"]
        and candidate_test_metrics["maximum_false_top3_share"] < baseline_test_metrics["maximum_false_top3_share"]
        and subgroup_pass
    )
    exposure_frame = pd.concat(exposure_rows, ignore_index=True)
    if args.watch_profile:
        watched = exposure_frame[exposure_frame["profile"].isin(args.watch_profile)].copy()
    else:
        watched = exposure_frame.iloc[0:0].copy()
    report = {
        "status": "exploratory_existing_test_reused",
        "seed": args.seed,
        "selection_unit": "dev only",
        "selection_objective": "minimize mean relative false-top3 HHI/Gini/max-share subject to MRR, Recall@3, and worst-decile non-inferiority",
        "selected_candidate": candidate_id(method, parameter),
        "selected_dev": selected_dev,
        "test_diagnostic_gate": bool(diagnostic_gate),
        "test_subgroup_non_degradation": bool(subgroup_pass),
        "paired_test_deltas": deltas,
        "production_change_authorized": False,
        "reason": "the existing test set was previously opened; confirm on newly frozen sources and recalibrate open-set rejection",
    }
    comparison.to_csv(args.output_dir / "concentration_test_diagnostic.csv", index=False)
    exposure_frame.to_csv(args.output_dir / "concentration_author_exposure.csv", index=False)
    watched.to_csv(args.output_dir / "concentration_watched_profiles.csv", index=False)
    subgroup.to_csv(args.output_dir / "concentration_subgroup_metrics.csv", index=False)
    (args.output_dir / "concentration_selection.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    print(f"RETURN: {args.output_dir / 'concentration_selection.json'}")


if __name__ == "__main__":
    main()
