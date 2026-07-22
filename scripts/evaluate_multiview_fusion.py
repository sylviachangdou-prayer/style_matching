from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.compare_retrieval_models import top1_calibration
from scripts.retrieval_metrics import paired_bootstrap_mrr, ranking_metrics
from scripts.score_artifact_utils import (
    aggregate_scores_by_source,
    aligned_metadata,
    candidate_mask,
    load_aligned_scores,
    normalized_score_features,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Learn and audit source-level fusion of independently motivated style views."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--scores", action="append", required=True, help="NAME=NPZ_PATH:ARRAY_KEY")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-runs", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260713)
    parser.add_argument("--l1-ratio", type=float, default=0.25)
    parser.add_argument("--c", type=float, default=0.25)
    parser.add_argument("--minimum-subgroup-sources", type=int, default=5)
    return parser.parse_args()


def macro_metrics(scores: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    rows = [ranking_metrics(scores[labels == label], labels[labels == label]) for label in np.unique(labels)]
    return {metric: float(np.mean([row[metric] for row in rows])) for metric in rows[0]}


def profile_bootstrap_intervals(
    scores: np.ndarray,
    labels: np.ndarray,
    runs: int,
    seed: int,
) -> dict[str, dict[str, float]]:
    """Bootstrap author-language profiles while keeping each profile's sources together."""
    profiles = np.unique(labels)
    per_profile = [
        ranking_metrics(scores[labels == profile], labels[labels == profile])
        for profile in profiles
    ]
    metrics = tuple(per_profile[0])
    values = np.asarray([[row[metric] for metric in metrics] for row in per_profile])
    rng = np.random.default_rng(seed)
    draws = np.empty((runs, len(metrics)), dtype="float64")
    for run in range(runs):
        sampled = rng.integers(0, len(profiles), len(profiles))
        draws[run] = values[sampled].mean(axis=0)
    return {
        metric: {
            "estimate": float(values[:, index].mean()),
            "ci_low": float(np.quantile(draws[:, index], 0.025)),
            "ci_high": float(np.quantile(draws[:, index], 0.975)),
        }
        for index, metric in enumerate(metrics)
    }


def make_view_features(
    matrices: dict[str, np.ndarray], languages: np.ndarray, profiles: np.ndarray
) -> tuple[list[str], dict[str, np.ndarray]]:
    feature_names: list[str] = []
    features: dict[str, np.ndarray] = {}
    for name, matrix in matrices.items():
        z_scores, percentiles = normalized_score_features(matrix, languages, profiles)
        features[f"{name}.z"] = z_scores
        features[f"{name}.percentile"] = percentiles
        feature_names.extend([f"{name}.z", f"{name}.percentile"])
    return feature_names, features


def pair_rows(
    row_indices: np.ndarray,
    feature_names: list[str],
    features: dict[str, np.ndarray],
    languages: np.ndarray,
    profiles: np.ndarray,
    labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    x_rows = []
    y_rows = []
    for row in row_indices:
        candidates = np.flatnonzero(candidate_mask(profiles, languages[row]))
        x_rows.append(np.column_stack([features[name][row, candidates] for name in feature_names]))
        y_rows.append((candidates == labels[row]).astype(int))
    return np.vstack(x_rows), np.concatenate(y_rows)


def fit_fusion(
    train_rows: np.ndarray,
    feature_names: list[str],
    features: dict[str, np.ndarray],
    languages: np.ndarray,
    profiles: np.ndarray,
    labels: np.ndarray,
    c: float,
    l1_ratio: float,
    seed: int,
) -> LogisticRegression:
    x_train, y_train = pair_rows(
        train_rows, feature_names, features, languages, profiles, labels
    )
    model = LogisticRegression(
        penalty="elasticnet",
        solver="saga",
        l1_ratio=l1_ratio,
        C=c,
        class_weight="balanced",
        max_iter=4000,
        random_state=seed,
    )
    model.fit(x_train, y_train)
    return model


def predict_fusion(
    model: LogisticRegression,
    row_indices: np.ndarray,
    feature_names: list[str],
    features: dict[str, np.ndarray],
    languages: np.ndarray,
    profiles: np.ndarray,
) -> np.ndarray:
    scores = np.full((len(row_indices), len(profiles)), -1e9, dtype="float64")
    for output_row, row in enumerate(row_indices):
        candidates = np.flatnonzero(candidate_mask(profiles, languages[row]))
        x = np.column_stack([features[name][row, candidates] for name in feature_names])
        scores[output_row, candidates] = model.predict_proba(x)[:, 1]
    return scores


def main() -> None:
    args = parse_args()
    if any("topic" in spec.split("=", 1)[0].lower() for spec in args.scores):
        raise ValueError("Topic/content views are excluded from the Style Match fusion")
    matrices, reference = load_aligned_scores(args.scores)
    if len(matrices) < 2:
        raise ValueError("At least two style views are required")
    frame = aligned_metadata(args.input, reference)
    source_frame, labels, source_matrices = aggregate_scores_by_source(
        frame, reference["y_true"].astype(int), matrices
    )
    profiles = reference["profiles"].astype(str)
    languages = source_frame["language"].astype(str).to_numpy()
    splits = source_frame["split"].astype(str).to_numpy()
    dev_rows = np.flatnonzero(splits == "dev")
    test_rows = np.flatnonzero(splits == "test")
    if not len(dev_rows) or not len(test_rows):
        raise ValueError("Both dev and test independent sources are required")
    feature_names, features = make_view_features(source_matrices, languages, profiles)
    fusion = fit_fusion(
        dev_rows,
        feature_names,
        features,
        languages,
        profiles,
        labels,
        args.c,
        args.l1_ratio,
        args.seed,
    )
    fusion_dev = predict_fusion(
        fusion, dev_rows, feature_names, features, languages, profiles
    )
    fusion_test = predict_fusion(
        fusion, test_rows, feature_names, features, languages, profiles
    )
    dev_metrics = {
        name: macro_metrics(matrix[dev_rows], labels[dev_rows])
        for name, matrix in source_matrices.items()
    }
    best_single = max(dev_metrics, key=lambda name: dev_metrics[name]["mrr"])
    test_scores = {
        name: matrix[test_rows]
        for name, matrix in source_matrices.items()
    }
    best_test_scores = test_scores[best_single]
    fusion_test_metrics = macro_metrics(fusion_test, labels[test_rows])
    test_scores["learned_fusion"] = fusion_test
    test_metrics = {
        name: macro_metrics(matrix, labels[test_rows])
        for name, matrix in test_scores.items()
    }
    test_intervals = {
        name: profile_bootstrap_intervals(
            matrix,
            labels[test_rows],
            args.bootstrap_runs,
            args.seed + index,
        )
        for index, (name, matrix) in enumerate(test_scores.items())
    }
    bootstrap = paired_bootstrap_mrr(
        best_test_scores, fusion_test, labels[test_rows], args.bootstrap_runs, args.seed
    )
    best_calibration = top1_calibration(
        source_matrices[best_single][dev_rows],
        labels[dev_rows],
        best_test_scores,
        labels[test_rows],
    )
    fusion_calibration = top1_calibration(
        fusion_dev, labels[dev_rows], fusion_test, labels[test_rows]
    )
    subgroup_comparison: dict[str, object] = {}
    subgroup_not_worse = True
    for column in ("language", "corpus"):
        values = source_frame[column].astype(str).to_numpy()[test_rows]
        field_report = {}
        for value in sorted(set(values)):
            mask = values == value
            if int(mask.sum()) < args.minimum_subgroup_sources:
                continue
            single_mrr = macro_metrics(best_test_scores[mask], labels[test_rows][mask])["mrr"]
            fused_mrr = macro_metrics(fusion_test[mask], labels[test_rows][mask])["mrr"]
            field_report[value] = {
                "n_sources": int(mask.sum()),
                "best_single_mrr": single_mrr,
                "fusion_mrr": fused_mrr,
                "delta": fused_mrr - single_mrr,
            }
            subgroup_not_worse &= fused_mrr >= single_mrr
        subgroup_comparison[column] = field_report
    ablations = {}
    view_names = list(source_matrices)
    for dropped in view_names:
        retained = [name for name in feature_names if not name.startswith(f"{dropped}.")]
        if not retained:
            continue
        ablation_model = fit_fusion(
            dev_rows,
            retained,
            features,
            languages,
            profiles,
            labels,
            args.c,
            args.l1_ratio,
            args.seed,
        )
        ablation_scores = predict_fusion(
            ablation_model, test_rows, retained, features, languages, profiles
        )
        ablation_metrics = macro_metrics(ablation_scores, labels[test_rows])
        ablations[dropped] = {
            **ablation_metrics,
            "mrr_delta_vs_full_fusion": ablation_metrics["mrr"] - fusion_test_metrics["mrr"],
        }
    adopt = (
        bootstrap["ci_low"] > 0
        and fusion_calibration["ece"] <= best_calibration["ece"] + 0.01
        and fusion_calibration["top1_precision_at_50pct_coverage"]
        >= best_calibration["top1_precision_at_50pct_coverage"]
        and subgroup_not_worse
    )
    report = {
        "protocol": {
            "evaluation_unit": "independent_source",
            "chunk_aggregation": "mean score",
            "ranking_weighting": "macro by author-language profile",
            "training_split": "dev",
            "locked_evaluation_split": "test",
            "n_dev_sources": int(len(dev_rows)),
            "n_test_sources": int(len(test_rows)),
            "n_test_profiles": int(np.unique(labels[test_rows]).size),
            "features": "within-language z-score and candidate percentile for each view",
            "learner": "elastic-net logistic candidate reranker",
            "topic_views_allowed": False,
        },
        "candidate_views": view_names,
        "dev_metrics": dev_metrics,
        "best_single": best_single,
        "test_metrics": test_metrics,
        "test_intervals": test_intervals,
        "calibration": {best_single: best_calibration, "learned_fusion": fusion_calibration},
        "paired_bootstrap": bootstrap,
        "subgroups": subgroup_comparison,
        "major_subgroups_not_worse": subgroup_not_worse,
        "coefficients": dict(zip(feature_names, fusion.coef_[0].tolist())),
        "intercept": float(fusion.intercept_[0]),
        "leave_one_view_out": ablations,
        "decision": "learned_fusion" if adopt else best_single,
        "fusion_adopted": adopt,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "multiview_fusion_metrics.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    np.savez_compressed(
        args.output_dir / "multiview_fusion_source_scores.npz",
        source_ids=source_frame["source_key"].astype(str).to_numpy(),
        splits=splits,
        query_languages=languages,
        query_corpora=source_frame["corpus"].astype(str).to_numpy(),
        profiles=profiles,
        y_true=labels,
        fusion_scores=np.vstack([fusion_dev, fusion_test]),
        fusion_score_row_indices=np.concatenate([dev_rows, test_rows]),
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
