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

from scripts.retrieval_metrics import expected_calibration_error, paired_bootstrap_mrr, ranking_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare retrieval scores and adopt fusion only on held-out evidence.")
    parser.add_argument(
        "--scores",
        action="append",
        required=True,
        help="NAME=NPZ_PATH:ARRAY_KEY; repeat for every candidate score matrix.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-runs", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260711)
    return parser.parse_args()


def load_spec(spec: str) -> tuple[str, Path, str]:
    name, location = spec.split("=", 1)
    path, key = location.rsplit(":", 1)
    return name, Path(path), key


def confidence_features(scores: np.ndarray) -> np.ndarray:
    ordered = np.sort(scores, axis=1)[:, ::-1]
    return np.column_stack([ordered[:, 0], ordered[:, 0] - ordered[:, 1]])


def top1_calibration(
    dev_scores: np.ndarray,
    dev_labels: np.ndarray,
    test_scores: np.ndarray,
    test_labels: np.ndarray,
) -> dict[str, float]:
    dev_correct = (dev_scores.argmax(axis=1) == dev_labels).astype(int)
    correct = test_scores.argmax(axis=1) == test_labels
    if len(np.unique(dev_correct)) == 2:
        calibrator = LogisticRegression().fit(confidence_features(dev_scores), dev_correct)
        confidence = calibrator.predict_proba(confidence_features(test_scores))[:, 1]
    else:
        confidence = np.full(len(test_scores), float(dev_correct[0]))
    top_half = np.argsort(confidence)[::-1][: max(1, len(confidence) // 2)]
    return {
        "ece": expected_calibration_error(correct.astype(int), confidence),
        "top1_precision_at_50pct_coverage": float(correct[top_half].mean()),
    }


def main() -> None:
    args = parse_args()
    matrices = {}
    reference = None
    for spec in args.scores:
        name, path, key = load_spec(spec)
        payload = np.load(path, allow_pickle=False)
        if reference is None:
            reference = {
                "chunk_ids": payload["chunk_ids"],
                "splits": payload["splits"],
                "query_languages": payload["query_languages"],
                "query_corpora": payload["query_corpora"],
                "profiles": payload["profiles"],
                "y_true": payload["y_true"],
            }
        else:
            for field in (
                "chunk_ids", "splits", "query_languages", "query_corpora", "profiles", "y_true"
            ):
                if not np.array_equal(reference[field], payload[field]):
                    raise ValueError(f"Score files are not aligned on {field}: {path}")
        matrices[name] = payload[key].astype("float64")
    if len(matrices) < 2:
        raise ValueError("At least two candidate score matrices are required")
    assert reference is not None
    splits = reference["splits"].astype(str)
    labels = reference["y_true"].astype(int)
    dev = splits == "dev"
    test = splits == "test"
    if not dev.any() or not test.any():
        raise ValueError("Aligned scores must contain separate dev and test queries")

    dev_results = {name: ranking_metrics(scores[dev], labels[dev]) for name, scores in matrices.items()}
    best_single = max(dev_results, key=lambda name: dev_results[name]["mrr"])
    names = list(matrices)
    dev_features = np.stack([matrices[name][dev] for name in names], axis=-1)
    n_queries, n_candidates, n_features = dev_features.shape
    binary_labels = (np.arange(n_candidates)[None, :] == labels[dev, None]).astype(int)
    fusion = LogisticRegression(class_weight="balanced", max_iter=2000)
    fusion.fit(dev_features.reshape(-1, n_features), binary_labels.reshape(-1))
    test_features = np.stack([matrices[name][test] for name in names], axis=-1)
    fusion_dev_scores = fusion.predict_proba(dev_features.reshape(-1, n_features))[:, 1].reshape(
        len(dev_features), n_candidates
    )
    fusion_scores = fusion.predict_proba(test_features.reshape(-1, n_features))[:, 1].reshape(
        len(test_features), n_candidates
    )
    single_test = matrices[best_single][test]
    bootstrap = paired_bootstrap_mrr(
        single_test, fusion_scores, labels[test], args.bootstrap_runs, args.seed
    )
    best_calibration = top1_calibration(
        matrices[best_single][dev], labels[dev], single_test, labels[test]
    )
    fusion_calibration = top1_calibration(
        fusion_dev_scores, labels[dev], fusion_scores, labels[test]
    )
    subgroup_comparison = {}
    subgroup_not_worse = True
    for field in ("query_languages", "query_corpora"):
        values = reference[field].astype(str)[test]
        field_report = {}
        for value in sorted(set(values)):
            mask = values == value
            if int(mask.sum()) < 50:
                continue
            single_mrr = ranking_metrics(single_test[mask], labels[test][mask])["mrr"]
            fusion_mrr = ranking_metrics(fusion_scores[mask], labels[test][mask])["mrr"]
            field_report[value] = {
                "n_queries": int(mask.sum()),
                "best_single_mrr": single_mrr,
                "fusion_mrr": fusion_mrr,
                "delta": fusion_mrr - single_mrr,
            }
            subgroup_not_worse &= fusion_mrr >= single_mrr
        subgroup_comparison[field] = field_report
    adopt_fusion = (
        bootstrap["ci_low"] > 0
        and fusion_calibration["ece"] <= best_calibration["ece"] + 0.01
        and fusion_calibration["top1_precision_at_50pct_coverage"]
        >= best_calibration["top1_precision_at_50pct_coverage"]
        and subgroup_not_worse
    )
    report = {
        "candidate_order": names,
        "dev_metrics": dev_results,
        "best_single": best_single,
        "test_metrics": {
            best_single: ranking_metrics(single_test, labels[test]),
            "learned_fusion": ranking_metrics(fusion_scores, labels[test]),
        },
        "calibration": {best_single: best_calibration, "learned_fusion": fusion_calibration},
        "paired_bootstrap": bootstrap,
        "major_subgroups": subgroup_comparison,
        "major_subgroups_not_worse": subgroup_not_worse,
        "fusion_coefficients": dict(zip(names, fusion.coef_[0].tolist())),
        "fusion_intercept": float(fusion.intercept_[0]),
        "decision": "learned_fusion" if adopt_fusion else best_single,
        "fusion_adopted": adopt_fusion,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
