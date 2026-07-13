from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.retrieval_metrics import open_set_metrics, ranking_metrics
from scripts.score_artifact_utils import (
    aggregate_scores_by_source,
    aligned_metadata,
    candidate_mask,
    load_aligned_scores,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate source-level multi-view open-set verification without re-encoding."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--scores", action="append", required=True, help="NAME=NPZ_PATH:ARRAY_KEY")
    parser.add_argument("--language", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260713)
    parser.add_argument("--unknown-dev-fraction", type=float, default=0.15)
    parser.add_argument("--unknown-test-fraction", type=float, default=0.15)
    parser.add_argument("--protocol-label", default="index_open_set")
    return parser.parse_args()


def local_labels(labels: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    mapping = {int(label): index for index, label in enumerate(candidates)}
    return np.asarray([mapping[int(label)] for label in labels], dtype=int)


def macro_mrr(scores: np.ndarray, labels: np.ndarray) -> float:
    rows = [ranking_metrics(scores[labels == label], labels[labels == label])["mrr"] for label in np.unique(labels)]
    return float(np.mean(rows))


def confidence_features(
    matrices: dict[str, np.ndarray],
    row_indices: np.ndarray,
    candidate_indices: np.ndarray,
) -> tuple[np.ndarray, list[str], np.ndarray]:
    blocks = []
    names = []
    predictions = []
    for name, matrix in matrices.items():
        values = matrix[row_indices][:, candidate_indices]
        mean = values.mean(axis=1, keepdims=True)
        standard_deviation = np.maximum(values.std(axis=1, keepdims=True), 1e-8)
        z = (values - mean) / standard_deviation
        ordered = np.sort(z, axis=1)[:, ::-1]
        shifted = z - z.max(axis=1, keepdims=True)
        probabilities = np.exp(shifted)
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        normalized_entropy = -np.sum(
            probabilities * np.log(np.maximum(probabilities, 1e-12)), axis=1
        ) / max(np.log(values.shape[1]), 1.0)
        top5_position = min(4, values.shape[1] - 1)
        blocks.append(
            np.column_stack(
                [
                    values.max(axis=1),
                    ordered[:, 0],
                    ordered[:, 0] - ordered[:, 1],
                    ordered[:, 0] - ordered[:, top5_position],
                    1.0 - normalized_entropy,
                ]
            )
        )
        names.extend(
            [
                f"{name}.raw_max",
                f"{name}.z_max",
                f"{name}.z_margin_1_2",
                f"{name}.z_gap_1_5",
                f"{name}.concentration",
            ]
        )
        predictions.append(values.argmax(axis=1))
    prediction_matrix = np.column_stack(predictions)
    agreement = []
    for row in prediction_matrix:
        counts = np.bincount(row, minlength=len(candidate_indices))
        agreement.append(counts.max() / len(row))
    blocks.append(np.asarray(agreement)[:, None])
    names.append("cross_view.top1_agreement")
    return np.column_stack(blocks), names, prediction_matrix


def fit_probability_model(
    features: np.ndarray, labels: np.ndarray, seed: int
) -> tuple[LogisticRegression | None, np.ndarray]:
    if len(np.unique(labels)) < 2:
        return None, np.full(len(labels), float(labels[0]))
    model = LogisticRegression(
        penalty="l2", C=0.25, class_weight="balanced", max_iter=2000, random_state=seed
    )
    model.fit(features, labels)
    return model, model.predict_proba(features)[:, 1]


def main() -> None:
    args = parse_args()
    matrices, reference = load_aligned_scores(args.scores)
    frame = aligned_metadata(args.input, reference)
    source_frame, labels, source_matrices = aggregate_scores_by_source(
        frame, reference["y_true"].astype(int), matrices
    )
    profiles = reference["profiles"].astype(str)
    language_candidates = np.flatnonzero(candidate_mask(profiles, args.language))
    observed_labels = np.intersect1d(
        language_candidates,
        np.unique(labels[source_frame["language"].astype(str).eq(args.language).to_numpy()]),
    )
    if len(observed_labels) < 10:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        skipped = {
            "language": args.language,
            "n_profiles": int(len(observed_labels)),
            "reason": "fewer than 10 source-heldout profiles",
        }
        (args.output_dir / "open_set_skipped.json").write_text(
            json.dumps(skipped, indent=2), encoding="utf-8"
        )
        print(json.dumps(skipped, indent=2))
        return
    rng = np.random.default_rng(args.seed)
    shuffled = observed_labels.copy()
    rng.shuffle(shuffled)
    n_unknown_dev = max(1, int(len(shuffled) * args.unknown_dev_fraction))
    n_unknown_test = max(1, int(len(shuffled) * args.unknown_test_fraction))
    unknown_dev = np.sort(shuffled[:n_unknown_dev])
    unknown_test = np.sort(shuffled[n_unknown_dev:n_unknown_dev + n_unknown_test])
    known = np.sort(shuffled[n_unknown_dev + n_unknown_test:])
    language_mask = source_frame["language"].astype(str).eq(args.language).to_numpy()
    splits = source_frame["split"].astype(str).to_numpy()
    dev_rows = np.flatnonzero(
        language_mask & (splits == "dev") & np.isin(labels, np.concatenate([known, unknown_dev]))
    )
    test_rows = np.flatnonzero(
        language_mask & (splits == "test") & np.isin(labels, np.concatenate([known, unknown_test]))
    )
    dev_known = np.isin(labels[dev_rows], known)
    test_known = np.isin(labels[test_rows], known)
    if not len(dev_rows) or not len(test_rows) or len(np.unique(dev_known)) < 2 or len(np.unique(test_known)) < 2:
        raise ValueError(f"Empty or degenerate open-set source partition for {args.language}")
    dev_features, feature_names, _ = confidence_features(
        source_matrices, dev_rows, known
    )
    test_features, _, _ = confidence_features(source_matrices, test_rows, known)
    knownness_model, _ = fit_probability_model(dev_features, dev_known.astype(int), args.seed)
    if knownness_model is None:
        raise ValueError("Knownness calibration requires both known and unknown dev sources")
    known_probability = knownness_model.predict_proba(test_features)[:, 1]
    dev_view_metrics = {}
    known_dev_rows = dev_rows[dev_known]
    known_dev_labels = local_labels(labels[known_dev_rows], known)
    for name, matrix in source_matrices.items():
        dev_view_metrics[name] = macro_mrr(matrix[known_dev_rows][:, known], known_dev_labels)
    ranking_view = max(dev_view_metrics, key=dev_view_metrics.get)
    ranking_test_scores = source_matrices[ranking_view][test_rows][:, known]
    predicted_local = ranking_test_scores.argmax(axis=1)
    predicted_global = known[predicted_local]
    correct = test_known & (predicted_global == labels[test_rows])
    ranking_confidence = np.column_stack(
        [
            ranking_test_scores.max(axis=1),
            np.sort(ranking_test_scores, axis=1)[:, -1]
            - np.sort(ranking_test_scores, axis=1)[:, -2],
        ]
    )
    dev_ranking_scores = source_matrices[ranking_view][dev_rows][:, known]
    dev_predicted_global = known[dev_ranking_scores.argmax(axis=1)]
    dev_correct = dev_known & (dev_predicted_global == labels[dev_rows])
    dev_ranking_confidence = np.column_stack(
        [
            dev_ranking_scores.max(axis=1),
            np.sort(dev_ranking_scores, axis=1)[:, -1]
            - np.sort(dev_ranking_scores, axis=1)[:, -2],
        ]
    )
    correctness_model, _ = fit_probability_model(
        np.column_stack([dev_features, dev_ranking_confidence]),
        dev_correct.astype(int),
        args.seed,
    )
    if correctness_model is None:
        correctness_probability = np.full(len(test_rows), float(dev_correct[0]))
    else:
        correctness_probability = correctness_model.predict_proba(
            np.column_stack([test_features, ranking_confidence])
        )[:, 1]
    top_half = np.argsort(correctness_probability)[::-1][: max(1, len(test_rows) // 2)]
    known_test_rows = np.flatnonzero(test_known)
    known_test_labels = local_labels(labels[test_rows][test_known], known)
    report = {
        "protocol": {
            "label": args.protocol_label,
            "evaluation_unit": "independent_source",
            "unknown_unit": "entire author-language profile",
            "unknown_dev_profiles": "used only to fit verifier",
            "unknown_test_profiles": "locked test",
            "candidate_profiles_removed": True,
            "warning": (
                "locally fine-tuned views make this index-open-set, not model-unseen-author evaluation"
                if args.protocol_label == "local_index"
                else "local author labels were not used, but external pretraining membership is not auditable"
            ),
        },
        "language": args.language,
        "views": list(source_matrices),
        "n_candidate_profiles": int(len(known)),
        "n_unknown_dev_profiles": int(len(unknown_dev)),
        "n_unknown_test_profiles": int(len(unknown_test)),
        "n_dev_sources": int(len(dev_rows)),
        "n_test_sources": int(len(test_rows)),
        "ranking_view": ranking_view,
        "dev_known_mrr_by_view": dev_view_metrics,
        "known_ranking": ranking_metrics(
            ranking_test_scores[test_known], known_test_labels
        ),
        "open_set": open_set_metrics(
            test_known.astype(int), known_probability, known_probability
        ),
        "selective_top1_precision_at_50pct_coverage": float(correct[top_half].mean()),
        "knownness_features": feature_names,
        "knownness_coefficients": dict(
            zip(feature_names, knownness_model.coef_[0].tolist())
        ),
        "knownness_intercept": float(knownness_model.intercept_[0]),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "open_set_metrics.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    pd.DataFrame(
        {
            "source_key": source_frame.iloc[test_rows]["source_key"].astype(str).to_numpy(),
            "true_profile": profiles[labels[test_rows]],
            "known": test_known,
            "predicted_profile": profiles[predicted_global],
            "known_probability": known_probability,
            "correctness_probability": correctness_probability,
            "correct": correct,
        }
    ).to_parquet(args.output_dir / "open_set_predictions.parquet", index=False)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
