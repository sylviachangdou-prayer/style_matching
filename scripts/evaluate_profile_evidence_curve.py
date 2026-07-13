from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.retrieval_metrics import ranking_metrics
from scripts.score_artifact_utils import (
    aggregate_scores_by_source,
    aligned_metadata,
    independent_source_keys,
)
from scripts.style_embedding_recall import balanced_train, mask_cross_language_candidates, profile_key


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure retrieval as independent training-source evidence increases."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--eval-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fractions", default="0.25,0.5,0.75,1.0")
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--train-cap", type=int, default=300)
    parser.add_argument("--seed", type=int, default=20260701)
    parser.add_argument("--split", default="test")
    return parser.parse_args()


def macro_by_profile(scores: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    per_profile = [ranking_metrics(scores[labels == label], labels[labels == label]) for label in np.unique(labels)]
    return {
        metric: float(np.mean([row[metric] for row in per_profile]))
        for metric in ("recall_at_1", "recall_at_3", "recall_at_5", "recall_at_20", "mrr")
    }


def summarize(rows: list[dict[str, object]]) -> dict[str, object]:
    output: dict[str, object] = {"n_repeats": len(rows)}
    for unit in ("source_macro", "chunk_diagnostic"):
        output[unit] = {}
        languages = sorted({language for row in rows for language in row[unit]})
        for language in languages:
            output[unit][language] = {}
            for metric in ("recall_at_1", "recall_at_3", "recall_at_5", "recall_at_20", "mrr"):
                values = np.asarray([row[unit][language][metric] for row in rows if language in row[unit]])
                output[unit][language][metric] = {
                    "mean": float(values.mean()),
                    "ci_low": float(np.quantile(values, 0.025)),
                    "ci_high": float(np.quantile(values, 0.975)),
                }
    selected_counts = np.asarray([row["median_selected_sources_per_profile"] for row in rows])
    output["median_selected_sources_per_profile"] = float(selected_counts.mean())
    return output


def evaluate_units(
    scores: np.ndarray,
    labels: np.ndarray,
    languages: np.ndarray,
    split_mask: np.ndarray,
) -> dict[str, dict[str, float]]:
    report: dict[str, dict[str, float]] = {}
    for language in sorted(set(languages[split_mask])):
        mask = split_mask & (languages == language)
        if mask.any():
            report[str(language)] = macro_by_profile(scores[mask], labels[mask])
    return report


def main() -> None:
    args = parse_args()
    score_path = args.eval_dir / "style_embedding_scores.npz"
    payload = np.load(score_path, allow_pickle=True)
    reference = {field: payload[field] for field in (
        "chunk_ids", "splits", "query_languages", "query_corpora", "profiles", "y_true"
    )}
    frame = aligned_metadata(args.input, reference)
    df = pd.read_parquet(args.input)
    train = balanced_train(df, args.train_cap, args.seed)
    train["profile_key"] = profile_key(train)
    train["source_key"] = independent_source_keys(train)
    train_embeddings = np.load(args.eval_dir / "style_embedding_train_embeddings.npy")
    eval_embeddings = np.load(args.eval_dir / "style_embedding_eval_embeddings.npy")
    if len(train) != len(train_embeddings) or len(frame) != len(eval_embeddings):
        raise ValueError("Saved embeddings do not align with the reconstructed train/eval rows")
    profiles = reference["profiles"].astype(str)
    profile_index = {profile: index for index, profile in enumerate(profiles)}
    train_labels = np.asarray([profile_index[key] for key in train["profile_key"].astype(str)], dtype=int)
    fractions = [float(value) for value in args.fractions.split(",")]
    if any(value <= 0 or value > 1 for value in fractions):
        raise ValueError("Fractions must be in (0, 1]")
    split_mask = reference["splits"].astype(str) == args.split
    rng = np.random.default_rng(args.seed)
    report: dict[str, object] = {
        "protocol": {
            "intervention": "fraction of independent training sources retained per author-language profile",
            "primary_unit": "independent held-out source",
            "primary_weighting": "macro by author-language profile",
            "chunk_results": "diagnostic only",
            "train_cap_reconstruction": args.train_cap,
            "split": args.split,
        },
        "fractions": {},
    }
    source_counts = train.groupby("profile_key")["source_key"].nunique()
    report["training_source_counts"] = {
        "minimum": int(source_counts.min()),
        "median": float(source_counts.median()),
        "maximum": int(source_counts.max()),
        "profiles_with_one_source": int((source_counts == 1).sum()),
    }
    for fraction in fractions:
        repetitions = 1 if fraction == 1.0 else args.repeats
        rows: list[dict[str, object]] = []
        for _ in range(repetitions):
            selected_positions = []
            selected_source_counts = []
            for profile, positions in train.groupby("profile_key", sort=True).indices.items():
                positions = np.asarray(positions, dtype=int)
                sources = np.unique(train.iloc[positions]["source_key"].astype(str))
                n_selected = max(1, int(np.ceil(len(sources) * fraction)))
                selected_sources = sources if n_selected == len(sources) else rng.choice(
                    sources, size=n_selected, replace=False
                )
                selected_positions.extend(
                    positions[train.iloc[positions]["source_key"].isin(selected_sources).to_numpy()].tolist()
                )
                selected_source_counts.append(n_selected)
            selected_positions_array = np.asarray(selected_positions, dtype=int)
            centroids = []
            for label in range(len(profiles)):
                positions = selected_positions_array[train_labels[selected_positions_array] == label]
                if not len(positions):
                    raise ValueError(f"No selected training evidence for profile: {profiles[label]}")
                centroids.append(train_embeddings[positions].mean(axis=0))
            centroid_matrix = normalize(np.vstack(centroids), norm="l2")
            chunk_scores = eval_embeddings @ centroid_matrix.T
            chunk_scores = mask_cross_language_candidates(
                chunk_scores, pd.Series(reference["query_languages"].astype(str)), profiles
            )
            source_frame, source_labels, source_matrices = aggregate_scores_by_source(
                frame,
                reference["y_true"].astype(int),
                {"centroid": chunk_scores},
            )
            source_split = source_frame["split"].astype(str).to_numpy() == args.split
            rows.append(
                {
                    "source_macro": evaluate_units(
                        source_matrices["centroid"],
                        source_labels,
                        source_frame["language"].astype(str).to_numpy(),
                        source_split,
                    ),
                    "chunk_diagnostic": evaluate_units(
                        chunk_scores,
                        reference["y_true"].astype(int),
                        reference["query_languages"].astype(str),
                        split_mask,
                    ),
                    "median_selected_sources_per_profile": float(np.median(selected_source_counts)),
                }
            )
        report["fractions"][str(fraction)] = summarize(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
