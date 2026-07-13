from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.retrieval_metrics import ranks_from_scores
from scripts.score_artifact_utils import (
    aggregate_scores_by_source,
    aligned_metadata,
    candidate_mask,
    load_aligned_scores,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare languages at matched candidate-set sizes using independent sources."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--scores", action="append", required=True, help="NAME=NPZ_PATH:ARRAY_KEY")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--candidate-sizes", default="5,10,20,40,all")
    parser.add_argument("--repeats", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260713)
    return parser.parse_args()


def harmonic_number(n: int) -> float:
    return float(np.sum(1.0 / np.arange(1, n + 1)))


def author_macro_metrics(scores: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    ranks = ranks_from_scores(scores, labels)
    per_author = []
    for label in np.unique(labels):
        selected = ranks[labels == label]
        per_author.append(
            [
                np.mean(selected <= 1),
                np.mean(selected <= 3),
                np.mean(selected <= 5),
                np.mean(1.0 / selected),
            ]
        )
    values = np.asarray(per_author, dtype=float).mean(axis=0)
    return {
        "recall_at_1": float(values[0]),
        "recall_at_3": float(values[1]),
        "recall_at_5": float(values[2]),
        "mrr": float(values[3]),
    }


def chance_adjust(metrics: dict[str, float], n_candidates: int) -> dict[str, float]:
    chance = {
        "recall_at_1": 1.0 / n_candidates,
        "recall_at_3": min(3.0 / n_candidates, 1.0),
        "recall_at_5": min(5.0 / n_candidates, 1.0),
        "mrr": harmonic_number(n_candidates) / n_candidates,
    }
    adjusted = {
        name: (metrics[name] - value) / max(1.0 - value, 1e-12)
        for name, value in chance.items()
    }
    return {"chance": chance, "chance_adjusted": adjusted}


def summarize_repeats(rows: list[dict[str, float]]) -> dict[str, object]:
    output: dict[str, object] = {"n_repeats": len(rows)}
    for family in ("observed", "chance", "chance_adjusted"):
        output[family] = {}
        for metric in ("recall_at_1", "recall_at_3", "recall_at_5", "mrr"):
            values = np.asarray([row[f"{family}.{metric}"] for row in rows])
            output[family][metric] = {
                "mean": float(values.mean()),
                "ci_low": float(np.quantile(values, 0.025)),
                "ci_high": float(np.quantile(values, 0.975)),
            }
    return output


def parse_candidate_sizes(value: str, native: int) -> list[int]:
    sizes = []
    for token in value.split(","):
        token = token.strip().lower()
        size = native if token == "all" else int(token)
        if 2 <= size <= native:
            sizes.append(size)
    return sorted(set(sizes))


def evaluate_sample(
    scores: np.ndarray,
    labels: np.ndarray,
    selected_candidates: np.ndarray,
) -> dict[str, float]:
    query_mask = np.isin(labels, selected_candidates)
    if not query_mask.any():
        raise ValueError("Candidate sample has no source queries")
    label_map = {int(label): index for index, label in enumerate(selected_candidates)}
    local_labels = np.asarray([label_map[int(label)] for label in labels[query_mask]], dtype=int)
    observed = author_macro_metrics(scores[query_mask][:, selected_candidates], local_labels)
    adjustment = chance_adjust(observed, len(selected_candidates))
    return {
        **{f"observed.{key}": value for key, value in observed.items()},
        **{f"chance.{key}": value for key, value in adjustment["chance"].items()},
        **{
            f"chance_adjusted.{key}": value
            for key, value in adjustment["chance_adjusted"].items()
        },
    }


def main() -> None:
    args = parse_args()
    matrices, reference = load_aligned_scores(args.scores)
    frame = aligned_metadata(args.input, reference)
    source_frame, source_labels, source_matrices = aggregate_scores_by_source(
        frame, reference["y_true"].astype(int), matrices
    )
    profiles = reference["profiles"].astype(str)
    report: dict[str, object] = {
        "protocol": {
            "evaluation_unit": "independent_source",
            "aggregation": "mean chunk score within source",
            "author_weighting": "macro",
            "split": args.split,
            "candidate_sampling_repeats": args.repeats,
            "interpretation": "fixed-N and chance-adjusted estimates separate candidate-pool difficulty from model quality",
        },
        "models": {},
    }
    for model_name, matrix in source_matrices.items():
        model_report: dict[str, object] = {}
        for language in sorted(source_frame["language"].unique()):
            candidate_indices = np.flatnonzero(candidate_mask(profiles, language))
            query_mask = source_frame["split"].eq(args.split).to_numpy() & source_frame[
                "language"
            ].eq(language).to_numpy()
            eligible_labels = np.intersect1d(candidate_indices, np.unique(source_labels[query_mask]))
            native = len(eligible_labels)
            if native < 2:
                continue
            language_scores = matrix[query_mask]
            language_labels = source_labels[query_mask]
            size_report: dict[str, object] = {}
            for size in parse_candidate_sizes(args.candidate_sizes, native):
                repetitions = 1 if size == native else args.repeats
                rows = []
                language_seed = args.seed + size * 1009 + sum(
                    (index + 1) * ord(character) for index, character in enumerate(str(language))
                )
                rng = np.random.default_rng(language_seed)
                for _ in range(repetitions):
                    selected = (
                        np.sort(eligible_labels)
                        if size == native
                        else np.sort(rng.choice(eligible_labels, size=size, replace=False))
                    )
                    rows.append(evaluate_sample(language_scores, language_labels, selected))
                size_report[str(size)] = summarize_repeats(rows)
            model_report[str(language)] = {
                "n_native_candidates": native,
                "n_test_sources": int(query_mask.sum()),
                "candidate_sizes": size_report,
            }
        report["models"][model_name] = model_report
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
