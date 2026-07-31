#!/usr/bin/env python3
"""Compare two indexes on the same frozen query embeddings."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--eval-embeddings", type=Path, required=True)
    parser.add_argument("--eval-chunk-ids", type=Path, required=True)
    parser.add_argument("--old-index", type=Path, required=True)
    parser.add_argument("--new-index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-runs", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260731)
    return parser.parse_args()


def load_index(path: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    profiles = pd.read_parquet(path / "profiles.parquet").sort_values("profile_id")
    keys = (
        profiles["language"].astype(str) + "::" + profiles["author_or_speaker"].astype(str)
    ).to_numpy()
    centroids = np.load(path / "centroids.npy").astype("float64")
    metadata = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
    if len(keys) != len(centroids):
        raise ValueError(f"Profile/centroid mismatch: {path}")
    return keys, centroids, metadata


def model_label(metadata: dict) -> str | None:
    return metadata.get("model_label") or metadata.get("selection_decision")


def rank_values(
    embeddings: np.ndarray,
    languages: np.ndarray,
    true_keys: np.ndarray,
    keys: np.ndarray,
    centroids: np.ndarray,
) -> np.ndarray:
    scores = embeddings @ centroids.T
    candidate_languages = np.asarray([key.split("::", 1)[0] for key in keys])
    scores[languages[:, None] != candidate_languages[None, :]] = -np.inf
    lookup = {key: position for position, key in enumerate(keys)}
    ranks = np.full(len(embeddings), np.inf)
    for row, true_key in enumerate(true_keys):
        true = lookup.get(str(true_key))
        if true is None:
            continue
        ranks[row] = 1 + int(np.sum(scores[row] > scores[row, true]))
    return ranks


def metric(values: np.ndarray) -> float:
    return float(values.mean()) if len(values) else float("nan")


def paired_profile_bootstrap(
    old: np.ndarray,
    new: np.ndarray,
    profiles: np.ndarray,
    runs: int,
    seed: int,
) -> dict[str, float]:
    unique = np.unique(profiles)
    old_profile = np.asarray([old[profiles == profile].mean() for profile in unique])
    new_profile = np.asarray([new[profiles == profile].mean() for profile in unique])
    deltas = new_profile - old_profile
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(deltas), (runs, len(deltas)))
    old_samples = old_profile[draws].mean(axis=1)
    new_samples = new_profile[draws].mean(axis=1)
    samples = new_samples - old_samples
    return {
        "old_ci_low": float(np.quantile(old_samples, 0.025)),
        "old_ci_high": float(np.quantile(old_samples, 0.975)),
        "new_ci_low": float(np.quantile(new_samples, 0.025)),
        "new_ci_high": float(np.quantile(new_samples, 0.975)),
        "delta": float(deltas.mean()),
        "ci_low": float(np.quantile(samples, 0.025)),
        "ci_high": float(np.quantile(samples, 0.975)),
        "bootstrap_unit": "author-language profile",
    }


def comparison(
    frame: pd.DataFrame,
    embeddings: np.ndarray,
    old_keys: np.ndarray,
    old_centroids: np.ndarray,
    new_keys: np.ndarray,
    new_centroids: np.ndarray,
    runs: int,
    seed: int,
) -> dict:
    languages = frame["language"].astype(str).to_numpy()
    true_keys = (
        frame["language"].astype(str) + "::" + frame["author_or_speaker"].astype(str)
    ).to_numpy()
    old_ranks = rank_values(embeddings, languages, true_keys, old_keys, old_centroids)
    new_ranks = rank_values(embeddings, languages, true_keys, new_keys, new_centroids)
    old_present = np.isin(true_keys, old_keys)
    new_present = np.isin(true_keys, new_keys)
    old_values = {"mrr": 1.0 / old_ranks}
    new_values = {"mrr": 1.0 / new_ranks}
    for cutoff in (1, 3, 5, 20):
        old_values[f"recall_at_{cutoff}"] = (old_ranks <= cutoff).astype("float64")
        new_values[f"recall_at_{cutoff}"] = (new_ranks <= cutoff).astype("float64")
    result = {
        "n_queries": int(len(frame)),
        "n_true_profiles": int(len(np.unique(true_keys))),
        "old_query_coverage": metric(old_present),
        "new_query_coverage": metric(new_present),
        "old_true_profiles_covered": int(len(np.unique(true_keys[old_present]))),
        "new_true_profiles_covered": int(len(np.unique(true_keys[new_present]))),
        "old": {name: metric(values) for name, values in old_values.items()},
        "new": {name: metric(values) for name, values in new_values.items()},
    }
    for position, name in enumerate(old_values):
        result[f"{name}_delta"] = paired_profile_bootstrap(
            old_values[name], new_values[name], true_keys, runs, seed + position
        )
    return result


def main() -> None:
    args = parse_args()
    frame = pd.read_parquet(args.input)
    frame = frame[frame["split"].eq("test")].copy().reset_index(drop=True)
    # The evaluation writer stores variable-length chunk IDs as an object array.
    # This is a trusted, locally generated artifact; numeric arrays remain loaded
    # with NumPy's default allow_pickle=False.
    chunk_ids = np.load(args.eval_chunk_ids, allow_pickle=True).astype(str)
    embeddings = np.load(args.eval_embeddings).astype("float64")
    if len(chunk_ids) != len(embeddings):
        raise ValueError("Evaluation chunk IDs and embeddings do not align")
    embedding_lookup = {chunk_id: row for row, chunk_id in enumerate(chunk_ids)}
    missing = sorted(set(frame["chunk_id"].astype(str)) - set(embedding_lookup))
    if missing:
        raise ValueError(f"Missing test embeddings, first IDs: {missing[:5]}")
    embeddings = embeddings[[embedding_lookup[value] for value in frame["chunk_id"].astype(str)]]

    old_keys, old_centroids, old_metadata = load_index(args.old_index)
    new_keys, new_centroids, new_metadata = load_index(args.new_index)
    if old_centroids.shape[1] != new_centroids.shape[1] or embeddings.shape[1] != old_centroids.shape[1]:
        raise ValueError("Embedding dimensions differ; indexes are not directly comparable")
    old_label = model_label(old_metadata)
    new_label = model_label(new_metadata)
    if old_label and new_label and old_label != new_label:
        raise ValueError(
            f"Indexes use different encoders ({old_label!r} versus {new_label!r}); "
            "their centroids cannot be compared with one frozen query embedding set"
        )

    shared = np.intersect1d(old_keys, new_keys)
    true_keys = frame["language"].astype(str) + "::" + frame["author_or_speaker"].astype(str)
    shared_mask = true_keys.isin(shared).to_numpy()
    old_shared_positions = np.asarray([np.flatnonzero(old_keys == key)[0] for key in shared])
    new_shared_positions = np.asarray([np.flatnonzero(new_keys == key)[0] for key in shared])

    expanded = comparison(
        frame, embeddings, old_keys, old_centroids, new_keys, new_centroids,
        args.bootstrap_runs, args.seed,
    )
    shared_operational = comparison(
        frame[shared_mask].reset_index(drop=True), embeddings[shared_mask],
        old_keys, old_centroids, new_keys, new_centroids,
        args.bootstrap_runs, args.seed + 2,
    )
    shared_candidates = comparison(
        frame[shared_mask].reset_index(drop=True), embeddings[shared_mask],
        shared, old_centroids[old_shared_positions], shared, new_centroids[new_shared_positions],
        args.bootstrap_runs, args.seed + 4,
    )

    group_reports = {"language": {}, "corpus": {}}
    for column in group_reports:
        for group_position, (value, subset) in enumerate(frame.groupby(column, sort=True)):
            positions = subset.index.to_numpy()
            local_shared = shared_mask[positions]
            if not local_shared.any():
                continue
            shared_positions = positions[local_shared]
            group_reports[column][str(value)] = comparison(
                frame.iloc[shared_positions].reset_index(drop=True),
                embeddings[shared_positions],
                old_keys, old_centroids, new_keys, new_centroids,
                args.bootstrap_runs, args.seed + 10 + group_position,
            )

    report = {
        "design": "same frozen test queries and embeddings; missing true profiles receive reciprocal rank zero",
        "old_index": {"path": str(args.old_index), "metadata": old_metadata},
        "new_index": {"path": str(args.new_index), "metadata": new_metadata},
        "candidate_profiles": {
            "old": int(len(old_keys)), "new": int(len(new_keys)), "shared": int(len(shared))
        },
        "encoder_label": old_label or new_label,
        "expanded_coverage_adjusted": expanded,
        "shared_true_full_candidate_universe": shared_operational,
        "shared_true_shared_candidate_control": shared_candidates,
        "shared_true_subgroups": group_reports,
    }
    expanded_mrr = expanded["mrr_delta"]
    shared_mrr = shared_operational["mrr_delta"]
    shared_r3 = shared_operational["recall_at_3_delta"]
    supported_groups = [
        result
        for groups in group_reports.values()
        for result in groups.values()
        if result["n_true_profiles"] >= 5
    ]
    report["new_index_better_gate"] = {
        "expanded_mrr_ci_wholly_positive": expanded_mrr["ci_low"] > 0,
        "shared_author_mrr_not_materially_worse": (
            shared_mrr["delta"] >= -0.01 and shared_mrr["ci_high"] >= 0
        ),
        "shared_author_recall_at_3_not_materially_worse": shared_r3["delta"] >= -0.01,
        "supported_subgroups_not_materially_worse": all(
            result["mrr_delta"]["delta"] >= -0.02
            and result["mrr_delta"]["ci_high"] >= 0
            for result in supported_groups
        ),
    }
    report["new_index_better"] = all(report["new_index_better_gate"].values())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({
        "candidate_profiles": report["candidate_profiles"],
        "expanded_coverage_adjusted": expanded,
        "shared_true_full_candidate_universe": shared_operational,
        "shared_true_shared_candidate_control": shared_candidates,
        "new_index_better_gate": report["new_index_better_gate"],
        "new_index_better": report["new_index_better"],
    }, indent=2))
    print(f"RETURN: {args.output}")


if __name__ == "__main__":
    main()
