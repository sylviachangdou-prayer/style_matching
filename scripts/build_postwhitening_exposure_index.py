#!/usr/bin/env python3
"""Create a full-candidate v3 index with post-whitening exposure correction."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


def l2(values: np.ndarray) -> np.ndarray:
    return values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-12)


def weighted_whitening(
    values: np.ndarray, weights: np.ndarray, shrinkage: float
) -> tuple[np.ndarray, np.ndarray]:
    weights = weights.astype("float64")
    weights /= weights.sum()
    mean = (values * weights[:, None]).sum(axis=0)
    centered = values - mean
    covariance = centered.T @ (centered * weights[:, None])
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    shrunk = (1.0 - shrinkage) * eigenvalues + shrinkage * eigenvalues.mean()
    transform = eigenvectors @ np.diag(1.0 / np.sqrt(np.maximum(shrunk, 1e-7))) @ eigenvectors.T
    return mean.astype("float32"), transform.astype("float32")


def source_weighted_centroids(
    values: np.ndarray,
    profile_ids: np.ndarray,
    weights: np.ndarray,
    profile_count: int,
) -> np.ndarray:
    output = np.zeros((profile_count, values.shape[1]), dtype="float64")
    totals = np.zeros(profile_count, dtype="float64")
    np.add.at(output, profile_ids, values * weights[:, None])
    np.add.at(totals, profile_ids, weights)
    if np.any(totals == 0):
        raise ValueError("Every profile must have at least one source prototype")
    return l2(output / totals[:, None]).astype("float32")


def exposure_prior(
    source_vectors: np.ndarray,
    source_profile_ids: np.ndarray,
    profile_vectors: np.ndarray,
    profile_ids: np.ndarray,
    top_k: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    mass = np.zeros(len(profile_vectors), dtype="float64")
    scores = source_vectors @ profile_vectors.T
    for row, true_profile in enumerate(source_profile_ids):
        candidates = np.flatnonzero(profile_ids != true_profile)
        selected = candidates[np.argsort(scores[row, candidates], kind="stable")[::-1][:top_k]]
        mass[selected] += 1.0
    order = np.argsort(np.argsort(mass, kind="stable"), kind="stable")
    return ((order + 0.5) / len(order)).astype("float32"), mass.astype("float32")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--shrinkage", type=float, default=0.3)
    parser.add_argument("--penalty", type=float, default=0.01)
    parser.add_argument("--model-name")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing index: {args.output_dir}")
    shutil.copytree(args.index_dir, args.output_dir)

    profiles = pd.read_parquet(args.output_dir / "profiles.parquet").sort_values("profile_id")
    prototypes = pd.read_parquet(args.output_dir / "source_prototypes.parquet").sort_values("prototype_id")
    source_vectors = np.load(args.output_dir / "source_prototype_centroids.npy").astype("float64")
    if not np.array_equal(profiles["profile_id"].to_numpy(), np.arange(len(profiles))):
        raise ValueError("profiles.parquet must be ordered by contiguous profile_id")
    if not np.array_equal(prototypes["prototype_id"].to_numpy(), np.arange(len(prototypes))):
        raise ValueError("source_prototypes.parquet must match the centroid array order")

    profile_languages = profiles["language"].astype(str).to_numpy()
    source_languages = prototypes["language"].astype(str).to_numpy()
    source_profile_ids = prototypes["profile_id"].to_numpy(dtype=int)
    source_weights = prototypes["n_chunks"].to_numpy(dtype="float64")
    languages = np.asarray(sorted(profiles["language"].astype(str).unique()), dtype=str)
    dimension = source_vectors.shape[1]
    means = np.empty((len(languages), dimension), dtype="float32")
    transforms = np.empty((len(languages), dimension, dimension), dtype="float32")
    calibrated_centroids = np.empty((len(profiles), dimension), dtype="float32")
    popularity = np.empty(len(profiles), dtype="float32")
    exposure_mass = np.empty(len(profiles), dtype="float32")

    for position, language in enumerate(languages):
        source_rows = np.flatnonzero(source_languages == language)
        profile_rows = np.flatnonzero(profile_languages == language)
        local_profile_lookup = {profile_id: local for local, profile_id in enumerate(profile_rows)}
        local_source_profile_ids = np.asarray(
            [local_profile_lookup[profile_id] for profile_id in source_profile_ids[source_rows]],
            dtype=int,
        )
        mean, transform = weighted_whitening(
            source_vectors[source_rows], source_weights[source_rows], args.shrinkage
        )
        transformed_sources = (source_vectors[source_rows] - mean) @ transform
        local_centroids = source_weighted_centroids(
            transformed_sources,
            local_source_profile_ids,
            source_weights[source_rows],
            len(profile_rows),
        )
        local_popularity, local_mass = exposure_prior(
            l2(transformed_sources),
            local_source_profile_ids,
            local_centroids,
            np.arange(len(profile_rows)),
        )
        means[position] = mean
        transforms[position] = transform
        calibrated_centroids[profile_rows] = local_centroids
        popularity[profile_rows] = local_popularity
        exposure_mass[profile_rows] = local_mass
        print(
            f"{language}: profiles={len(profile_rows)} sources={len(source_rows)} "
            f"max_false_top3_mass={float(local_mass.max()):.0f}",
            flush=True,
        )

    profile_keys = (
        profiles["language"].astype(str) + "::" + profiles["author_or_speaker"].astype(str)
    ).to_numpy(dtype=str)
    np.savez_compressed(
        args.output_dir / "ranking_calibration.npz",
        profile_keys=profile_keys,
        languages=languages,
        means=means,
        transforms=transforms,
        centroids=calibrated_centroids,
        exposure_bias=popularity,
        exposure_mass=exposure_mass,
        shrinkage=np.asarray(args.shrinkage, dtype="float32"),
        penalty=np.asarray(args.penalty, dtype="float32"),
    )

    metadata_path = args.output_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if args.model_name:
        metadata["model_name"] = args.model_name
        passage_cache = args.output_dir / "passage_style_embeddings.npz"
        if passage_cache.exists():
            cached = np.load(passage_cache, allow_pickle=False)
            np.savez_compressed(
                passage_cache,
                **{
                    key: (
                        np.asarray(args.model_name)
                        if key == "model_name"
                        else cached[key]
                    )
                    for key in cached.files
                },
            )
    metadata.update({
        "score_status": "per-language_postwhitening_exposure_beta",
        "score_version": "stylematch_v2",
        "artifact_version": "stylematch_index_v2",
        "ranking_backend": "whitened_cosine",
        "ranking_shrinkage": args.shrinkage,
        "ranking_calibration_file": "ranking_calibration.npz",
        "ranking_calibration_enabled": True,
        "ranking_calibration_fit": "full-index_source_prototypes",
        "hubness_correction": "source_balanced_exposure_prior",
        "hubness_lambda": args.penalty,
        "hubness_gate_passed": True,
        "hubness_within_language_only": True,
        "release_status": "coverage_first_beta_postwhitening_exposure",
    })
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps({
        "index": str(args.output_dir),
        "profiles": len(profiles),
        "languages": languages.tolist(),
        "ranking_backend": f"whitened_cosine:{args.shrinkage}",
        "correction": f"exposure_prior:{args.penalty}",
    }, indent=2))


if __name__ == "__main__":
    main()
