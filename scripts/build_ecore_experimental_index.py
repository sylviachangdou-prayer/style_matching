"""Attach an experimental ECoRe scorer to a completed StyleMatch index."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-index", type=Path, required=True)
    parser.add_argument("--scorer", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scorer = json.loads(args.scorer.read_text(encoding="utf-8"))
    if scorer["environment_policy"] != "language_only":
        raise ValueError("Production queries have no reliable register; scorer must use language_only")
    if args.output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(args.output_dir)
        shutil.rmtree(args.output_dir)
    shutil.copytree(args.base_index, args.output_dir)

    prototypes = pd.read_parquet(args.output_dir / "source_prototypes.parquet")
    vectors = np.load(args.output_dir / "source_prototype_centroids.npy")
    centre_rows, centre_vectors = [], []
    for centre_id, (language, positions) in enumerate(
        prototypes.groupby("language", sort=True).indices.items()
    ):
        centre = normalize(vectors[np.asarray(positions)].mean(axis=0, keepdims=True))[0]
        centre_rows.append({
            "centre_id": centre_id,
            "environment": f"{language}::__fallback__",
            "language": str(language),
            "n_prototypes": int(len(positions)),
        })
        centre_vectors.append(centre)
    np.save(
        args.output_dir / "ecore_cohort_centres.npy",
        np.vstack(centre_vectors).astype("float32"),
    )
    pd.DataFrame(centre_rows).to_parquet(
        args.output_dir / "ecore_cohort_centres.parquet", index=False
    )
    shutil.copy2(args.scorer, args.output_dir / "ecore_scorer.json")

    metadata_path = args.output_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata.update({
        "profile_strategy": "ecore_episodic_linear",
        "score_status": "experimental_uncalibrated_energy",
        "artifact_version": "ecore_episodic_challenger_v1",
        "ecore_experimental": True,
        "ecore_within_language_only": True,
        "ecore_scorer_version": scorer["scorer_version"],
        "ecore_feature_names": scorer["feature_names"],
        "ecore_temperature": scorer["temperature"],
        "ecore_environment_policy": scorer["environment_policy"],
        "ecore_deployment_gate_passed": scorer["deployment_gate_passed"],
        "ecore_direct_delta_vs_centroid": scorer["direct_deployment_contrast_vs_centroid"],
        "ecore_scorer_sha256": hashlib.sha256(args.scorer.read_bytes()).hexdigest(),
        "open_set_calibration": {},
        "calibrated": False,
    })
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps({
        "output_dir": str(args.output_dir),
        "status": metadata["score_status"],
        "deployment_gate_passed": scorer["deployment_gate_passed"],
        "languages": metadata.get("languages", []),
    }, indent=2))


if __name__ == "__main__":
    main()
