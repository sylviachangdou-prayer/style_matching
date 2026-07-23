#!/usr/bin/env python3
"""Attach a gated within-language hubness correction to an existing index."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    report = json.loads(args.report.read_text(encoding="utf-8"))
    selected = report["selected"]
    metadata_path = args.index_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if selected == "uncorrected":
        metadata.update({
            "hubness_correction": "none",
            "hubness_gate_passed": False,
        })
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        print("No hubness correction passed the locked adoption gate.")
        return

    result = report["methods"][selected]
    if not result.get("adopted"):
        raise ValueError(f"Selected correction did not pass its gate: {selected}")
    profiles = pd.read_parquet(args.index_dir / "profiles.parquet")
    bias_by_key = result["all_dev_candidates_bias"]
    keys = (
        profiles["language"].astype(str)
        + "::"
        + profiles["author_or_speaker"].astype(str)
    )
    missing = sorted(set(keys) - set(bias_by_key))
    if missing:
        raise ValueError(f"Hubness report does not cover {len(missing)} index profiles")
    bias = np.asarray([bias_by_key[key] for key in keys], dtype="float32")
    np.save(args.index_dir / "hubness_bias.npy", bias)
    metadata.update({
        "hubness_correction": selected,
        "hubness_lambda": float(result["selected_lambda"]),
        "hubness_gate_passed": True,
        "hubness_within_language_only": True,
        "hubness_support_balanced": bool(report.get("support_balanced")),
    })
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps({
        "index": str(args.index_dir),
        "method": selected,
        "lambda": result["selected_lambda"],
        "profiles": len(profiles),
    }, indent=2))


if __name__ == "__main__":
    main()
