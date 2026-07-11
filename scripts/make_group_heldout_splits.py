from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create globally group-heldout authorship splits.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--group-column", choices=["topic", "domain", "register", "decade"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260711)
    parser.add_argument("--min-train", type=int, default=30)
    parser.add_argument("--min-dev", type=int, default=10)
    parser.add_argument("--min-test", type=int, default=10)
    return parser.parse_args()


def assign_group(value: str, seed: int) -> str:
    digest = hashlib.sha256(f"{seed}|{value}".encode()).digest()
    bucket = int.from_bytes(digest[:8], "big") % 10
    return "test" if bucket < 2 else "dev" if bucket < 4 else "train"


def build_split(df: pd.DataFrame, group_column: str, seed: int, minima: dict[str, int]) -> tuple[pd.DataFrame, dict]:
    required = {"language", "author_or_speaker", "text", group_column}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    valid = df[df[group_column].fillna("").astype(str).str.strip().ne("")].copy()
    valid["split"] = valid[group_column].astype(str).map(lambda value: assign_group(value, seed))
    kept = []
    profiles = []
    for key, group in valid.groupby(["language", "author_or_speaker"], sort=True):
        counts = group["split"].value_counts().to_dict()
        eligible = all(counts.get(split, 0) >= minimum for split, minimum in minima.items())
        profiles.append({
            "language": str(key[0]),
            "author_or_speaker": str(key[1]),
            "eligible": eligible,
            **{f"{split}_chunks": int(counts.get(split, 0)) for split in ("train", "dev", "test")},
        })
        if eligible:
            kept.append(group)
    output = pd.concat(kept, ignore_index=True) if kept else valid.iloc[0:0].copy()
    leakage = int(output.groupby(group_column)["split"].nunique().max()) if not output.empty else 0
    report = {
        "group_column": group_column,
        "n_input_rows": int(len(df)),
        "n_rows": int(len(output)),
        "n_profiles": int(sum(row["eligible"] for row in profiles)),
        "global_group_leakage": leakage > 1,
        "profiles": profiles,
    }
    return output, report


def main() -> None:
    args = parse_args()
    df = pd.read_parquet(args.input)
    output, report = build_split(
        df,
        args.group_column,
        args.seed,
        {"train": args.min_train, "dev": args.min_dev, "test": args.min_test},
    )
    if output.empty:
        raise ValueError(f"No profiles eligible for global {args.group_column}-heldout evaluation")
    if report["global_group_leakage"]:
        raise RuntimeError(f"{args.group_column} values cross split boundaries")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    output.to_parquet(args.output, index=False)
    args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("group_column", "n_rows", "n_profiles")}, indent=2))


if __name__ == "__main__":
    main()
