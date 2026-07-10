from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create source-heldout splits across literary and rhetorical sources.")
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="chunks_with_text parquet with source_id/title/text columns.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output parquet for source-heldout splits.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        required=True,
        help="Output JSON report.",
    )
    parser.add_argument("--seed", type=int, default=20260701)
    parser.add_argument("--train-cap", type=int, default=300)
    parser.add_argument("--dev-cap", type=int, default=50)
    parser.add_argument("--test-cap", type=int, default=50)
    parser.add_argument("--min-train", type=int, default=30)
    parser.add_argument("--min-dev", type=int, default=10)
    parser.add_argument("--min-test", type=int, default=10)
    return parser.parse_args()


def validate(df: pd.DataFrame) -> None:
    required = {"chunk_id", "author_or_speaker", "source_id", "title", "text", "language", "corpus"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if df["chunk_id"].duplicated().any():
        raise ValueError("chunk_id must be unique")


def cap(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    if len(df) <= n:
        return df
    return df.sample(n=n, random_state=seed)


def split_author(df: pd.DataFrame, seed: int, args: argparse.Namespace) -> tuple[pd.DataFrame | None, dict]:
    author = str(df["author_or_speaker"].iloc[0])
    language = str(df["language"].iloc[0])
    source_key = df["corpus"].astype(str) + "::" + df["source_id"].astype(str)
    source_counts = source_key.groupby(source_key).size().sort_values(ascending=False)
    report = {
        "author": author,
        "language": language,
        "n_sources": int(source_counts.shape[0]),
        "n_chunks": int(len(df)),
        "source_corpora": sorted(df["corpus"].astype(str).unique().tolist()),
        "source_counts": {str(k): int(v) for k, v in source_counts.items()},
        "eligible": False,
        "reason": "",
    }

    if source_counts.shape[0] < 3:
        report["reason"] = "fewer_than_3_sources"
        return None, report

    source_ids = list(source_counts.sample(frac=1, random_state=seed).index)
    test_source = source_ids[0]
    dev_source = source_ids[1]
    train_sources = source_ids[2:]

    train = df[source_key.isin(train_sources)].copy()
    dev = df[source_key == dev_source].copy()
    test = df[source_key == test_source].copy()

    train = cap(train, args.train_cap, seed).assign(split="train")
    dev = cap(dev, args.dev_cap, seed).assign(split="dev")
    test = cap(test, args.test_cap, seed).assign(split="test")

    if len(train) < args.min_train or len(dev) < args.min_dev or len(test) < args.min_test:
        report["reason"] = "insufficient_chunks_after_source_split"
        report["train_chunks"] = int(len(train))
        report["dev_chunks"] = int(len(dev))
        report["test_chunks"] = int(len(test))
        return None, report

    report.update(
        {
            "eligible": True,
            "reason": "ok",
            "train_sources": [str(s) for s in train_sources],
            "dev_source": str(dev_source),
            "test_source": str(test_source),
            "train_corpora": sorted(df[source_key.isin(train_sources)]["corpus"].astype(str).unique().tolist()),
            "dev_corpus": str(df.loc[source_key == dev_source, "corpus"].iloc[0]),
            "test_corpus": str(df.loc[source_key == test_source, "corpus"].iloc[0]),
            "train_chunks": int(len(train)),
            "dev_chunks": int(len(dev)),
            "test_chunks": int(len(test)),
        }
    )
    return pd.concat([train, dev, test], ignore_index=True), report


def main() -> None:
    args = parse_args()
    df = pd.read_parquet(args.input)
    validate(df)

    splits = []
    author_reports = []
    for index, (_, author_df) in enumerate(df.groupby(["language", "author_or_speaker"])):
        split, report = split_author(author_df, args.seed + index, args)
        author_reports.append(report)
        if split is not None:
            splits.append(split)

    if not splits:
        raise ValueError("No authors eligible for source-heldout split")

    out = pd.concat(splits, ignore_index=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.output, index=False)

    report = {
        "input": str(args.input),
        "output": str(args.output),
        "n_rows": int(len(out)),
        "n_authors": int(out["author_or_speaker"].nunique()),
        "n_author_language_profiles": int(out[["language", "author_or_speaker"]].drop_duplicates().shape[0]),
        "n_sources": int(out[["language", "corpus", "source_id"]].drop_duplicates().shape[0]),
        "split_counts": out["split"].value_counts().to_dict(),
        "eligible_authors": int(sum(r["eligible"] for r in author_reports)),
        "excluded_authors": int(sum(not r["eligible"] for r in author_reports)),
        "authors": author_reports,
    }
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
