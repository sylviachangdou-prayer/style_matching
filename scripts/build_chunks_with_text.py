from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", choices=["literary", "rhetorical", "both"], default="literary")
    parser.add_argument("--language", help="Keep only one source language, e.g. en/de/fr.")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def read_chunks(corpus: str) -> pd.DataFrame:
    path = ROOT / "data" / corpus / "meta" / "chunks.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    if df.empty:
        return df
    df["text"] = [
        (ROOT / chunk_path).read_text(encoding="utf-8", errors="replace")
        for chunk_path in df["chunk_path"]
    ]
    return df


def main() -> None:
    args = parse_args()
    corpora = ["literary", "rhetorical"] if args.corpus == "both" else [args.corpus]
    frames = [read_chunks(corpus) for corpus in corpora]
    frames = [frame for frame in frames if not frame.empty]
    if not frames:
        raise ValueError("No chunk metadata found.")
    df = pd.concat(frames, ignore_index=True)
    if args.language:
        df = df[df["language"].eq(args.language)].copy()
    if df.empty:
        raise ValueError("No chunks remain after filtering.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.output, index=False)
    print({"output": str(args.output), "shape": df.shape, "authors": int(df["author_or_speaker"].nunique())})


if __name__ == "__main__":
    main()
