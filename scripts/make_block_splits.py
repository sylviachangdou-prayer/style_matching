from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate chunk-level splits into longer blocks.")
    parser.add_argument("--input", type=Path, required=True, help="Split parquet with text.")
    parser.add_argument("--output", type=Path, required=True, help="Output block split parquet.")
    parser.add_argument("--report", type=Path, required=True, help="Output JSON report.")
    parser.add_argument("--chunks-per-block", type=int, default=3)
    return parser.parse_args()


def validate(df: pd.DataFrame) -> None:
    required = {"chunk_id", "author_or_speaker", "split", "source_id", "title", "text"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")


def make_blocks(df: pd.DataFrame, chunks_per_block: int) -> pd.DataFrame:
    rows = []
    sort_cols = ["author_or_speaker", "split", "source_id", "chunk_id"]
    df = df.sort_values(sort_cols).reset_index(drop=True)

    for (author, split, source_id), group in df.groupby(["author_or_speaker", "split", "source_id"], sort=False):
        group = group.reset_index(drop=True)
        for start in range(0, len(group), chunks_per_block):
            block = group.iloc[start:start + chunks_per_block]
            if len(block) < chunks_per_block:
                continue
            block_id = f"{split}_{source_id}_{start // chunks_per_block:04d}"
            rows.append(
                {
                    "chunk_id": block_id,
                    "author_or_speaker": author,
                    "split": split,
                    "source_id": source_id,
                    "title": block["title"].iloc[0],
                    "word_count": int(block["word_count"].sum()) if "word_count" in block else None,
                    "source_chunk_ids": "|".join(block["chunk_id"].astype(str)),
                    "text": "\n\n".join(block["text"].astype(str)),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    df = pd.read_parquet(args.input)
    validate(df)
    out = make_blocks(df, args.chunks_per_block)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.output, index=False)

    report = {
        "input": str(args.input),
        "output": str(args.output),
        "chunks_per_block": args.chunks_per_block,
        "n_rows": int(len(out)),
        "n_authors": int(out["author_or_speaker"].nunique()) if len(out) else 0,
        "split_counts": out["split"].value_counts().to_dict() if len(out) else {},
        "per_author_min": int(out.groupby("author_or_speaker").size().min()) if len(out) else 0,
        "per_author_max": int(out.groupby("author_or_speaker").size().max()) if len(out) else 0,
    }
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
