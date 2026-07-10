"""Lightweight GPU fine-tuning for the multilingual style encoder.

This script uses only original-language corpus text.  Positive pairs are two
different sources by the same author, so the model cannot solve the pair by
memorising one book.  Multiple-negatives ranking supplies in-batch negatives.
The script is intentionally a Colab-stage job; the web app consumes the saved
encoder and never trains during a request.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune mStyleDistance on source-separated author pairs.")
    parser.add_argument("--input", type=Path, required=True, help="Chunk parquet with text, author, language, source_id.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-name", default="StyleDistance/mstyledistance")
    parser.add_argument("--split", default="train")
    parser.add_argument("--pairs-per-author", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=20260710)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def make_pairs(df: pd.DataFrame, pairs_per_author: int, seed: int) -> list[tuple[str, str]]:
    required = {"author_or_speaker", "language", "corpus", "source_id", "text"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    rng = random.Random(seed)
    pairs: list[tuple[str, str]] = []
    for (language, author), group in df.groupby(["language", "author_or_speaker"]):
        by_source = {
            source_key: rows["text"].dropna().astype(str).tolist()
            for source_key, rows in group.groupby(group["corpus"].astype(str) + "::" + group["source_id"].astype(str))
        }
        sources = [source for source, texts in by_source.items() if texts]
        if len(sources) < 2:
            continue
        for _ in range(pairs_per_author):
            left_source, right_source = rng.sample(sources, 2)
            pairs.append((rng.choice(by_source[left_source]), rng.choice(by_source[right_source])))

    rng.shuffle(pairs)
    return pairs


def main() -> None:
    args = parse_args()
    df = pd.read_parquet(args.input)
    if "split" in df.columns:
        df = df[df["split"].eq(args.split)].copy()
    pairs = make_pairs(df, args.pairs_per_author, args.seed)
    if len(pairs) < 2:
        raise ValueError("Need at least two source-separated positive pairs; add sources before fine-tuning.")

    from sentence_transformers import InputExample, SentenceTransformer, losses
    from sentence_transformers.datasets import NoDuplicatesDataLoader

    model = SentenceTransformer(args.model_name, device=args.device)
    examples = [InputExample(texts=[left, right]) for left, right in pairs]
    loader = NoDuplicatesDataLoader(examples, batch_size=args.batch_size)
    loss = losses.MultipleNegativesRankingLoss(model)
    warmup_steps = max(1, int(len(loader) * args.epochs * args.warmup_ratio))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model.fit(
        train_objectives=[(loader, loss)],
        epochs=args.epochs,
        warmup_steps=warmup_steps,
        optimizer_params={"lr": args.learning_rate},
        output_path=str(args.output_dir),
        show_progress_bar=True,
    )
    report = {
        "input": str(args.input),
        "base_model": args.model_name,
        "output_dir": str(args.output_dir),
        "split": args.split,
        "n_rows": int(len(df)),
        "n_pairs": int(len(pairs)),
        "n_author_language_profiles_with_cross_source_pairs": int(
            df.assign(source_key=df["corpus"].astype(str) + "::" + df["source_id"].astype(str))
            .groupby(["language", "author_or_speaker"])["source_key"].nunique().ge(2).sum()
        ),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "seed": args.seed,
        "original_language_only": True,
        "translation_used": False,
    }
    (args.output_dir / "training_config.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
