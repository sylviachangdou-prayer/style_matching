"""Lightweight GPU fine-tuning for the multilingual style encoder.

This script uses only original-language corpus text.  Positive pairs are two
different sources by the same author, so the model cannot solve the pair by
memorising one book.  Multiple-negatives ranking supplies in-batch negatives.
The script is intentionally a Colab-stage job; the web app consumes the saved
encoder and never trains during a request.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def independent_source_keys(df: pd.DataFrame) -> pd.Series:
    identity = df["independent_source_id"] if "independent_source_id" in df else df["source_id"]
    return df["corpus"].astype(str) + "::" + identity.fillna("").astype(str)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune mStyleDistance on source-separated author pairs.")
    parser.add_argument("--input", type=Path, required=True, help="Chunk parquet with text, author, language, source_id.")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--model-name", default="StyleDistance/mstyledistance")
    parser.add_argument("--split", default="train")
    parser.add_argument("--pairs-per-author", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=20260710)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--registry", type=Path, default=ROOT / "data/source_registry/all_people.csv")
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--language-aware-batches", action="store_true")
    parser.add_argument("--hard-negatives", action="store_true")
    return parser.parse_args()


def training_coverage(df: pd.DataFrame, registry_path: Path) -> dict:
    source_keys = independent_source_keys(df)
    counts = (
        df.assign(source_key=source_keys)
        .groupby(["language", "author_or_speaker"])
        .agg(n_sources=("source_key", "nunique"), n_chunks=("text", "size"))
        .reset_index()
    )
    observed = {
        (str(row.language), str(row.author_or_speaker)): {
            "language": str(row.language),
            "author_or_speaker": str(row.author_or_speaker),
            "n_sources": int(row.n_sources),
            "n_chunks": int(row.n_chunks),
        }
        for row in counts.itertuples(index=False)
    }
    with registry_path.open(newline="", encoding="utf-8") as handle:
        registry = list(csv.DictReader(handle))
    expected = {(row["original_language"], row["name"]) for row in registry}
    eligible = sorted(key for key, value in observed.items() if value["n_sources"] >= 2)
    heldout_ready = sorted(key for key, value in observed.items() if value["n_sources"] >= 3)
    missing = sorted(expected - observed.keys())
    one_source = sorted(key for key, value in observed.items() if value["n_sources"] == 1)
    unexpected = sorted(observed.keys() - expected)
    return {
        "n_registry_author_language_profiles": len(expected),
        "n_profiles_with_chunks": len(observed),
        "n_profiles_eligible_for_finetuning": len(eligible),
        "n_profiles_with_at_least_3_sources": len(heldout_ready),
        "eligible_for_finetuning": [observed[key] for key in eligible],
        "profiles_with_one_source": [observed[key] for key in one_source],
        "registry_profiles_without_chunks": [
            {"language": language, "author_or_speaker": author} for language, author in missing
        ],
        "chunk_profiles_not_in_registry": [
            {"language": language, "author_or_speaker": author} for language, author in unexpected
        ],
    }


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
            for source_key, rows in group.groupby(independent_source_keys(group))
        }
        sources = [source for source, texts in by_source.items() if texts]
        if len(sources) < 2:
            continue
        for _ in range(pairs_per_author):
            left_source, right_source = rng.sample(sources, 2)
            pairs.append((rng.choice(by_source[left_source]), rng.choice(by_source[right_source])))

    rng.shuffle(pairs)
    return pairs


def make_hard_negative_examples(
    df: pd.DataFrame, pairs_per_author: int, seed: int
) -> tuple[dict[str, list[tuple[str, str, str]]], int]:
    rng = random.Random(seed)
    output: dict[str, list[tuple[str, str, str]]] = {}
    relaxed_negatives = 0
    for (language, author), group in df.groupby(["language", "author_or_speaker"], sort=True):
        source_key = independent_source_keys(group)
        sources = {key: rows for key, rows in group.groupby(source_key)}
        if len(sources) < 2:
            continue
        same_language = df[
            df["language"].eq(language) & ~df["author_or_speaker"].eq(author)
        ]
        fallback = df[~df["author_or_speaker"].eq(author)]
        if fallback.empty:
            continue
        examples = output.setdefault(str(language), [])
        for _ in range(pairs_per_author):
            left_source, right_source = rng.sample(list(sources), 2)
            anchor = sources[left_source].sample(n=1, random_state=rng.randrange(2**31)).iloc[0]
            positive = sources[right_source].sample(n=1, random_state=rng.randrange(2**31)).iloc[0]
            candidates = same_language
            for column in ("corpus", "topic", "decade"):
                value = str(anchor.get(column, ""))
                if value and column in candidates.columns:
                    narrowed = candidates[candidates[column].fillna("").astype(str).eq(value)]
                    if not narrowed.empty:
                        candidates = narrowed
            if candidates.empty:
                candidates = fallback
                relaxed_negatives += 1
            negative = candidates.sample(n=1, random_state=rng.randrange(2**31)).iloc[0]
            examples.append((str(anchor["text"]), str(positive["text"]), str(negative["text"])))
    for examples in output.values():
        rng.shuffle(examples)
    return output, relaxed_negatives


def main() -> None:
    args = parse_args()
    df = pd.read_parquet(args.input)
    if "split" in df.columns:
        df = df[df["split"].eq(args.split)].copy()
    coverage = training_coverage(df, args.registry)
    if args.audit_only:
        print(json.dumps(coverage, indent=2, ensure_ascii=False))
        return
    if args.output_dir is None:
        raise ValueError("--output-dir is required unless --audit-only is used")
    if args.hard_negatives:
        examples_by_language, relaxed_negatives = make_hard_negative_examples(
            df, args.pairs_per_author, args.seed
        )
    else:
        examples_by_language = {
            str(language): make_pairs(group, args.pairs_per_author, args.seed + index)
            for index, (language, group) in enumerate(df.groupby("language", sort=True))
        }
        relaxed_negatives = 0
    pairs = [pair for language_pairs in examples_by_language.values() for pair in language_pairs]
    random.Random(args.seed).shuffle(pairs)
    expected_pairs = coverage["n_profiles_eligible_for_finetuning"] * args.pairs_per_author
    if len(pairs) != expected_pairs:
        raise RuntimeError(f"Expected {expected_pairs} pairs from every eligible profile, got {len(pairs)}")
    if len(pairs) < 2:
        raise ValueError("Need at least two source-separated positive pairs; add sources before fine-tuning.")

    from sentence_transformers import InputExample, SentenceTransformer, losses
    from sentence_transformers.datasets import NoDuplicatesDataLoader

    model = SentenceTransformer(args.model_name, device=args.device)
    if args.language_aware_batches:
        loaders = [
            NoDuplicatesDataLoader(
                [InputExample(texts=list(example)) for example in language_pairs],
                batch_size=args.batch_size,
            )
            for language_pairs in examples_by_language.values()
            if language_pairs
        ]
    else:
        loaders = [NoDuplicatesDataLoader(
            [InputExample(texts=list(example)) for example in pairs],
            batch_size=args.batch_size,
        )]
    train_objectives = [(loader, losses.MultipleNegativesRankingLoss(model)) for loader in loaders]
    warmup_steps = max(1, int(sum(len(loader) for loader in loaders) * args.epochs * args.warmup_ratio))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model.fit(
        train_objectives=train_objectives,
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
            df.assign(source_key=independent_source_keys(df))
            .groupby(["language", "author_or_speaker"])["source_key"].nunique().ge(2).sum()
        ),
        "n_registry_author_language_profiles": coverage["n_registry_author_language_profiles"],
        "n_profiles_with_chunks": coverage["n_profiles_with_chunks"],
        "n_profiles_with_at_least_3_sources": coverage["n_profiles_with_at_least_3_sources"],
        "registry_profiles_without_chunks": coverage["registry_profiles_without_chunks"],
        "profiles_with_one_source": coverage["profiles_with_one_source"],
        "all_eligible_profiles_contributed_pairs": len(pairs) == expected_pairs,
        "language_aware_batches": args.language_aware_batches,
        "hard_negatives": args.hard_negatives,
        "relaxed_hard_negatives": relaxed_negatives,
        "pairs_by_language": {language: len(values) for language, values in examples_by_language.items()},
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
