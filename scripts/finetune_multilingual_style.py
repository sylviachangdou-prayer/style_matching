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
import os
import random
from pathlib import Path

import pandas as pd

# Set before torch is imported (via sentence_transformers inside main) so it
# takes effect: reduces CUDA fragmentation that turns headroom into OOM.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


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
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=256,
        help="Hard cap on tokens per text; only ever lowers the model's native limit. Bounds attention memory so heavier challenger models fit.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip when output-dir already holds a finished model (training_config.json); lets a two-model batch resume without retraining the first.",
    )
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
    """Anchor/positive from two sources of one author; negative from another
    author, narrowed corpus -> topic -> decade when a narrowing keeps candidates.

    Implemented with precomputed positional indexes: the naive per-draw
    DataFrame filtering was O(authors x pairs x corpus_size) and took hours
    once the corpus grew past a few hundred sources.
    """
    import numpy as np

    rng = random.Random(seed)
    output: dict[str, list[tuple[str, str, str]]] = {}
    relaxed_negatives = 0
    all_texts = df["text"].astype(str).to_numpy()
    all_authors = df["author_or_speaker"].astype(str).to_numpy()
    narrowing_columns = [c for c in ("corpus", "topic", "decade") if c in df.columns]

    for language, lang_frame in df.groupby("language", sort=True):
        lang = lang_frame.reset_index(drop=True)
        authors = lang["author_or_speaker"].astype(str).to_numpy()
        texts = lang["text"].astype(str).to_numpy()
        col_values = {
            c: lang[c].fillna("").astype(str).to_numpy() for c in narrowing_columns
        }
        # Positions per applied-column state, built lazily: state -> {values: ndarray}
        index_cache: dict[tuple[str, ...], dict] = {(): {(): np.arange(len(lang))}}

        def state_positions(cols: tuple[str, ...], values: tuple[str, ...]):
            if cols not in index_cache:
                grouped = lang.groupby(
                    [lang[c].fillna("").astype(str) for c in cols]
                ).indices
                index_cache[cols] = {
                    (key if isinstance(key, tuple) else (key,)): positions
                    for key, positions in grouped.items()
                }
            return index_cache[cols].get(values)

        excl_cache: dict[tuple, np.ndarray] = {}

        def excluding_author(cols, values, author):
            key = (cols, values, author)
            if key not in excl_cache:
                positions = state_positions(cols, values)
                if positions is None:
                    excl_cache[key] = np.empty(0, dtype=int)
                else:
                    excl_cache[key] = positions[authors[positions] != author]
            return excl_cache[key]

        for author, group in lang.groupby("author_or_speaker", sort=True):
            source_key = independent_source_keys(group)
            sources = {key: rows.index.to_numpy() for key, rows in group.groupby(source_key)}
            if len(sources) < 2:
                continue
            global_other = np.flatnonzero(all_authors != str(author))
            if not len(global_other):
                continue
            examples = output.setdefault(str(language), [])
            source_names = list(sources)
            for _ in range(pairs_per_author):
                left_source, right_source = rng.sample(source_names, 2)
                anchor_pos = sources[left_source][rng.randrange(len(sources[left_source]))]
                positive_pos = sources[right_source][rng.randrange(len(sources[right_source]))]
                cols: tuple[str, ...] = ()
                values: tuple[str, ...] = ()
                for column in narrowing_columns:
                    value = col_values[column][anchor_pos]
                    if not value:
                        continue
                    trial_cols, trial_values = cols + (column,), values + (value,)
                    if len(excluding_author(trial_cols, trial_values, str(author))):
                        cols, values = trial_cols, trial_values
                candidates = excluding_author(cols, values, str(author))
                if len(candidates):
                    negative_text = texts[candidates[rng.randrange(len(candidates))]]
                else:
                    negative_text = all_texts[global_other[rng.randrange(len(global_other))]]
                    relaxed_negatives += 1
                examples.append((texts[anchor_pos], texts[positive_pos], negative_text))
        print(
            f"pairs built: {language}: {len(output.get(str(language), []))} examples",
            flush=True,
        )
    for examples in output.values():
        rng.shuffle(examples)
    return output, relaxed_negatives


def main() -> None:
    args = parse_args()
    # Check before touching the parquet so a finished model is skipped instantly.
    if args.skip_existing and args.output_dir and (args.output_dir / "training_config.json").exists():
        print(f"skip existing model: {args.output_dir} already has training_config.json", flush=True)
        return
    df = pd.read_parquet(args.input)
    if "split" in df.columns:
        df = df[df["split"].eq(args.split)].copy()
    coverage = training_coverage(df, args.registry)
    if args.audit_only:
        print(json.dumps(coverage, indent=2, ensure_ascii=False))
        return
    if args.output_dir is None:
        raise ValueError("--output-dir is required unless --audit-only is used")
    print(
        f"loaded {len(df)} chunks; building pairs for "
        f"{coverage['n_profiles_eligible_for_finetuning']} eligible profiles...",
        flush=True,
    )
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

    print(f"{len(pairs)} pairs ready; loading {args.model_name} on {args.device}...", flush=True)
    model = SentenceTransformer(args.model_name, device=args.device)
    # Only ever lower the limit. Some models report a huge sentinel
    # max_seq_length, so an unbounded long chunk would OOM at train time; cap it.
    native = getattr(model, "max_seq_length", None)
    model.max_seq_length = min(native, args.max_seq_length) if native else args.max_seq_length
    print(f"max_seq_length capped at {model.max_seq_length}", flush=True)
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
