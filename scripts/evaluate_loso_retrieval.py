"""Leave-one-source-out retrieval evaluation.

Every author-language profile with at least two independent sources gets one
fold per source: that source's chunks become queries while the profile
centroid is rebuilt from the remaining sources. Other authors keep their full
centroids. Compared with the single fixed source-heldout split, every cached
chunk is used as both profile evidence and a query, which tightens confidence
intervals without any same-work leakage. The evaluated rows are aligned to the
index build cache by ``chunk_id``; no model loading or extra encoding occurs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

def independent_source_keys(df: pd.DataFrame) -> pd.Series:
    identity = df["independent_source_id"] if "independent_source_id" in df else df["source_id"]
    return df["corpus"].astype(str) + "::" + identity.fillna("").astype(str)


def _normalize(vector: np.ndarray) -> np.ndarray:
    return vector / max(np.linalg.norm(vector), 1e-12)


def _rank_metrics(ranks: list[int]) -> dict[str, float]:
    if not ranks:
        return {}
    array = np.asarray(ranks)
    return {
        "n_queries": int(len(array)),
        "top1_accuracy": float((array <= 1).mean()),
        "top3_accuracy": float((array <= 3).mean()),
        "top5_accuracy": float((array <= 5).mean()),
        "top20_accuracy": float((array <= 20).mean()),
        "mrr": float((1.0 / array).mean()),
    }


def load_aligned_cache(
    df: pd.DataFrame, cache_path: Path, model_name: str
) -> tuple[pd.DataFrame, np.ndarray, dict[str, int | float]]:
    required = {
        "chunk_id", "language", "author_or_speaker", "corpus", "source_id", "text"
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Input is missing required columns: {sorted(missing)}")
    if df["chunk_id"].astype(str).duplicated().any():
        raise ValueError("Input chunk_id values must be unique")
    if not cache_path.exists():
        raise FileNotFoundError(f"Embedding cache not found: {cache_path}")

    with np.load(cache_path, allow_pickle=False) as payload:
        missing_cache_fields = {"model_name", "chunk_ids", "embeddings"} - set(payload.files)
        if missing_cache_fields:
            raise ValueError(
                f"Embedding cache is missing fields: {sorted(missing_cache_fields)}"
            )
        cached_model_values = np.asarray(payload["model_name"]).reshape(-1)
        if len(cached_model_values) != 1:
            raise ValueError("Embedding cache model_name must contain exactly one value")
        cached_model = str(cached_model_values[0])
        if cached_model != model_name:
            raise ValueError(
                f"Embedding cache model mismatch: cache={cached_model!r}, requested={model_name!r}"
            )
        cache_ids = payload["chunk_ids"].astype(str)
        embeddings = payload["embeddings"].astype("float32")

    if embeddings.ndim != 2 or len(embeddings) != len(cache_ids):
        raise ValueError("Embedding cache chunk_ids and embeddings are not row-aligned")
    if pd.Series(cache_ids).duplicated().any():
        raise ValueError("Embedding cache chunk_ids must be unique")
    if not np.isfinite(embeddings).all():
        raise ValueError("Embedding cache contains non-finite values")

    input_ids = set(df["chunk_id"].astype(str))
    keep = np.asarray([chunk_id in input_ids for chunk_id in cache_ids])
    aligned_ids = cache_ids[keep]
    if not len(aligned_ids):
        raise ValueError("Embedding cache and input parquet have no chunk_id overlap")
    aligned_embeddings = embeddings[keep]
    norms = np.linalg.norm(aligned_embeddings, axis=1, keepdims=True)
    if np.any(norms <= 1e-12):
        raise ValueError("Embedding cache contains zero-length vectors")
    aligned_embeddings = aligned_embeddings / norms

    indexed = df.assign(chunk_id=df["chunk_id"].astype(str)).set_index("chunk_id", drop=False)
    aligned_df = indexed.loc[aligned_ids].reset_index(drop=True)
    def profile_source_count(frame: pd.DataFrame) -> int:
        keys = independent_source_keys(frame)
        return int(
            frame.assign(_source_key=keys)[
                ["language", "author_or_speaker", "_source_key"]
            ].drop_duplicates().shape[0]
        )

    input_source_count = profile_source_count(df)
    evaluated_source_count = profile_source_count(aligned_df)
    coverage = {
        "input_chunks": int(len(df)),
        "cache_chunks": int(len(cache_ids)),
        "evaluated_chunks": int(len(aligned_df)),
        "input_chunk_coverage": float(len(aligned_df) / len(df)) if len(df) else 0.0,
        "input_sources": int(input_source_count),
        "evaluated_sources": int(evaluated_source_count),
        "input_source_coverage": (
            float(evaluated_source_count / input_source_count) if input_source_count else 0.0
        ),
    }
    return aligned_df, aligned_embeddings.astype("float32"), coverage


def compute_loso_metrics(
    df: pd.DataFrame,
    embeddings: np.ndarray,
    per_source_cap: int = 50,
    query_cap: int = 50,
    min_sources: int = 2,
    seed: int = 20260710,
) -> dict:
    """df rows must align positionally with embeddings (already normalized)."""
    rng = np.random.default_rng(seed)
    df = df.reset_index(drop=True)
    chunk_ranks: list[int] = []
    source_ranks: list[int] = []
    chunk_ranks_by_language: dict[str, list[int]] = {}
    n_profiles = 0
    n_folds = 0

    for language, lang_frame in df.groupby("language", sort=True):
        positions_by_author: dict[str, dict[str, np.ndarray]] = {}
        for author, group in lang_frame.groupby("author_or_speaker", sort=True):
            sources = {}
            for key, rows in group.groupby(independent_source_keys(group)):
                positions = rows.index.to_numpy()
                if per_source_cap and len(positions) > per_source_cap:
                    positions = rng.choice(positions, size=per_source_cap, replace=False)
                sources[str(key)] = positions
            positions_by_author[str(author)] = sources

        eligible = {a: s for a, s in positions_by_author.items() if len(s) >= min_sources}
        if not eligible:
            continue
        authors = sorted(positions_by_author)
        full_centroids = {
            author: _normalize(
                embeddings[np.concatenate(list(sources.values()))].mean(axis=0)
            )
            for author, sources in positions_by_author.items()
        }
        for author, sources in sorted(eligible.items()):
            n_profiles += 1
            author_index = authors.index(author)
            for held_out, held_positions in sorted(sources.items()):
                n_folds += 1
                remaining = np.concatenate(
                    [p for key, p in sources.items() if key != held_out]
                )
                candidate_matrix = np.vstack([
                    _normalize(embeddings[remaining].mean(axis=0))
                    if candidate == author else full_centroids[candidate]
                    for candidate in authors
                ])
                queries = held_positions
                if query_cap and len(queries) > query_cap:
                    queries = rng.choice(queries, size=query_cap, replace=False)
                scores = embeddings[queries] @ candidate_matrix.T
                true_scores = scores[:, author_index]
                ranks = 1 + (scores > true_scores[:, None]).sum(axis=1)
                chunk_ranks.extend(int(r) for r in ranks)
                chunk_ranks_by_language.setdefault(str(language), []).extend(
                    int(r) for r in ranks
                )
                source_query = _normalize(embeddings[queries].mean(axis=0))
                source_scores = candidate_matrix @ source_query
                source_ranks.append(
                    int(1 + (source_scores > source_scores[author_index]).sum())
                )

    return {
        "protocol": "leave_one_source_out",
        "n_profiles_evaluated": n_profiles,
        "n_folds": n_folds,
        "chunk_level": _rank_metrics(chunk_ranks),
        "source_level": _rank_metrics(source_ranks),
        "by_language": {
            language: _rank_metrics(ranks)
            for language, ranks in sorted(chunk_ranks_by_language.items())
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Leave-one-source-out retrieval evaluation.")
    parser.add_argument("--input", type=Path, required=True, help="Chunk parquet with text column.")
    parser.add_argument("--model-name", default="StyleDistance/mstyledistance")
    parser.add_argument("--embedding-cache", type=Path, required=True,
                        help="NPZ chunk-embedding cache; reuse the index build cache to avoid re-encoding.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--batch-size", type=int, default=128,
        help="Retained for command compatibility; cache-only LOSO never encodes.",
    )
    parser.add_argument("--per-source-cap", type=int, default=50)
    parser.add_argument("--query-cap", type=int, default=50)
    parser.add_argument("--min-sources", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260710)
    parser.add_argument(
        "--device", default="auto",
        help="Retained for command compatibility; cache-only LOSO never loads a model.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_parquet(args.input).reset_index(drop=True)
    df, embeddings, cache_coverage = load_aligned_cache(
        df, args.embedding_cache, args.model_name
    )
    metrics = compute_loso_metrics(
        df, embeddings,
        per_source_cap=args.per_source_cap,
        query_cap=args.query_cap,
        min_sources=args.min_sources,
        seed=args.seed,
    )
    if not metrics["n_folds"]:
        raise ValueError("No author-language profile has enough cached independent sources for LOSO")
    metrics["model_name"] = args.model_name
    metrics["cache_coverage"] = cache_coverage
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
