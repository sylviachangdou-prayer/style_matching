"""Leave-one-source-out retrieval evaluation.

Every author-language profile with at least two independent sources gets one
fold per source: that source's chunks become queries while the profile
centroid is rebuilt from the remaining sources. Other authors keep their full
centroids. Compared with the single fixed source-heldout split, every chunk is
used both as profile evidence and as a query, which tightens confidence
intervals without any same-work leakage. Chunk embeddings are reused from the
index build cache, so no extra GPU encoding is needed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


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
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--per-source-cap", type=int, default=50)
    parser.add_argument("--query-cap", type=int, default=50)
    parser.add_argument("--min-sources", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260710)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from scripts.multilingual_style_index import encode_cached, load_model

    df = pd.read_parquet(args.input).reset_index(drop=True)
    model = load_model(args.model_name, None, args.device)
    embeddings = encode_cached(model, df, args.embedding_cache, args.model_name, args.batch_size)
    metrics = compute_loso_metrics(
        df, embeddings,
        per_source_cap=args.per_source_cap,
        query_cap=args.query_cap,
        min_sources=args.min_sources,
        seed=args.seed,
    )
    metrics["model_name"] = args.model_name
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
