from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_MODEL = "StyleDistance/mstyledistance"
DEFAULT_TOPIC_MODEL = "intfloat/multilingual-e5-base"
# These are the language codes currently present in the registry. mStyleDistance
# can encode arbitrary XLM-R text, but quality must be calibrated per language pair.
SUPPORTED_LANGUAGES = {"de", "en", "es", "fr", "it", "ja", "pl", "ru", "zh"}


def validate_frame(df: pd.DataFrame) -> None:
    required = {"corpus", "author_or_speaker", "language", "source_id", "text"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    unsupported = set(df["language"].dropna().unique()) - SUPPORTED_LANGUAGES
    if unsupported:
        raise ValueError(f"Languages not supported by {DEFAULT_MODEL}: {sorted(unsupported)}")


def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def load_model(model_name: str, backend: str | None = None, device: str = "auto"):
    from sentence_transformers import SentenceTransformer

    kwargs = {"backend": backend} if backend else {}
    kwargs["device"] = resolve_device(device)
    return SentenceTransformer(model_name, **kwargs)


def encode(model, texts: list[str], batch_size: int) -> np.ndarray:
    return model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")


def encode_topic(model, texts: list[str], batch_size: int, query: bool = False) -> np.ndarray:
    prefix = "query: Find texts with similar subject matter and tonal register: " if query else "passage: "
    return encode(model, [prefix + text for text in texts], batch_size)


def encode_cached(
    model,
    df: pd.DataFrame,
    cache_path: Path,
    model_name: str,
    batch_size: int,
) -> np.ndarray:
    """Encode each chunk once and reuse it on later index rebuilds.

    The cache is keyed by chunk_id and invalidated when the encoder name changes.
    This is deliberately a flat NPZ cache so it can be copied between Colab and
    the web app without a database or one-file-per-chunk output.
    """
    ids = df["chunk_id"].astype(str).tolist()
    cached: dict[str, np.ndarray] = {}
    if cache_path.exists():
        try:
            payload = np.load(cache_path, allow_pickle=False)
            if str(payload["model_name"]) == model_name:
                cached = {
                    str(chunk_id): embedding
                    for chunk_id, embedding in zip(payload["chunk_ids"].tolist(), payload["embeddings"])
                }
        except (OSError, KeyError, ValueError):
            cached = {}

    missing = [index for index, chunk_id in enumerate(ids) if chunk_id not in cached]
    if missing:
        new_embeddings = encode(model, df.iloc[missing]["text"].fillna("").tolist(), batch_size)
        for index, embedding in zip(missing, new_embeddings):
            cached[ids[index]] = embedding
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        ordered_ids = list(dict.fromkeys([*cached.keys(), *ids]))
        matrix = np.vstack([cached[chunk_id] for chunk_id in ordered_ids]).astype("float32")
        np.savez_compressed(
            cache_path,
            model_name=np.asarray(model_name),
            chunk_ids=np.asarray(ordered_ids),
            embeddings=matrix,
        )

    return np.vstack([cached[chunk_id] for chunk_id in ids]).astype("float32")


def encode_topic_cached(
    model,
    df: pd.DataFrame,
    cache_path: Path,
    model_name: str,
    batch_size: int,
) -> np.ndarray:
    ids = df["chunk_id"].astype(str).tolist()
    cached: dict[str, np.ndarray] = {}
    if cache_path.exists():
        try:
            payload = np.load(cache_path, allow_pickle=False)
            if str(payload["model_name"]) == model_name:
                cached = {
                    str(chunk_id): embedding
                    for chunk_id, embedding in zip(payload["chunk_ids"].tolist(), payload["embeddings"])
                }
        except (OSError, KeyError, ValueError):
            cached = {}
    missing = [index for index, chunk_id in enumerate(ids) if chunk_id not in cached]
    if missing:
        new_embeddings = encode_topic(model, df.iloc[missing]["text"].fillna("").tolist(), batch_size)
        for index, embedding in zip(missing, new_embeddings):
            cached[ids[index]] = embedding
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        ordered_ids = list(dict.fromkeys([*cached.keys(), *ids]))
        matrix = np.vstack([cached[chunk_id] for chunk_id in ordered_ids]).astype("float32")
        np.savez_compressed(
            cache_path,
            model_name=np.asarray(model_name),
            chunk_ids=np.asarray(ordered_ids),
            embeddings=matrix,
        )
    return np.vstack([cached[chunk_id] for chunk_id in ids]).astype("float32")


def balanced_profile_sample(
    df: pd.DataFrame,
    per_source_cap: int,
    profile_cap: int,
    seed: int,
) -> pd.DataFrame:
    """Keep source diversity while preventing prolific authors dominating profiles."""
    if per_source_cap <= 0 and profile_cap <= 0:
        return df.reset_index(drop=True)
    sampled = []
    for _, profile in df.groupby(["language", "author_or_speaker"], sort=True):
        source_parts = []
        for _, source in profile.groupby(["corpus", "source_id"], sort=True):
            if per_source_cap > 0 and len(source) > per_source_cap:
                source = source.sample(n=per_source_cap, random_state=seed)
            source_parts.append(source)
        profile_sample = pd.concat(source_parts, ignore_index=False)
        if profile_cap > 0 and len(profile_sample) > profile_cap:
            profile_sample = profile_sample.sample(n=profile_cap, random_state=seed)
        sampled.append(profile_sample)
    if not sampled:
        return df.iloc[0:0].copy()
    return pd.concat(sampled, ignore_index=True)


def profile_metadata(df: pd.DataFrame) -> list[dict[str, object]]:
    rows = []
    for profile_id, (key, indices) in enumerate(df.groupby(["language", "author_or_speaker"]).indices.items()):
        language, author = key
        source_corpora = sorted(df.iloc[indices]["corpus"].astype(str).unique().tolist())
        source_ids = sorted({
            str(row["corpus"]) + "::" + str(row["source_id"])
            for _, row in df.iloc[indices].iterrows()
        })
        rows.append({
            "profile_id": profile_id,
            "language": language,
            "author_or_speaker": author,
            "n_chunks": len(indices),
            "n_sources": len(source_ids),
            "source_corpora": source_corpora,
        })
    return rows


def build_index(args: argparse.Namespace) -> None:
    df = pd.read_parquet(args.input)
    validate_frame(df)
    if "split" in df and args.split:
        df = df[df["split"].eq(args.split)].copy()
    df = balanced_profile_sample(df, args.per_source_cap, args.profile_cap, args.seed)
    if df.empty:
        raise ValueError("No rows available to build profiles")

    model = load_model(args.model_name, args.backend, args.device)
    cache_path = args.embedding_cache or (args.out_dir / "chunk_embeddings.npz")
    embeddings = encode_cached(model, df, cache_path, args.model_name, args.batch_size)
    topic_embeddings = None
    topic_cache_path = None
    if args.topic_model_name:
        topic_model = load_model(args.topic_model_name, None, args.device)
        topic_cache_path = args.topic_embedding_cache or (args.out_dir / "topic_chunk_embeddings.npz")
        topic_embeddings = encode_topic_cached(
            topic_model, df, topic_cache_path, args.topic_model_name, args.batch_size
        )
    group_columns = ["language", "author_or_speaker"]
    profile_rows = profile_metadata(df)
    centroids = []
    topic_centroids = []
    representative_rows = []
    for profile_id, (key, indices) in enumerate(df.groupby(group_columns).indices.items()):
        group_embeddings = embeddings[indices]
        centroid = group_embeddings.mean(axis=0)
        centroid /= max(np.linalg.norm(centroid), 1e-12)
        centroids.append(centroid)
        if topic_embeddings is not None:
            topic_centroid = topic_embeddings[indices].mean(axis=0)
            topic_centroid /= max(np.linalg.norm(topic_centroid), 1e-12)
            topic_centroids.append(topic_centroid)
        local_scores = group_embeddings @ centroid
        for local_index in np.argsort(local_scores)[-3:][::-1]:
            row = df.iloc[indices[local_index]]
            representative_rows.append({
                "profile_id": profile_id,
                "corpus": row["corpus"],
                "source_id": row["source_id"],
                "title": row.get("title", ""),
                "text": row["text"],
                "centroid_similarity": float(local_scores[local_index]),
            })

    args.out_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.out_dir / "centroids.npy", np.vstack(centroids).astype("float32"))
    if topic_centroids:
        np.save(args.out_dir / "topic_centroids.npy", np.vstack(topic_centroids).astype("float32"))
    pd.DataFrame(profile_rows).to_parquet(args.out_dir / "profiles.parquet", index=False)
    pd.DataFrame(representative_rows).to_parquet(args.out_dir / "representative_passages.parquet", index=False)
    metadata = {
        "model_name": args.model_name,
        "backend": args.backend,
        "input": str(args.input),
        "split": args.split,
        "profile_cap": args.profile_cap,
        "per_source_cap": args.per_source_cap,
        "embedding_cache": str(cache_path),
        "device": resolve_device(args.device),
        "topic_model_name": args.topic_model_name,
        "topic_embedding_cache": str(topic_cache_path) if topic_cache_path else None,
        "style_weight_within": args.style_weight_within,
        "style_weight_cross": args.style_weight_cross,
        "n_profiles": len(profile_rows),
        "n_chunks": len(df),
        "languages": sorted(df["language"].unique().tolist()),
        "score_status": "uncalibrated_cosine",
    }
    (args.out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


class StyleIndex:
    def __init__(self, index_dir: Path, backend: str | None = None, device: str = "auto") -> None:
        self.index_dir = index_dir
        self.metadata = json.loads((index_dir / "metadata.json").read_text(encoding="utf-8"))
        self.profiles = pd.read_parquet(index_dir / "profiles.parquet")
        self.passages = pd.read_parquet(index_dir / "representative_passages.parquet")
        self.centroids = np.load(index_dir / "centroids.npy")
        topic_centroid_path = index_dir / "topic_centroids.npy"
        self.topic_centroids = np.load(topic_centroid_path) if topic_centroid_path.exists() else None
        self.model = load_model(
            self.metadata["model_name"],
            backend or self.metadata.get("backend"),
            device,
        )
        self.topic_model = None
        if self.topic_centroids is not None and self.metadata.get("topic_model_name"):
            self.topic_model = load_model(
                self.metadata["topic_model_name"], None, device
            )

    def query(self, text: str, language: str, mode: str, top_k: int) -> dict:
        if language not in SUPPORTED_LANGUAGES:
            raise ValueError(f"Unsupported input language: {language}")
        query_embedding = self.model.encode(
            [text],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )[0].astype("float32")
        scores = self.centroids @ query_embedding
        topic_scores = None
        if self.topic_model is not None and self.topic_centroids is not None:
            topic_query = encode_topic(self.topic_model, [text], 1, query=True)[0]
            topic_scores = self.topic_centroids @ topic_query

        if mode == "within":
            groups = [(language, self.profiles.index[self.profiles["language"].eq(language)].to_numpy())]
            confidence = "standard"
            scope = "within_language"
        else:
            groups = [
                (target_language, indices.to_numpy())
                for target_language, indices in self.profiles.groupby("language").groups.items()
            ]
            confidence = "reduced"
            scope = "per_target_language"

        results = {}
        style_weight = self.metadata.get(
            "style_weight_within" if mode == "within" else "style_weight_cross",
            0.7 if mode == "within" else 0.5,
        )
        for target_language, indices in groups:
            ranking_scores = scores if topic_scores is None else (
                style_weight * scores + (1.0 - style_weight) * topic_scores
            )
            ranked = indices[np.argsort(ranking_scores[indices])[::-1][:top_k]]
            matches = []
            for index in ranked:
                profile = self.profiles.iloc[index]
                profile_id = int(profile["profile_id"])
                passages = self.passages[self.passages["profile_id"].eq(profile_id)]
                matches.append({
                    "author_or_speaker": profile["author_or_speaker"],
                    "target_language": profile["language"],
                    "source_corpora": profile["source_corpora"],
                    "n_sources": int(profile["n_sources"]),
                    "style_similarity": float(scores[index]),
                    "topic_similarity": float(topic_scores[index]) if topic_scores is not None else None,
                    "affinity_score": float(ranking_scores[index]),
                    "style_weight": float(style_weight),
                    "calibrated": False,
                    "representative_passages": passages[["title", "source_id", "text"]].to_dict("records"),
                })
            results[target_language] = matches
        return {
            "mode": mode,
            "input_language": language,
            "confidence": confidence,
            "ranking_scope": scope,
            "score_status": "uncalibrated_cosine",
            "results": results,
        }


def query_index(args: argparse.Namespace) -> None:
    index = StyleIndex(args.index_dir, args.backend, args.device)
    print(json.dumps(index.query(args.text, args.language, args.mode, args.top_k), ensure_ascii=False, indent=2))


def benchmark(args: argparse.Namespace) -> None:
    index = StyleIndex(args.index_dir, args.backend, args.device)
    index.query(args.text, args.language, args.mode, args.top_k)
    durations = []
    for _ in range(args.runs):
        started = time.perf_counter()
        index.query(args.text, args.language, args.mode, args.top_k)
        durations.append((time.perf_counter() - started) * 1000)
    report = {
        "runs": args.runs,
        "mode": args.mode,
        "backend": args.backend or index.metadata.get("backend") or "torch",
        "device": resolve_device(args.device),
        "p50_ms": float(np.percentile(durations, 50)),
        "p95_ms": float(np.percentile(durations, 95)),
        "min_ms": float(np.min(durations)),
        "max_ms": float(np.max(durations)),
    }
    print(json.dumps(report, indent=2))


def shared_query_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--index-dir", type=Path, required=True)
    parser.add_argument("--text", required=True)
    parser.add_argument("--language", required=True, choices=sorted(SUPPORTED_LANGUAGES))
    parser.add_argument("--mode", choices=["within", "cross"], default="within")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--backend", choices=["torch", "onnx", "openvino"])
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or mps")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build and query a precomputed multilingual style-profile index.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build")
    build.add_argument("--input", type=Path, required=True)
    build.add_argument("--out-dir", type=Path, required=True)
    build.add_argument("--model-name", default=DEFAULT_MODEL)
    build.add_argument("--backend", choices=["torch", "onnx", "openvino"])
    build.add_argument("--batch-size", type=int, default=64)
    build.add_argument("--split", default="train")
    build.add_argument("--profile-cap", type=int, default=600)
    build.add_argument("--per-source-cap", type=int, default=50)
    build.add_argument("--seed", type=int, default=20260710)
    build.add_argument("--embedding-cache", type=Path)
    build.add_argument("--topic-model-name", default=DEFAULT_TOPIC_MODEL)
    build.add_argument("--topic-embedding-cache", type=Path)
    build.add_argument("--style-weight-within", type=float, default=0.7)
    build.add_argument("--style-weight-cross", type=float, default=0.5)
    build.add_argument("--device", default="auto", help="auto, cpu, cuda, or mps")
    build.set_defaults(function=build_index)

    query = subparsers.add_parser("query")
    shared_query_args(query)
    query.set_defaults(function=query_index)

    timing = subparsers.add_parser("benchmark")
    shared_query_args(timing)
    timing.add_argument("--runs", type=int, default=30)
    timing.set_defaults(function=benchmark)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
