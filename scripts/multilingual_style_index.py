from __future__ import annotations

import argparse
import csv
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
        profile = profile.copy()
        profile["_independent_source_id"] = (
            profile["independent_source_id"]
            if "independent_source_id" in profile
            else profile["source_id"]
        ).fillna("").astype(str)
        source_parts = []
        for _, source in profile.groupby(["corpus", "_independent_source_id"], sort=True):
            if per_source_cap > 0 and len(source) > per_source_cap:
                source = source.sample(n=per_source_cap, random_state=seed)
            source_parts.append(source)
        profile_sample = pd.concat(source_parts, ignore_index=False)
        if profile_cap > 0 and len(profile_sample) > profile_cap:
            profile_sample = profile_sample.sample(n=profile_cap, random_state=seed)
        sampled.append(profile_sample)
    if not sampled:
        return df.iloc[0:0].copy()
    return pd.concat(sampled, ignore_index=True).drop(columns="_independent_source_id", errors="ignore")


def profile_metadata(df: pd.DataFrame) -> list[dict[str, object]]:
    rows = []
    for profile_id, (key, indices) in enumerate(df.groupby(["language", "author_or_speaker"]).indices.items()):
        language, author = key
        source_corpora = sorted(df.iloc[indices]["corpus"].astype(str).unique().tolist())
        source_ids = sorted({
            str(row["corpus"]) + "::" + str(
                row.get("independent_source_id") or row["source_id"]
            )
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


def attach_registry_metadata(rows: list[dict[str, object]], registry_path: Path) -> None:
    if not registry_path.exists():
        return
    with registry_path.open(newline="", encoding="utf-8") as handle:
        metadata = {
            (row["original_language"], row["name"]): row for row in csv.DictReader(handle)
        }
    for row in rows:
        person = metadata.get((str(row["language"]), str(row["author_or_speaker"])), {})
        row["profile"] = person.get("profile", "")
        row["style_traits"] = person.get("style_traits", "")
        row["photo_url"] = person.get("photo_url", "")


def attach_admission_tiers(rows: list[dict[str, object]], heldout_report_path: Path | None) -> None:
    eligible = set()
    if heldout_report_path and heldout_report_path.exists():
        report = json.loads(heldout_report_path.read_text(encoding="utf-8"))
        eligible = {
            (str(row["language"]), str(row["author"]))
            for row in report.get("authors", [])
            if row.get("eligible")
        }
    for row in rows:
        key = (str(row["language"]), str(row["author_or_speaker"]))
        row["admission_tier"] = "formal" if key in eligible else "exploratory"


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
    attach_registry_metadata(profile_rows, args.registry)
    attach_admission_tiers(profile_rows, args.heldout_report)
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
        display_allowed = (
            df.iloc[indices]["display_allowed"].fillna(False).astype(bool).to_numpy()
            if "display_allowed" in df.columns
            else np.zeros(len(indices), dtype=bool)
        )
        representative_order = [
            local_index for local_index in np.argsort(local_scores)[::-1] if display_allowed[local_index]
        ][:3]
        for local_index in representative_order:
            row = df.iloc[indices[local_index]]
            representative_rows.append({
                "profile_id": profile_id,
                "corpus": row["corpus"],
                "source_id": row["source_id"],
                "title": row.get("title", ""),
                "text": row["text"],
                "centroid_similarity": float(local_scores[local_index]),
            })

    profile_lookup = {
        (str(row["language"]), str(row["author_or_speaker"])): int(row["profile_id"])
        for row in profile_rows
    }
    prototype_rows = []
    prototype_centroids = []
    source_identity_column = (
        "independent_source_id" if "independent_source_id" in df else "source_id"
    )
    for prototype_id, (key, indices) in enumerate(
        df.groupby(
            ["language", "author_or_speaker", "corpus", source_identity_column], sort=True
        ).indices.items()
    ):
        language, author, corpus, source_id = key
        centroid = embeddings[indices].mean(axis=0)
        centroid /= max(np.linalg.norm(centroid), 1e-12)
        prototype_centroids.append(centroid)
        prototype_rows.append({
            "prototype_id": prototype_id,
            "profile_id": profile_lookup[(str(language), str(author))],
            "language": str(language),
            "author_or_speaker": str(author),
            "corpus": str(corpus),
            "source_id": str(source_id),
            "n_chunks": int(len(indices)),
        })

    decade_rows = []
    decade_centroids = []
    validated_decade_groups = set()
    if args.decade_validation and args.decade_validation.exists():
        validation = json.loads(args.decade_validation.read_text(encoding="utf-8"))
        validated_decade_groups = {
            key for key, value in validation.get("groups", {}).items() if value.get("display_eligible")
        }
    if "decade" in df.columns and validated_decade_groups:
        dated = df[df["decade"].fillna("").astype(str).ne("")]
        for decade_id, (key, indices) in enumerate(
            dated.groupby(["language", "corpus", "decade"]).indices.items()
        ):
            language, corpus, decade = key
            rows = dated.iloc[indices]
            identity = (
                rows["independent_source_id"]
                if "independent_source_id" in rows
                else rows["source_id"]
            )
            source_keys = rows["corpus"].astype(str) + "::" + identity.fillna("").astype(str)
            validation_key = f"{language}::{corpus}"
            if (
                validation_key not in validated_decade_groups
                or rows["author_or_speaker"].nunique() < args.decade_min_authors
                or source_keys.nunique() < args.decade_min_sources
            ):
                continue
            decade_embedding = embeddings[dated.index.to_numpy()[indices]].mean(axis=0)
            decade_embedding /= max(np.linalg.norm(decade_embedding), 1e-12)
            decade_centroids.append(decade_embedding)
            decade_rows.append({
                "decade_id": len(decade_rows),
                "language": str(language),
                "corpus": str(corpus),
                "decade": str(decade),
                "n_authors": int(rows["author_or_speaker"].nunique()),
                "n_sources": int(source_keys.nunique()),
                "n_chunks": int(len(rows)),
            })

    args.out_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.out_dir / "centroids.npy", np.vstack(centroids).astype("float32"))
    if topic_centroids:
        np.save(args.out_dir / "topic_centroids.npy", np.vstack(topic_centroids).astype("float32"))
    pd.DataFrame(profile_rows).to_parquet(args.out_dir / "profiles.parquet", index=False)
    pd.DataFrame(
        representative_rows,
        columns=["profile_id", "corpus", "source_id", "title", "text", "centroid_similarity"],
    ).to_parquet(args.out_dir / "representative_passages.parquet", index=False)
    np.save(args.out_dir / "source_prototype_centroids.npy", np.vstack(prototype_centroids).astype("float32"))
    pd.DataFrame(prototype_rows).to_parquet(args.out_dir / "source_prototypes.parquet", index=False)
    if decade_centroids:
        np.save(args.out_dir / "decade_centroids.npy", np.vstack(decade_centroids).astype("float32"))
        pd.DataFrame(decade_rows).to_parquet(args.out_dir / "decades.parquet", index=False)
    open_set_calibration = {}
    if args.open_set_calibration_dir and args.open_set_calibration_dir.exists():
        for path in sorted(args.open_set_calibration_dir.glob("*/open_set_metrics.json")):
            report = json.loads(path.read_text(encoding="utf-8"))
            open_set_calibration[str(report["language"])] = {
                "coefficient": float(report["calibrator"]["coefficient"]),
                "intercept": float(report["calibrator"]["intercept"]),
                "similarity_threshold": float(report["open_set"]["equal_error_threshold"]),
                "auroc": float(report["open_set"]["auroc"]),
                "ece": float(report["open_set"]["ece"]),
            }
    selection_decision = args.model_label
    fusion_adopted = False
    if args.model_comparison and args.model_comparison.exists():
        comparison = json.loads(args.model_comparison.read_text(encoding="utf-8"))
        selection_decision = str(comparison["decision"])
        fusion_adopted = bool(comparison.get("fusion_adopted", False))
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
        "n_source_prototypes": len(prototype_rows),
        "n_chunks": len(df),
        "n_decade_profiles": len(decade_rows),
        "decade_display_enabled": bool(decade_rows),
        "languages": sorted(df["language"].unique().tolist()),
        "score_status": "uncalibrated_cosine",
        "score_version": args.score_version,
        "artifact_version": args.artifact_version,
        "profile_strategy": args.profile_strategy,
        "prototype_top_k": args.prototype_top_k,
        "open_set_calibration": open_set_calibration,
        "model_label": args.model_label,
        "selection_decision": selection_decision,
        "fusion_adopted": fusion_adopted,
        "deployment_matches_selection": selection_decision == args.model_label,
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
        prototype_centroid_path = index_dir / "source_prototype_centroids.npy"
        prototype_path = index_dir / "source_prototypes.parquet"
        self.prototype_centroids = (
            np.load(prototype_centroid_path) if prototype_centroid_path.exists() else None
        )
        self.prototypes = pd.read_parquet(prototype_path) if prototype_path.exists() else None
        topic_centroid_path = index_dir / "topic_centroids.npy"
        self.topic_centroids = np.load(topic_centroid_path) if topic_centroid_path.exists() else None
        self.decade_centroids = None
        self.decades = None
        if self.metadata.get("n_decade_profiles", 0):
            self.decade_centroids = np.load(index_dir / "decade_centroids.npy")
            self.decades = pd.read_parquet(index_dir / "decades.parquet")
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
        single_centroid_scores = self.centroids @ query_embedding
        scores = single_centroid_scores
        if (
            self.metadata.get("profile_strategy") == "source_prototype_topk_mean"
            and self.prototype_centroids is not None
            and self.prototypes is not None
        ):
            prototype_scores = self.prototype_centroids @ query_embedding
            scores = np.full(len(self.profiles), -1.0, dtype="float32")
            prototype_top_k = int(self.metadata.get("prototype_top_k", 3))
            for profile_id, indices in self.prototypes.groupby("profile_id").groups.items():
                values = np.sort(prototype_scores[np.asarray(list(indices), dtype=int)])[::-1][:prototype_top_k]
                scores[int(profile_id)] = float(values.mean())
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
        rejection = {}
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
                source_corpora = profile["source_corpora"]
                if isinstance(source_corpora, np.ndarray):
                    source_corpora = source_corpora.tolist()
                matches.append({
                    "author_or_speaker": profile["author_or_speaker"],
                    "target_language": profile["language"],
                    "source_corpora": source_corpora,
                    "n_sources": int(profile["n_sources"]),
                    "style_similarity": float(scores[index]),
                    "single_centroid_similarity": float(single_centroid_scores[index]),
                    "topic_similarity": float(topic_scores[index]) if topic_scores is not None else None,
                    "affinity_score": float(ranking_scores[index]),
                    "style_weight": float(style_weight),
                    "calibrated": False,
                    "profile": profile.get("profile", ""),
                    "style_traits": profile.get("style_traits", ""),
                    "photo_url": profile.get("photo_url", ""),
                    "admission_tier": profile.get("admission_tier", "exploratory"),
                    "representative_passages": passages[["title", "source_id", "text"]].to_dict("records"),
                })
            results[target_language] = matches
            calibration = self.metadata.get("open_set_calibration", {}).get(target_language)
            if mode == "within" and calibration and len(indices):
                max_similarity = float(np.max(scores[indices]))
                logit = calibration["coefficient"] * max_similarity + calibration["intercept"]
                known_probability = float(1.0 / (1.0 + np.exp(-logit)))
                rejection[target_language] = {
                    "status": "calibrated_open_set",
                    "max_style_similarity": max_similarity,
                    "known_probability": known_probability,
                    "similarity_threshold": float(calibration["similarity_threshold"]),
                    "accept": max_similarity >= float(calibration["similarity_threshold"]),
                }
            else:
                rejection[target_language] = {"status": "uncalibrated", "accept": None}
        decade_match = None
        decade_matches = {}
        decade_status = "unavailable_not_validated"
        if self.decade_centroids is not None and self.decades is not None:
            decade_indices = self.decades.index[self.decades["language"].eq(language)].to_numpy()
            if len(decade_indices):
                decade_scores = self.decade_centroids @ query_embedding
                for corpus, corpus_indices in self.decades.iloc[decade_indices].groupby("corpus").groups.items():
                    corpus_indices = np.asarray(list(corpus_indices), dtype=int)
                    best_index = int(corpus_indices[np.argmax(decade_scores[corpus_indices])])
                    best = self.decades.iloc[best_index]
                    decade_matches[str(corpus)] = {
                        "decade": str(best["decade"]),
                        "corpus": str(corpus),
                        "style_similarity": float(decade_scores[best_index]),
                        "n_authors": int(best["n_authors"]),
                        "n_sources": int(best["n_sources"]),
                        "n_chunks": int(best["n_chunks"]),
                        "calibrated": False,
                        "experimental": True,
                    }
                decade_match = next(iter(decade_matches.values())) if len(decade_matches) == 1 else None
                decade_status = "validated_experimental" if decade_matches else decade_status
        return {
            "mode": mode,
            "input_language": language,
            "confidence": confidence,
            "ranking_scope": scope,
            "score_status": "uncalibrated_cosine",
            "score_version": self.metadata.get("score_version", "stylematch_v1"),
            "artifact_version": self.metadata.get("artifact_version", "unversioned"),
            "profile_strategy": self.metadata.get("profile_strategy", "single_centroid"),
            "rejection": rejection,
            "decade_status": decade_status,
            "decade_match": decade_match,
            "decade_matches": decade_matches,
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
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
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
    build.add_argument("--registry", type=Path, default=Path("data/source_registry/all_people.csv"))
    build.add_argument("--heldout-report", type=Path)
    build.add_argument(
        "--profile-strategy",
        choices=["single_centroid", "source_prototype_topk_mean"],
        default="single_centroid",
    )
    build.add_argument("--prototype-top-k", type=int, default=3)
    build.add_argument("--score-version", default="stylematch_v1")
    build.add_argument("--artifact-version", default="baseline_v1")
    build.add_argument("--open-set-calibration-dir", type=Path)
    build.add_argument("--model-label", default="mstyle_finetuned")
    build.add_argument("--model-comparison", type=Path)
    build.add_argument("--decade-validation", type=Path)
    build.add_argument("--decade-min-authors", type=int, default=5)
    build.add_argument("--decade-min-sources", type=int, default=20)
    build.set_defaults(function=build_index)

    query = subparsers.add_parser("query")
    shared_query_args(query)
    query.set_defaults(function=query_index)

    timing = subparsers.add_parser("benchmark")
    shared_query_args(timing)
    timing.add_argument("--runs", type=int, default=30)
    timing.add_argument("--output", type=Path)
    timing.set_defaults(function=benchmark)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
