from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_MODEL = "StyleDistance/mstyledistance"
DEFAULT_TOPIC_MODEL = "intfloat/multilingual-e5-base"
# These are the language codes currently present in the registry. mStyleDistance
# can encode arbitrary XLM-R text, but quality must be calibrated per language pair.
SUPPORTED_LANGUAGES = {"de", "en", "es", "fr", "it", "ja", "pl", "ru", "zh"}

_DISPLAY_NOISE_RE = re.compile(
    r"\[(?:illustration|picture|frontispiece|music|decoration|map|plate)(?::[^\]]*)?\]"
    r"|(?:\s*\*\s*){3,}",
    flags=re.IGNORECASE,
)
_SENTENCE_END_RE = re.compile(r"[.!?。！？…]+[\"'’”」』】）》）)]*")


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
        representative_order = []
        prepared_passages: dict[int, str] = {}
        for local_index in np.argsort(local_scores)[::-1]:
            if not display_allowed[local_index]:
                continue
            prepared = prepare_display_passage(str(df.iloc[indices[local_index]]["text"]))
            if not prepared:
                continue
            representative_order.append(local_index)
            prepared_passages[local_index] = prepared
            if len(representative_order) == 3:
                break
        for local_index in representative_order:
            row = df.iloc[indices[local_index]]
            representative_rows.append({
                "profile_id": profile_id,
                "corpus": row["corpus"],
                "source_id": row["source_id"],
                "title": row.get("title", ""),
                "text": prepared_passages[local_index],
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
        open_set_calibration, _ = load_open_set_calibration(args.open_set_calibration_dir)
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


def _strip_leading_heading(text: str) -> str:
    """Remove a short all-caps chapter or section label from a passage start."""
    tokens = list(re.finditer(r"\S+", text))[:14]
    uppercase_words = 0
    for token in tokens:
        letters = "".join(character for character in token.group() if character.isalpha())
        if letters and letters.isupper():
            uppercase_words += 1
            continue
        if uppercase_words >= 2 and letters:
            return text[token.start():]
        break
    return text


def prepare_display_passage(passage: str, target_chars: int | None = None) -> str:
    """Return clean, whole sentences for user-facing evidence.

    Training chunks remain unchanged. Display candidates containing Gutenberg
    illustration markers or ornamental dividers are rejected because removing
    those markers can join a caption or chapter title to unrelated prose.
    """
    passage = " ".join(str(passage).split()).strip()
    if not passage or _DISPLAY_NOISE_RE.search(passage):
        return ""
    passage = _strip_leading_heading(passage).strip()
    endings = list(_SENTENCE_END_RE.finditer(passage))
    if not endings:
        return ""

    first_character = next((character for character in passage if character.isalpha()), "")
    start = endings[0].end() if first_character and first_character.islower() else 0
    endings = [ending for ending in endings if ending.end() > start]
    if not endings:
        return ""

    limit = int(target_chars * 1.4) if target_chars else len(passage)
    eligible_ends = [ending.end() for ending in endings if ending.end() - start <= limit]
    end = eligible_ends[-1] if eligible_ends else endings[0].end()
    excerpt = passage[start:end].strip(" \t\n—–-")
    return excerpt if any(character.isalpha() for character in excerpt) else ""


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
        self.ecore_scorer = None
        self.ecore_cohort_centres = {}
        scorer_path = index_dir / "ecore_scorer.json"
        cohort_vectors_path = index_dir / "ecore_cohort_centres.npy"
        cohort_rows_path = index_dir / "ecore_cohort_centres.parquet"
        if scorer_path.exists() and cohort_vectors_path.exists() and cohort_rows_path.exists():
            self.ecore_scorer = json.loads(scorer_path.read_text(encoding="utf-8"))
            cohort_vectors = np.load(cohort_vectors_path)
            cohort_rows = pd.read_parquet(cohort_rows_path)
            self.ecore_cohort_centres = {
                str(row["environment"]): cohort_vectors[int(row["centre_id"])]
                for _, row in cohort_rows.iterrows()
            }
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
        # Style embeddings for the licence-approved representative passages, so the
        # API can show each matched author's passage closest to the query. Encoded
        # once at startup (never in a request handler) and cached beside the index.
        self.passage_style_embeddings = None
        if len(self.passages) and "text" in self.passages.columns:
            self.passage_style_embeddings = self._load_or_encode_passages()

    def _load_or_encode_passages(self) -> np.ndarray:
        cache_path = self.index_dir / "passage_style_embeddings.npz"
        texts = self.passages["text"].fillna("").tolist()
        text_sha256 = hashlib.sha256("\0".join(texts).encode("utf-8")).hexdigest()
        if cache_path.exists():
            payload = np.load(cache_path, allow_pickle=False)
            if (
                str(payload["model_name"]) == str(self.metadata["model_name"])
                and "text_sha256" in payload.files
                and str(payload["text_sha256"]) == text_sha256
                and len(payload["embeddings"]) == len(texts)
            ):
                return payload["embeddings"]
        embeddings = self.model.encode(
            texts,
            batch_size=32,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).astype("float32")
        try:
            np.savez_compressed(
                cache_path,
                model_name=str(self.metadata["model_name"]),
                text_sha256=text_sha256,
                embeddings=embeddings,
            )
        except OSError:
            pass
        return embeddings

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
        elif (
            self.metadata.get("profile_strategy") == "ecore_episodic_linear"
            and self.ecore_scorer is not None
            and self.prototype_centroids is not None
            and self.prototypes is not None
        ):
            # The validated first scorer has only a language-fallback cohort.
            # Cross-language candidates retain centroid scores until ordered-pair
            # ECoRe calibration exists.
            temperature = float(self.ecore_scorer["temperature"])
            weights = np.asarray(self.ecore_scorer["weights"], dtype="float32")
            prototype_scores = self.prototype_centroids @ query_embedding
            scores = single_centroid_scores.copy()
            centre = self.ecore_cohort_centres.get(f"{language}::__fallback__")
            if centre is not None:
                query_residual = query_embedding - centre
                query_residual /= max(np.linalg.norm(query_residual), 1e-12)
                for profile_id, indices in self.prototypes.groupby("profile_id").groups.items():
                    profile_id = int(profile_id)
                    if str(self.profiles.iloc[profile_id]["language"]) != language:
                        continue
                    positions = np.asarray(list(indices), dtype=int)
                    values = prototype_scores[positions]
                    scaled = values / temperature
                    soft = temperature * (
                        float(scaled.max())
                        + math.log(float(np.exp(scaled - scaled.max()).mean()))
                    )
                    residual_prototypes = self.prototype_centroids[positions] - centre
                    residual_prototypes /= np.maximum(
                        np.linalg.norm(residual_prototypes, axis=1, keepdims=True), 1e-12
                    )
                    residual_values = residual_prototypes @ query_residual
                    residual_scaled = residual_values / temperature
                    cohort = temperature * (
                        float(residual_scaled.max())
                        + math.log(float(np.exp(residual_scaled - residual_scaled.max()).mean()))
                    )
                    features = np.asarray([
                        single_centroid_scores[profile_id],
                        values.max(),
                        soft,
                        cohort,
                        values.std() if len(values) > 1 else 0.0,
                    ], dtype="float32")
                    scores[profile_id] = float(features @ weights + self.ecore_scorer.get("intercept", 0.0))
        topic_scores = None
        if self.topic_model is not None and self.topic_centroids is not None:
            topic_query = encode_topic(self.topic_model, [text], 1, query=True)[0]
            topic_scores = self.topic_centroids @ topic_query

        style_weight_within = self.metadata.get("style_weight_within", 0.7)
        style_weight_cross = self.metadata.get("style_weight_cross", 0.5)

        def build_match(index: int, style_weight: float, affinity: float) -> dict:
            profile = self.profiles.iloc[index]
            profile_id = int(profile["profile_id"])
            passages = self.passages[self.passages["profile_id"].eq(profile_id)]
            passage_columns = ["title", "source_id", "text"]
            passage_columns.extend(
                column for column in (
                    "translated_text",
                    "translation_language",
                    "translator",
                    "translation_year",
                    "translation_publisher",
                )
                if column in passages.columns
            )
            passage_records = [
                {key: value for key, value in record.items() if not pd.isna(value)}
                for record in passages[passage_columns].to_dict("records")
            ]
            if passage_records and self.passage_style_embeddings is not None:
                similarities = (
                    self.passage_style_embeddings[passages.index.to_numpy()] @ query_embedding
                )
                selected = []
                for best in np.argsort(similarities)[::-1]:
                    record = dict(passage_records[int(best)])
                    prepared = prepare_display_passage(record["text"], max(len(text), 200))
                    if prepared:
                        record["text"] = prepared
                        record["passage_style_similarity"] = float(similarities[int(best)])
                        selected = [record]
                        break
                passage_records = selected
            source_corpora = profile["source_corpora"]
            if isinstance(source_corpora, np.ndarray):
                source_corpora = source_corpora.tolist()
            return {
                "author_or_speaker": profile["author_or_speaker"],
                "target_language": profile["language"],
                "cross_language": bool(profile["language"] != language),
                "source_corpora": source_corpora,
                "n_sources": int(profile["n_sources"]),
                "style_similarity": float(scores[index]),
                "single_centroid_similarity": float(single_centroid_scores[index]),
                "topic_similarity": float(topic_scores[index]) if topic_scores is not None else None,
                "affinity_score": float(affinity),
                "style_weight": float(style_weight),
                "calibrated": False,
                "profile": profile.get("profile", ""),
                "style_traits": profile.get("style_traits", ""),
                "photo_url": profile.get("photo_url", ""),
                "admission_tier": profile.get("admission_tier", "exploratory"),
                "representative_passages": passage_records,
            }

        def open_set_rejection(target_language: str, indices: np.ndarray) -> dict:
            calibration = self.metadata.get("open_set_calibration", {}).get(target_language)
            if calibration and len(indices):
                max_similarity = float(np.max(scores[indices]))
                logit = calibration["coefficient"] * max_similarity + calibration["intercept"]
                known_probability = float(1.0 / (1.0 + np.exp(-logit)))
                return {
                    "status": "calibrated_open_set",
                    "max_style_similarity": max_similarity,
                    "known_probability": known_probability,
                    "similarity_threshold": float(calibration["similarity_threshold"]),
                    "accept": max_similarity >= float(calibration["similarity_threshold"]),
                }
            return {"status": "uncalibrated", "accept": None}

        results = {}
        rejection = {}
        if mode == "all":
            # Single global ranking over every profile. Same-language candidates keep the
            # within-language style/topic mix; cross-language candidates use the cross mix
            # and stay flagged: raw cosines are not calibrated across language pairs, and
            # in practice same-language matches dominate unless a cross-language profile
            # is genuinely closer.
            profile_languages = self.profiles["language"].astype(str).to_numpy()
            weights = np.where(
                profile_languages == str(language), style_weight_within, style_weight_cross
            )
            ranking_scores = weights * scores if topic_scores is None else (
                weights * scores + (1.0 - weights) * topic_scores
            )
            ranked = np.argsort(ranking_scores)[::-1][:top_k]
            matches = [
                build_match(int(index), float(weights[int(index)]), float(ranking_scores[int(index)]))
                for index in ranked
            ]
            results["all"] = matches
            same_language_indices = self.profiles.index[self.profiles["language"].eq(language)].to_numpy()
            rejection[language] = open_set_rejection(language, same_language_indices)
            for target_language in np.unique(profile_languages):
                rejection.setdefault(str(target_language), {"status": "uncalibrated", "accept": None})
            confidence = "standard" if all(not match["cross_language"] for match in matches) else "reduced"
            scope = "global_all_languages"
        else:
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
            style_weight = style_weight_within if mode == "within" else style_weight_cross
            for target_language, indices in groups:
                ranking_scores = scores if topic_scores is None else (
                    style_weight * scores + (1.0 - style_weight) * topic_scores
                )
                ranked = indices[np.argsort(ranking_scores[indices])[::-1][:top_k]]
                results[target_language] = [
                    build_match(int(index), float(style_weight), float(ranking_scores[index]))
                    for index in ranked
                ]
                if mode == "within":
                    rejection[target_language] = open_set_rejection(target_language, indices)
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


def load_open_set_calibration(calibration_dir: Path) -> tuple[dict, dict[str, str]]:
    """Read per-language open-set calibrators; also return which encoder each
    report was fitted on, so callers can refuse cross-model calibration."""
    calibration: dict = {}
    source_models: dict[str, str] = {}
    for path in sorted(calibration_dir.glob("*/open_set_metrics.json")):
        report = json.loads(path.read_text(encoding="utf-8"))
        language = str(report["language"])
        calibration[language] = {
            "coefficient": float(report["calibrator"]["coefficient"]),
            "intercept": float(report["calibrator"]["intercept"]),
            "similarity_threshold": float(report["open_set"]["equal_error_threshold"]),
            "auroc": float(report["open_set"]["auroc"]),
            "ece": float(report["open_set"]["ece"]),
        }
        source_models[language] = str(report.get("model_name", ""))
    return calibration, source_models


def calibrate_index(args: argparse.Namespace) -> None:
    """Refit metadata.open_set_calibration from a directory of per-language
    open-set reports without rebuilding centroids or re-encoding anything."""
    metadata_path = args.index_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    calibration, source_models = load_open_set_calibration(args.open_set_calibration_dir)
    if not calibration:
        raise SystemExit(f"No */open_set_metrics.json found under {args.open_set_calibration_dir}")
    mismatched = {
        language: model
        for language, model in source_models.items()
        if model and model != metadata["model_name"]
    }
    if mismatched and not args.allow_model_mismatch:
        raise SystemExit(
            "Refusing to install calibration fitted on a different encoder than the index "
            f"({metadata['model_name']}): {json.dumps(mismatched)}. Re-run evaluate_open_set.py "
            "with the index's model, or pass --allow-model-mismatch if this is deliberate."
        )
    metadata["open_set_calibration"] = calibration
    metadata["open_set_calibration_source"] = str(args.open_set_calibration_dir)
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps({
        "index_dir": str(args.index_dir),
        "calibrated_languages": sorted(calibration),
        "auroc": {language: values["auroc"] for language, values in calibration.items()},
    }, indent=2))


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
    parser.add_argument("--mode", choices=["all", "within", "cross"], default="within")
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

    calibrate = subparsers.add_parser(
        "calibrate", help="Install per-language open-set calibration into an existing index's metadata."
    )
    calibrate.add_argument("--index-dir", type=Path, required=True)
    calibrate.add_argument("--open-set-calibration-dir", type=Path, required=True)
    calibrate.add_argument("--allow-model-mismatch", action="store_true")
    calibrate.set_defaults(function=calibrate_index)

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
