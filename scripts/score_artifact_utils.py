from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


REFERENCE_FIELDS = (
    "chunk_ids",
    "splits",
    "query_languages",
    "query_corpora",
    "profiles",
    "y_true",
)


def parse_score_spec(spec: str) -> tuple[str, Path, str]:
    name, location = spec.split("=", 1)
    path, key = location.rsplit(":", 1)
    return name, Path(path), key


def load_aligned_scores(specs: list[str]) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    matrices: dict[str, np.ndarray] = {}
    reference: dict[str, np.ndarray] | None = None
    for spec in specs:
        name, path, key = parse_score_spec(spec)
        if name in matrices:
            raise ValueError(f"Duplicate score name: {name}")
        payload = np.load(path, allow_pickle=True)
        missing = set(REFERENCE_FIELDS + (key,)) - set(payload.files)
        if missing:
            raise ValueError(f"Missing arrays in {path}: {sorted(missing)}")
        current = {field: payload[field] for field in REFERENCE_FIELDS}
        if reference is None:
            reference = current
        else:
            for field in REFERENCE_FIELDS:
                if not np.array_equal(reference[field], current[field]):
                    raise ValueError(f"Score files are not aligned on {field}: {path}")
        matrix = payload[key].astype("float64")
        if matrix.shape != (len(current["chunk_ids"]), len(current["profiles"])):
            raise ValueError(f"Unexpected score shape for {name}: {matrix.shape}")
        matrices[name] = matrix
    if reference is None:
        raise ValueError("At least one --scores specification is required")
    return matrices, reference


def aligned_metadata(input_path: Path, reference: dict[str, np.ndarray]) -> pd.DataFrame:
    df = pd.read_parquet(input_path)
    if "chunk_id" not in df.columns or df["chunk_id"].duplicated().any():
        raise ValueError("Input must contain unique chunk_id values")
    aligned = df.set_index(df["chunk_id"].astype(str), drop=False).reindex(
        reference["chunk_ids"].astype(str)
    )
    if aligned["chunk_id"].isna().any():
        missing = reference["chunk_ids"].astype(str)[aligned["chunk_id"].isna().to_numpy()]
        raise ValueError(f"Score artifact contains unknown chunk ids: {missing[:5].tolist()}")
    aligned = aligned.reset_index(drop=True)
    checks = {
        "split": reference["splits"].astype(str),
        "language": reference["query_languages"].astype(str),
        "corpus": reference["query_corpora"].astype(str),
    }
    for column, expected in checks.items():
        if column in aligned.columns and not np.array_equal(aligned[column].astype(str), expected):
            raise ValueError(f"Input metadata is not aligned on {column}")
    return aligned


def independent_source_keys(frame: pd.DataFrame) -> pd.Series:
    if "independent_source_id" in frame.columns:
        source = frame["independent_source_id"].fillna("").astype(str)
    elif "source_id" in frame.columns:
        source = frame["source_id"].fillna("").astype(str)
    else:
        raise ValueError("Input needs independent_source_id or source_id for source-unit evaluation")
    empty = source.eq("")
    if empty.any():
        source = source.copy()
        source.loc[empty] = frame.loc[empty, "chunk_id"].astype(str)
    corpus = frame["corpus"].astype(str) if "corpus" in frame.columns else "unknown"
    return (
        frame["language"].astype(str)
        + "::"
        + frame["author_or_speaker"].astype(str)
        + "::"
        + corpus
        + "::"
        + source
    )


def aggregate_scores_by_source(
    frame: pd.DataFrame,
    labels: np.ndarray,
    matrices: dict[str, np.ndarray],
) -> tuple[pd.DataFrame, np.ndarray, dict[str, np.ndarray]]:
    if len(frame) != len(labels) or any(len(matrix) != len(frame) for matrix in matrices.values()):
        raise ValueError("Metadata, labels, and scores must have equal row counts")
    work = frame.copy().reset_index(drop=True)
    work["source_key"] = independent_source_keys(work)
    groups = list(work.groupby("source_key", sort=True, observed=True).indices.items())
    rows: list[dict[str, object]] = []
    source_labels: list[int] = []
    aggregated = {name: [] for name in matrices}
    for source_key, positions in groups:
        positions = np.asarray(positions, dtype=int)
        group = work.iloc[positions]
        unique_labels = np.unique(labels[positions])
        if len(unique_labels) != 1:
            raise ValueError(f"Source maps to multiple labels: {source_key}")
        for column in ("split", "language", "author_or_speaker"):
            if group[column].astype(str).nunique() != 1:
                raise ValueError(f"Source maps to multiple {column} values: {source_key}")
        rows.append(
            {
                "source_key": source_key,
                "split": str(group["split"].iloc[0]),
                "language": str(group["language"].iloc[0]),
                "corpus": str(group["corpus"].iloc[0]) if "corpus" in group else "unknown",
                "author_or_speaker": str(group["author_or_speaker"].iloc[0]),
                "n_chunks": int(len(group)),
            }
        )
        source_labels.append(int(unique_labels[0]))
        for name, matrix in matrices.items():
            aggregated[name].append(matrix[positions].mean(axis=0))
    return (
        pd.DataFrame(rows),
        np.asarray(source_labels, dtype=int),
        {name: np.vstack(values) for name, values in aggregated.items()},
    )


def candidate_mask(profiles: np.ndarray, language: str) -> np.ndarray:
    profile_languages = np.asarray([str(profile).split("::", 1)[0] for profile in profiles])
    return profile_languages == str(language)


def normalized_score_features(
    scores: np.ndarray,
    query_languages: np.ndarray,
    profiles: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    z_scores = np.zeros_like(scores, dtype="float64")
    percentiles = np.zeros_like(scores, dtype="float64")
    z_scores.fill(-8.0)
    percentiles.fill(0.0)
    for row, language in enumerate(query_languages.astype(str)):
        mask = candidate_mask(profiles, language)
        values = scores[row, mask]
        if not len(values):
            raise ValueError(f"No candidates for query language: {language}")
        standard_deviation = float(values.std())
        z_scores[row, mask] = (values - values.mean()) / max(standard_deviation, 1e-8)
        order = np.argsort(np.argsort(values, kind="stable"), kind="stable")
        percentiles[row, mask] = (order + 1) / len(values)
    return z_scores, percentiles
