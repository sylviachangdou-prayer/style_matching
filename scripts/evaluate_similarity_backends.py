#!/usr/bin/env python3
"""Compare frozen-encoder retrieval backends on source-heldout data.

Transforms are fitted on train embeddings, hyperparameters are selected on dev,
and the selected variant in each family is reported once on test.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluate_hubness_reranking import delta_interval, exposure, intervals
from scripts.style_embedding_recall import balanced_train, profile_key


SEED = 20260902


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--embedding-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-cap", type=int, default=300)
    parser.add_argument("--bootstrap-runs", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args()


def l2(values: np.ndarray) -> np.ndarray:
    return values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-12)


def rank_rows(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, axis=1, kind="stable")
    ranks = np.empty_like(order, dtype="float64")
    rows = np.arange(len(values))[:, None]
    ranks[rows, order] = np.arange(values.shape[1], dtype="float64")
    ranks -= ranks.mean(axis=1, keepdims=True)
    return l2(ranks)


def source_key(frame: pd.DataFrame) -> pd.Series:
    identity = frame.get("independent_source_id", frame["source_id"])
    return frame["corpus"].astype(str) + "::" + identity.fillna("").astype(str)


def align_embeddings(
    frame: pd.DataFrame, ids_path: Path, embeddings_path: Path
) -> np.ndarray:
    # Older local writers used object dtype for IDs. These are trusted artifacts
    # produced by this repository; model arrays still use allow_pickle=False.
    ids = np.load(ids_path, allow_pickle=True).astype(str)
    embeddings = np.load(embeddings_path, allow_pickle=False).astype("float64")
    if len(ids) != len(embeddings) or len(np.unique(ids)) != len(ids):
        raise ValueError(f"Invalid embedding alignment artifact: {ids_path}")
    lookup = {value: position for position, value in enumerate(ids)}
    missing = sorted(set(frame["chunk_id"].astype(str)) - set(lookup))
    if missing:
        raise ValueError(f"Missing embeddings for chunk IDs: {missing[:5]}")
    return embeddings[[lookup[value] for value in frame["chunk_id"].astype(str)]]


def centroids(embeddings: np.ndarray, labels: np.ndarray, n_profiles: int) -> np.ndarray:
    return np.vstack([embeddings[labels == label].mean(axis=0) for label in range(n_profiles)])


def cosine_scores(query: np.ndarray, enroll: np.ndarray) -> np.ndarray:
    return l2(query) @ l2(enroll).T


def all_but_top_fit(train: np.ndarray, components: int) -> tuple[np.ndarray, np.ndarray]:
    mean = train.mean(axis=0)
    _, _, vh = np.linalg.svd(train - mean, full_matrices=False)
    return mean, vh[:components]


def remove_components(values: np.ndarray, mean: np.ndarray, pcs: np.ndarray) -> np.ndarray:
    centered = values - mean
    return centered - (centered @ pcs.T) @ pcs


def whitening_fit(train: np.ndarray, shrinkage: float) -> tuple[np.ndarray, np.ndarray]:
    mean = train.mean(axis=0)
    covariance = np.cov(train - mean, rowvar=False)
    target = np.trace(covariance) / covariance.shape[0]
    covariance = (1.0 - shrinkage) * covariance + shrinkage * target * np.eye(len(covariance))
    values, vectors = np.linalg.eigh(covariance)
    transform = vectors @ np.diag(1.0 / np.sqrt(np.maximum(values, 1e-7))) @ vectors.T
    return mean, transform


def csls(
    query_scores: np.ndarray,
    fit_scores: np.ndarray,
    fit_labels: np.ndarray,
    k: int,
) -> np.ndarray:
    row_k = min(k, query_scores.shape[1])
    row_density = np.sort(query_scores, axis=1)[:, -row_k:].mean(axis=1)
    column_density = np.zeros(query_scores.shape[1], dtype="float64")
    for candidate in range(query_scores.shape[1]):
        impostors = fit_scores[fit_labels != candidate, candidate]
        local_k = min(k, len(impostors))
        column_density[candidate] = np.sort(impostors)[-local_k:].mean() if local_k else 0.0
    return 2.0 * query_scores - row_density[:, None] - column_density[None, :]


def adaptive_snorm(
    query_scores: np.ndarray,
    fit_scores: np.ndarray,
    fit_labels: np.ndarray,
    k: int,
) -> np.ndarray:
    z_mean = np.zeros(query_scores.shape[1])
    z_std = np.ones(query_scores.shape[1])
    for candidate in range(query_scores.shape[1]):
        impostors = fit_scores[fit_labels != candidate, candidate]
        cohort = np.sort(impostors)[-min(k, len(impostors)):]
        if len(cohort):
            z_mean[candidate] = cohort.mean()
            z_std[candidate] = max(cohort.std(), 1e-6)
    z = (query_scores - z_mean) / z_std
    row_k = min(k, query_scores.shape[1])
    cohort = np.sort(query_scores, axis=1)[:, -row_k:]
    t = (query_scores - cohort.mean(axis=1, keepdims=True)) / np.maximum(
        cohort.std(axis=1, keepdims=True), 1e-6
    )
    return 0.5 * (z + t)


def plda_fit(
    train: np.ndarray,
    labels: np.ndarray,
    dimension: int,
    regularization: float,
    cache: dict | None = None,
) -> dict[str, np.ndarray]:
    cache = cache if cache is not None else {}
    mean = cache.setdefault("mean", train.mean(axis=0))
    centered = train - mean
    if "vh" not in cache:
        _, _, cache["vh"] = np.linalg.svd(centered, full_matrices=False)
    vh = cache["vh"]
    projection = vh[: min(dimension, len(vh))].T
    reduced = centered @ projection
    classes = np.unique(labels)
    means = np.vstack([reduced[labels == label].mean(axis=0) for label in classes])
    within = np.zeros((reduced.shape[1], reduced.shape[1]))
    denominator = 0
    for label, local_mean in zip(classes, means):
        residual = reduced[labels == label] - local_mean
        within += residual.T @ residual
        denominator += len(residual)
    within /= max(denominator - len(classes), 1)
    between = np.cov(means, rowvar=False)
    for covariance in (within, between):
        scale = np.trace(covariance) / len(covariance)
        covariance *= 1.0 - regularization
        covariance += regularization * scale * np.eye(len(covariance))
    return {"mean": mean, "projection": projection, "within": within, "between": between}


def plda_scores(
    query: np.ndarray,
    enrollment: np.ndarray,
    enrollment_counts: np.ndarray,
    model: dict[str, np.ndarray],
) -> np.ndarray:
    query = (query - model["mean"]) @ model["projection"]
    enrollment = (enrollment - model["mean"]) @ model["projection"]
    within, between = model["within"], model["between"]
    background = between + within
    background_inverse = np.linalg.pinv(background)
    background_logdet = np.linalg.slogdet(background)[1]
    background_quad = np.einsum("ni,ij,nj->n", query, background_inverse, query)
    output = np.empty((len(query), len(enrollment)), dtype="float64")
    for candidate, count in enumerate(enrollment_counts):
        enroll_covariance = between + within / max(int(count), 1)
        gain = between @ np.linalg.pinv(enroll_covariance)
        conditional_mean = gain @ enrollment[candidate]
        conditional_covariance = background - gain @ between
        conditional_inverse = np.linalg.pinv(conditional_covariance)
        conditional_logdet = np.linalg.slogdet(conditional_covariance)[1]
        residual = query - conditional_mean
        conditional_quad = np.einsum("ni,ij,nj->n", residual, conditional_inverse, residual)
        output[:, candidate] = -0.5 * (
            conditional_quad + conditional_logdet - background_quad - background_logdet
        )
    return output


def language_scores(
    train: np.ndarray,
    train_labels: np.ndarray,
    query: np.ndarray,
    method: str,
    parameter: float | int | None,
    cache: dict | None = None,
) -> np.ndarray:
    cache = cache if cache is not None else {}
    n_profiles = int(train_labels.max()) + 1
    if "enroll" not in cache:
        cache["enroll"] = centroids(train, train_labels, n_profiles)
        cache["raw_train"] = cosine_scores(train, cache["enroll"])
        cache["raw_query"] = cosine_scores(query, cache["enroll"])
    enroll = cache["enroll"]
    raw_train = cache["raw_train"]
    raw_query = cache["raw_query"]
    if method == "cosine":
        return raw_query
    if method == "centered_cosine":
        mean = cache.setdefault("mean", train.mean(axis=0))
        return cosine_scores(query - mean, enroll - mean)
    if method == "all_but_top":
        mean = cache.setdefault("mean", train.mean(axis=0))
        if "vh" not in cache:
            _, _, cache["vh"] = np.linalg.svd(train - mean, full_matrices=False)
        pcs = cache["vh"][: int(parameter)]
        return cosine_scores(
            remove_components(query, mean, pcs), remove_components(enroll, mean, pcs)
        )
    if method == "whitened_cosine":
        mean = cache.setdefault("mean", train.mean(axis=0))
        if "covariance_eigh" not in cache:
            cache["covariance_eigh"] = np.linalg.eigh(np.cov(train - mean, rowvar=False))
        values, vectors = cache["covariance_eigh"]
        shrinkage = float(parameter)
        shrunk = (1.0 - shrinkage) * values + shrinkage * values.mean()
        transform = vectors @ np.diag(1.0 / np.sqrt(np.maximum(shrunk, 1e-7))) @ vectors.T
        return cosine_scores((query - mean) @ transform, (enroll - mean) @ transform)
    if method == "l1":
        result = np.empty((len(query), len(enroll)), dtype="float64")
        for start in range(0, len(query), 256):
            stop = min(start + 256, len(query))
            result[start:stop] = -np.abs(
                query[start:stop, None, :] - enroll[None, :, :]
            ).mean(axis=2)
        return result
    if method == "spearman":
        return rank_rows(query) @ rank_rows(enroll).T
    if method == "csls":
        return csls(raw_query, raw_train, train_labels, int(parameter))
    if method == "adaptive_snorm":
        return adaptive_snorm(raw_query, raw_train, train_labels, int(parameter))
    if method in {"plda", "plda_snorm"}:
        dimension, regularization = parameter
        model_key = ("plda", int(dimension), float(regularization))
        if model_key not in cache:
            cache[model_key] = plda_fit(
                train, train_labels, int(dimension), float(regularization), cache
            )
        model = cache[model_key]
        counts = np.bincount(train_labels, minlength=n_profiles)
        fit = plda_scores(train, enroll, counts, model)
        result = plda_scores(query, enroll, counts, model)
        return adaptive_snorm(result, fit, train_labels, 50) if method == "plda_snorm" else result
    raise ValueError(method)


def assemble_scores(
    train: pd.DataFrame,
    train_embeddings: np.ndarray,
    query: pd.DataFrame,
    query_embeddings: np.ndarray,
    profiles: np.ndarray,
    method: str,
    parameter: object,
    fit_cache: dict[str, dict] | None = None,
) -> np.ndarray:
    fit_cache = fit_cache if fit_cache is not None else {}
    output = np.full((len(query), len(profiles)), -np.inf, dtype="float64")
    profile_languages = np.asarray([profile.split("::", 1)[0] for profile in profiles])
    train_keys = profile_key(train).to_numpy()
    for language in sorted(query["language"].astype(str).unique()):
        rows = np.flatnonzero(query["language"].astype(str).to_numpy() == language)
        train_rows = np.flatnonzero(train["language"].astype(str).to_numpy() == language)
        columns = np.flatnonzero(profile_languages == language)
        if not len(rows) or not len(train_rows) or not len(columns):
            continue
        local_lookup = {profile: index for index, profile in enumerate(profiles[columns])}
        labels = np.asarray([local_lookup[key] for key in train_keys[train_rows]])
        output[np.ix_(rows, columns)] = language_scores(
            train_embeddings[train_rows], labels, query_embeddings[rows], method, parameter,
            fit_cache.setdefault(language, {}),
        )
    return output


def macro_mrr(scores: np.ndarray, labels: np.ndarray) -> float:
    order = np.argsort(scores, axis=1)[:, ::-1]
    ranks = np.asarray([np.flatnonzero(row == true)[0] + 1 for row, true in zip(order, labels)])
    unique = np.unique(labels)
    return float(np.mean([(1.0 / ranks[labels == label]).mean() for label in unique]))


def point_metrics(scores: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    order = np.argsort(scores, axis=1)[:, ::-1]
    ranks = np.asarray([np.flatnonzero(row == true)[0] + 1 for row, true in zip(order, labels)])
    unique = np.unique(labels)
    result = {"mrr": float(np.mean([(1.0 / ranks[labels == label]).mean() for label in unique]))}
    for cutoff in (1, 3, 5, 20):
        result[f"recall_at_{cutoff}"] = float(
            np.mean([(ranks[labels == label] <= cutoff).mean() for label in unique])
        )
    return result


def geometry(train: pd.DataFrame, embeddings: np.ndarray) -> pd.DataFrame:
    rows = []
    for language, subset in train.groupby("language", sort=True):
        positions = subset.index.to_numpy()
        matrix = l2(embeddings[positions])
        singular = np.linalg.svd(matrix - matrix.mean(axis=0), compute_uv=False)
        variance = np.square(singular)
        probability = variance / max(variance.sum(), 1e-12)
        sampled = matrix if len(matrix) <= 2000 else matrix[:2000]
        similarities = sampled @ sampled.T
        rows.append({
            "language": str(language),
            "n_train_chunks": int(len(matrix)),
            "pc1_variance_share": float(probability[0]),
            "effective_rank": float(np.exp(-(probability * np.log(probability + 1e-12)).sum())),
            "mean_pairwise_cosine": float(similarities[np.triu_indices(len(sampled), 1)].mean()),
        })
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.read_parquet(args.input).copy()
    frame["profile_key"] = profile_key(frame)
    train = balanced_train(frame, args.train_cap, args.seed)
    train["profile_key"] = profile_key(train)
    evaluation = frame[frame["split"].isin(["dev", "test"])].copy().reset_index(drop=True)
    profiles = np.asarray(sorted(train["profile_key"].unique()), dtype=str)
    profile_lookup = {profile: position for position, profile in enumerate(profiles)}
    evaluation = evaluation[evaluation["profile_key"].isin(profile_lookup)].reset_index(drop=True)
    train_embeddings = align_embeddings(
        train,
        args.embedding_dir / "style_embedding_train_chunk_ids.npy",
        args.embedding_dir / "style_embedding_train_embeddings.npy",
    )
    eval_embeddings = align_embeddings(
        evaluation,
        args.embedding_dir / "style_embedding_eval_chunk_ids.npy",
        args.embedding_dir / "style_embedding_eval_embeddings.npy",
    )
    labels = np.asarray([profile_lookup[value] for value in evaluation["profile_key"]])

    families = {
        "cosine": [None],
        "centered_cosine": [None],
        "all_but_top": [1, 2, 4, 8],
        "whitened_cosine": [0.1, 0.3, 0.5],
        "l1": [None],
        "spearman": [None],
        "csls": [5, 10, 20],
        "adaptive_snorm": [20, 50, 100],
        "plda": [(32, 0.1), (64, 0.1), (64, 0.3)],
        "plda_snorm": [(32, 0.1), (64, 0.1), (64, 0.3)],
    }
    dev = evaluation["split"].eq("dev").to_numpy()
    test = evaluation["split"].eq("test").to_numpy()
    selected: dict[str, tuple[object, np.ndarray, float]] = {}
    tuning_rows = []
    fit_cache: dict[str, dict] = {}
    for family, parameters in families.items():
        for parameter in parameters:
            scores = assemble_scores(
                train, train_embeddings, evaluation, eval_embeddings,
                profiles, family, parameter, fit_cache,
            )
            value = macro_mrr(scores[dev], labels[dev])
            tuning_rows.append({"family": family, "parameter": repr(parameter), "dev_macro_mrr": value})
            if family not in selected or value > selected[family][2]:
                selected[family] = (parameter, scores, value)

    baseline = selected["cosine"][1][test]
    test_labels = labels[test]
    test_frame = evaluation[test].reset_index(drop=True)
    groups = source_key(test_frame).to_numpy()
    results = []
    exposure_tables = []
    subgroup_rows = []
    for position, (family, (parameter, scores, dev_mrr)) in enumerate(selected.items()):
        test_scores = scores[test]
        estimates = intervals(
            test_scores, test_labels, args.bootstrap_runs, args.seed + 100 * position
        )
        concentration, table = exposure(
            test_scores, test_labels, profiles, groups=groups, top_k=3
        )
        table.insert(0, "method", family)
        exposure_tables.append(table)
        deltas = {
            metric: delta_interval(
                baseline, test_scores, test_labels, metric,
                args.bootstrap_runs, args.seed + 100 * position + offset,
            )
            for offset, metric in enumerate(("mrr", "recall_at_3"), 1)
        }
        r3_values = []
        order = np.argsort(test_scores, axis=1)[:, ::-1]
        for label in np.unique(test_labels):
            local = test_labels == label
            ranks = np.asarray([np.flatnonzero(row == true)[0] + 1 for row, true in zip(order[local], test_labels[local])])
            r3_values.append(float(np.mean(ranks <= 3)))
        results.append({
            "method": family,
            "selected_parameter": repr(parameter),
            "dev_macro_mrr": dev_mrr,
            **{name: values["value"] for name, values in estimates.items()},
            **{f"{name}_ci_low": values["ci_low"] for name, values in estimates.items()},
            **{f"{name}_ci_high": values["ci_high"] for name, values in estimates.items()},
            "worst_decile_profile_recall_at_3": float(np.quantile(r3_values, 0.1)),
            **concentration,
            "mrr_delta_vs_cosine": deltas["mrr"]["delta"],
            "mrr_delta_ci_low": deltas["mrr"]["ci_low"],
            "mrr_delta_ci_high": deltas["mrr"]["ci_high"],
            "recall_at_3_delta_vs_cosine": deltas["recall_at_3"]["delta"],
        })
        for column in ("language", "corpus"):
            for value, subset in test_frame.groupby(column, sort=True):
                positions = subset.index.to_numpy()
                subgroup_rows.append({
                    "method": family,
                    "group_type": column,
                    "group": str(value),
                    "n_queries": int(len(positions)),
                    "n_true_profiles": int(len(np.unique(test_labels[positions]))),
                    **point_metrics(test_scores[positions], test_labels[positions]),
                })

    result_frame = pd.DataFrame(results)
    baseline_row = result_frame[result_frame["method"].eq("cosine")].iloc[0]
    result_frame["adoption_gate"] = (
        (result_frame["mrr_delta_ci_low"] >= -0.01)
        & (result_frame["recall_at_3_delta_vs_cosine"] >= -0.01)
        & (result_frame["false_top3_hhi"] < baseline_row["false_top3_hhi"])
        & (result_frame["false_top3_gini"] < baseline_row["false_top3_gini"])
    )
    eligible = result_frame[result_frame["adoption_gate"]]
    recommendation = None if eligible.empty else str(
        eligible.sort_values(["mrr", "false_top3_hhi"], ascending=[False, True]).iloc[0]["method"]
    )
    report = {
        "design": "train-fitted transforms; dev-selected hyperparameters; locked source-heldout test",
        "seed": args.seed,
        "baseline": "cosine",
        "adoption_rule": "MRR paired-profile CI lower bound >= -0.01; Recall@3 delta >= -0.01; both false-top3 HHI and Gini improve",
        "recommended_backend": recommendation,
        "production_change_authorized": recommendation is not None,
        "open_set": "not tested here; requires author-heldout unknown queries",
        "methods": result_frame.to_dict("records"),
    }
    pd.DataFrame(tuning_rows).to_csv(args.output_dir / "backend_dev_tuning.csv", index=False)
    result_frame.to_csv(args.output_dir / "backend_test_metrics.csv", index=False)
    pd.concat(exposure_tables, ignore_index=True).to_csv(
        args.output_dir / "backend_author_exposure.csv", index=False
    )
    pd.DataFrame(subgroup_rows).to_csv(
        args.output_dir / "backend_subgroup_metrics.csv", index=False
    )
    geometry(train.reset_index(drop=True), train_embeddings).to_csv(
        args.output_dir / "geometry_diagnostics.csv", index=False
    )
    (args.output_dir / "backend_metrics.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps({
        "recommended_backend": recommendation,
        "production_change_authorized": recommendation is not None,
        "results": result_frame[["method", "mrr", "recall_at_3", "false_top3_hhi", "false_top3_gini", "adoption_gate"]].to_dict("records"),
    }, indent=2))
    print(f"RETURN: {args.output_dir / 'backend_metrics.json'}")


if __name__ == "__main__":
    main()
