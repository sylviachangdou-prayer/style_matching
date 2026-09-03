#!/usr/bin/env python3
"""Dev-select hub corrections on top of fixed 0.30-shrinkage whitening."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import ndtri

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluate_hubness_reranking import delta_interval, exposure, intervals
from scripts.evaluate_similarity_backends import (
    align_embeddings,
    assemble_scores,
    cached_whitening,
    centroids,
    cosine_scores,
    point_metrics,
    source_key,
)
from scripts.search_concentration_backends import subgroup_table, worst_decile_recall_at_3
from scripts.style_embedding_recall import balanced_train, profile_key


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--embedding-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--shrinkage", type=float, default=0.3)
    parser.add_argument("--train-cap", type=int, default=300)
    parser.add_argument("--bootstrap-runs", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260904)
    parser.add_argument("--watch-profile", action="append", default=[])
    return parser.parse_args()


def same_language_mask(
    scores: np.ndarray,
    query_languages: np.ndarray,
    profile_languages: np.ndarray,
) -> np.ndarray:
    result = scores.copy()
    result[query_languages[:, None] != profile_languages[None, :]] = -np.inf
    return result


def row_standardize(
    scores: np.ndarray,
    query_languages: np.ndarray,
    profile_languages: np.ndarray,
) -> np.ndarray:
    result = np.full_like(scores, -np.inf, dtype="float64")
    for language in np.unique(query_languages):
        rows = np.flatnonzero(query_languages == language)
        columns = np.flatnonzero(profile_languages == language)
        if not len(rows) or not len(columns):
            continue
        local = scores[np.ix_(rows, columns)]
        result[np.ix_(rows, columns)] = (
            local - local.mean(axis=1, keepdims=True)
        ) / np.maximum(local.std(axis=1, keepdims=True), 1e-6)
    return result


def row_percentiles(
    scores: np.ndarray,
    query_languages: np.ndarray,
    profile_languages: np.ndarray,
) -> np.ndarray:
    result = np.full_like(scores, -np.inf, dtype="float64")
    for language in np.unique(query_languages):
        rows = np.flatnonzero(query_languages == language)
        columns = np.flatnonzero(profile_languages == language)
        if not len(rows) or not len(columns):
            continue
        local = scores[np.ix_(rows, columns)]
        order = np.argsort(np.argsort(local, axis=1, kind="stable"), axis=1, kind="stable")
        result[np.ix_(rows, columns)] = (order + 0.5) / len(columns)
    return result


def blend_scores(
    first: np.ndarray,
    second: np.ndarray,
    alpha: float,
    query_languages: np.ndarray,
    profile_languages: np.ndarray,
) -> np.ndarray:
    result = np.full_like(first, -np.inf, dtype="float64")
    for language in np.unique(query_languages):
        rows = np.flatnonzero(query_languages == language)
        columns = np.flatnonzero(profile_languages == language)
        result[np.ix_(rows, columns)] = (
            (1.0 - alpha) * first[np.ix_(rows, columns)]
            + alpha * second[np.ix_(rows, columns)]
        )
    return result


def candidate_calibration(
    fit_scores: np.ndarray,
    fit_labels: np.ndarray,
    fit_languages: np.ndarray,
    fit_corpora: np.ndarray,
    query_scores: np.ndarray,
    query_languages: np.ndarray,
    query_corpora: np.ndarray,
    profile_languages: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    percentiles = np.full_like(query_scores, -np.inf, dtype="float64")
    robust_z = np.full_like(query_scores, -np.inf, dtype="float64")
    for candidate, language in enumerate(profile_languages):
        fit_pool = (fit_languages == language) & (fit_labels != candidate)
        query_pool = query_languages == language
        for corpus in np.unique(query_corpora[query_pool]):
            rows = query_pool & (query_corpora == corpus)
            reference = fit_scores[fit_pool & (fit_corpora == corpus), candidate]
            if len(reference) < 30:
                reference = fit_scores[fit_pool, candidate]
            reference = np.sort(reference[np.isfinite(reference)])
            if not len(reference):
                percentiles[rows, candidate] = 0.5
                robust_z[rows, candidate] = 0.0
                continue
            values = query_scores[rows, candidate]
            percentiles[rows, candidate] = (
                np.searchsorted(reference, values, side="right") + 0.5
            ) / (len(reference) + 1.0)
            median = np.median(reference)
            mad = np.median(np.abs(reference - median))
            scale = max(1.4826 * mad, np.std(reference) * 0.1, 1e-6)
            robust_z[rows, candidate] = np.clip((values - median) / scale, -12.0, 12.0)
    return percentiles, robust_z


def fit_exposure_prior(
    scores: np.ndarray,
    labels: np.ndarray,
    languages: np.ndarray,
    profile_languages: np.ndarray,
    groups: np.ndarray,
    top_k: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    counts = pd.Series(groups).value_counts().to_dict()
    weights = np.asarray([1.0 / counts[group] for group in groups])
    mass = np.zeros(scores.shape[1], dtype="float64")
    for row, (values, true, weight) in enumerate(zip(scores, labels, weights)):
        columns = np.flatnonzero(profile_languages == languages[row])
        candidates = columns[columns != true]
        selected = candidates[np.argsort(values[candidates])[::-1][:top_k]]
        mass[selected] += weight
    popularity = np.zeros_like(mass)
    for language in np.unique(profile_languages):
        columns = np.flatnonzero(profile_languages == language)
        order = np.argsort(np.argsort(mass[columns], kind="stable"), kind="stable")
        popularity[columns] = (order + 0.5) / len(columns)
    return popularity, mass


def structural_scores(
    baseline: np.ndarray,
    reverse_percentile: np.ndarray,
    query_languages: np.ndarray,
    profile_languages: np.ndarray,
    popularity: np.ndarray,
    anchor_threshold: float,
    popularity_threshold: float,
    penalty: float,
    bonus: float,
    anchor_rank: int = 3,
    shortlist: int = 10,
) -> np.ndarray:
    result = baseline.copy()
    for language in np.unique(query_languages):
        rows = np.flatnonzero(query_languages == language)
        columns = np.flatnonzero(profile_languages == language)
        if not len(rows) or not len(columns):
            continue
        local = baseline[np.ix_(rows, columns)]
        descending = np.argsort(-local, axis=1, kind="stable")
        ranks = np.empty_like(descending)
        ranks[np.arange(len(rows))[:, None], descending] = np.arange(1, len(columns) + 1)
        reverse = reverse_percentile[np.ix_(rows, columns)]
        local_popularity = popularity[columns][None, :]
        anchor = (ranks <= anchor_rank) & (reverse >= anchor_threshold)
        hub = (
            (ranks <= shortlist)
            & (local_popularity >= popularity_threshold)
            & (reverse < anchor_threshold)
        )
        adjustment = bonus * anchor - penalty * local_popularity * hub
        result[np.ix_(rows, columns)] = local + adjustment
    return result


def gaussianize_fit_apply(
    fit: np.ndarray,
    query: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    fit_output = np.empty_like(fit, dtype="float32")
    query_output = np.empty_like(query, dtype="float32")
    for dimension in range(fit.shape[1]):
        reference = np.sort(fit[:, dimension], kind="stable")
        fit_probability = (
            np.searchsorted(reference, fit[:, dimension], side="right") - 0.5
        ) / len(reference)
        query_probability = (
            np.searchsorted(reference, query[:, dimension], side="right") + 0.5
        ) / (len(reference) + 1.0)
        fit_output[:, dimension] = ndtri(np.clip(fit_probability, 1e-5, 1 - 1e-5))
        query_output[:, dimension] = ndtri(np.clip(query_probability, 1e-5, 1 - 1e-5))
    return fit_output, query_output


def whitened_fnorm_scores(
    train: pd.DataFrame,
    train_embeddings: np.ndarray,
    query: pd.DataFrame,
    query_embeddings: np.ndarray,
    profiles: np.ndarray,
    shrinkage: float,
) -> tuple[np.ndarray, np.ndarray]:
    fit_scores = np.full((len(train), len(profiles)), -np.inf, dtype="float64")
    query_scores = np.full((len(query), len(profiles)), -np.inf, dtype="float64")
    profile_languages = np.asarray([profile.split("::", 1)[0] for profile in profiles])
    train_keys = profile_key(train).to_numpy()
    cache: dict[str, dict] = {}
    for language in sorted(query["language"].astype(str).unique()):
        fit_rows = np.flatnonzero(train["language"].astype(str).to_numpy() == language)
        query_rows = np.flatnonzero(query["language"].astype(str).to_numpy() == language)
        columns = np.flatnonzero(profile_languages == language)
        if not len(fit_rows) or not len(query_rows) or not len(columns):
            continue
        lookup = {profile: index for index, profile in enumerate(profiles[columns])}
        labels = np.asarray([lookup[key] for key in train_keys[fit_rows]])
        local_cache = cache.setdefault(language, {})
        mean, transform = cached_whitening(
            train_embeddings[fit_rows], labels, shrinkage, local_cache
        )
        fit = (train_embeddings[fit_rows] - mean) @ transform
        values = (query_embeddings[query_rows] - mean) @ transform
        fit, values = gaussianize_fit_apply(fit, values)
        enrollment = centroids(fit, labels, len(columns))
        fit_scores[np.ix_(fit_rows, columns)] = cosine_scores(fit, enrollment)
        query_scores[np.ix_(query_rows, columns)] = cosine_scores(values, enrollment)
    return fit_scores, query_scores


def source_balanced_maui(
    scores: np.ndarray,
    labels: np.ndarray,
    query_languages: np.ndarray,
    profile_languages: np.ndarray,
    groups: np.ndarray,
    top_k: int = 3,
) -> dict[str, float]:
    group_counts = pd.Series(groups).value_counts().to_dict()
    weights = np.asarray([1.0 / group_counts[group] for group in groups])
    language_values = []
    total_weights = []
    maximum_ratio = 0.0
    for language in np.unique(query_languages):
        rows = np.flatnonzero(query_languages == language)
        columns = np.flatnonzero(profile_languages == language)
        if not len(rows) or len(columns) <= top_k:
            continue
        counts = np.zeros(len(columns), dtype="float64")
        local_lookup = {candidate: position for position, candidate in enumerate(columns)}
        for row in rows:
            selected = columns[np.argsort(scores[row, columns])[::-1][:top_k]]
            for candidate in selected:
                if candidate != labels[row]:
                    counts[local_lookup[candidate]] += weights[row]
        query_mass = weights[rows].sum()
        expected = top_k * query_mass / len(columns)
        numerator = np.maximum(counts - expected, 0.0).sum()
        denominator = top_k * max(query_mass - expected, 1e-12)
        language_values.append(numerator / denominator)
        total_weights.append(query_mass)
        maximum_ratio = max(maximum_ratio, float(counts.max(initial=0.0) / max(expected, 1e-12)))
    if not language_values:
        return {"source_balanced_maui_at_3": 0.0, "maximum_exposure_ratio": 0.0}
    return {
        "source_balanced_maui_at_3": float(np.average(language_values, weights=total_weights)),
        "maximum_exposure_ratio": maximum_ratio,
    }


def evaluate(
    scores: np.ndarray,
    labels: np.ndarray,
    profiles: np.ndarray,
    frame: pd.DataFrame,
) -> tuple[dict[str, float], pd.DataFrame]:
    groups = source_key(frame).to_numpy()
    query_languages = frame["language"].astype(str).to_numpy()
    profile_languages = np.asarray([profile.split("::", 1)[0] for profile in profiles])
    concentration, table = exposure(scores, labels, profiles, groups=groups, top_k=3)
    return {
        **point_metrics(scores, labels),
        "worst_decile_profile_recall_at_3": worst_decile_recall_at_3(scores, labels),
        **concentration,
        **source_balanced_maui(
            scores, labels, query_languages, profile_languages, groups, top_k=3
        ),
    }, table


def subgroup_pass(
    frame: pd.DataFrame,
    labels: np.ndarray,
    baseline: np.ndarray,
    candidate: np.ndarray,
) -> tuple[bool, pd.DataFrame]:
    tables = pd.concat([
        subgroup_table("whitened_cosine:0.3", frame, baseline, labels),
        subgroup_table("candidate", frame, candidate, labels),
    ], ignore_index=True)
    old = tables[tables["method"].eq("whitened_cosine:0.3")].set_index(
        ["group_type", "group"]
    )
    new = tables[tables["method"].eq("candidate")].set_index(
        ["group_type", "group"]
    )
    supported = new[new["n_true_profiles"].ge(10)]
    passed = bool(len(supported)) and all(
        float(row["mrr"] - old.loc[key, "mrr"]) >= -0.02
        and float(row["recall_at_3"] - old.loc[key, "recall_at_3"]) >= -0.02
        for key, row in supported.iterrows()
    )
    return passed, tables


def profile_geometry(
    train: pd.DataFrame,
    embeddings: np.ndarray,
    profiles: np.ndarray,
    popularity: np.ndarray,
    exposure_mass: np.ndarray,
    shrinkage: float,
) -> pd.DataFrame:
    rows = []
    keys = profile_key(train).to_numpy()
    profile_languages = np.asarray([profile.split("::", 1)[0] for profile in profiles])
    for language in np.unique(profile_languages):
        fit_rows = np.flatnonzero(train["language"].astype(str).to_numpy() == language)
        columns = np.flatnonzero(profile_languages == language)
        lookup = {profile: index for index, profile in enumerate(profiles[columns])}
        labels = np.asarray([lookup[key] for key in keys[fit_rows]])
        mean, transform = cached_whitening(
            embeddings[fit_rows], labels, shrinkage, {}, author_balanced=False
        )
        enrollment = centroids((embeddings[fit_rows] - mean) @ transform, labels, len(columns))
        normalized = enrollment / np.maximum(np.linalg.norm(enrollment, axis=1, keepdims=True), 1e-12)
        center = normalized.mean(axis=0)
        center /= max(np.linalg.norm(center), 1e-12)
        distances = 1.0 - normalized @ center
        for local, column in enumerate(columns):
            rows.append({
                "profile": profiles[column],
                "language": language,
                "distance_to_language_centroid": float(distances[local]),
                "train_false_top3_popularity_percentile": float(popularity[column]),
                "train_source_balanced_false_top3_mass": float(exposure_mass[column]),
            })
    return pd.DataFrame(rows)


def candidate_name(family: str, parameters: dict[str, float]) -> str:
    values = ",".join(f"{key}={value:g}" for key, value in parameters.items())
    return f"{family}[{values}]" if values else family


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.read_parquet(args.input).copy()
    frame["profile_key"] = profile_key(frame)
    train = balanced_train(frame, args.train_cap, args.seed).reset_index(drop=True)
    train["profile_key"] = profile_key(train)
    evaluation = frame[frame["split"].isin(["dev", "test"])].copy().reset_index(drop=True)
    profiles = np.asarray(sorted(train["profile_key"].unique()), dtype=str)
    profile_lookup = {profile: position for position, profile in enumerate(profiles)}
    evaluation = evaluation[evaluation["profile_key"].isin(profile_lookup)].reset_index(drop=True)
    labels = np.asarray([profile_lookup[value] for value in evaluation["profile_key"]])
    train_labels = np.asarray([profile_lookup[value] for value in train["profile_key"]])
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
    fit_cache: dict[str, dict] = {}
    fit_scores = assemble_scores(
        train, train_embeddings, train, train_embeddings,
        profiles, "whitened_cosine", args.shrinkage, fit_cache,
    )
    eval_cache: dict[str, dict] = {}
    baseline = assemble_scores(
        train, train_embeddings, evaluation, eval_embeddings,
        profiles, "whitened_cosine", args.shrinkage, eval_cache,
    )
    profile_languages = np.asarray([profile.split("::", 1)[0] for profile in profiles])
    train_languages = train["language"].astype(str).to_numpy()
    eval_languages = evaluation["language"].astype(str).to_numpy()
    train_corpora = train["corpus"].astype(str).to_numpy()
    eval_corpora = evaluation["corpus"].astype(str).to_numpy()
    reverse, robust_z = candidate_calibration(
        fit_scores, train_labels, train_languages, train_corpora,
        baseline, eval_languages, eval_corpora, profile_languages,
    )
    forward = row_percentiles(baseline, eval_languages, profile_languages)
    mutual = same_language_mask(
        np.sqrt(np.clip(forward, 0, 1) * np.clip(reverse, 0, 1)),
        eval_languages, profile_languages,
    )
    popularity, exposure_mass = fit_exposure_prior(
        fit_scores, train_labels, train_languages, profile_languages,
        source_key(train).to_numpy(), top_k=3,
    )
    fnorm_fit, fnorm = whitened_fnorm_scores(
        train, train_embeddings, evaluation, eval_embeddings,
        profiles, args.shrinkage,
    )
    fnorm_reverse, _ = candidate_calibration(
        fnorm_fit, train_labels, train_languages, train_corpora,
        fnorm, eval_languages, eval_corpora, profile_languages,
    )
    fnorm_forward = row_percentiles(fnorm, eval_languages, profile_languages)
    fnorm_mutual = same_language_mask(
        np.sqrt(np.clip(fnorm_forward, 0, 1) * np.clip(fnorm_reverse, 0, 1)),
        eval_languages, profile_languages,
    )

    standardized_baseline = row_standardize(baseline, eval_languages, profile_languages)
    methods: list[tuple[str, dict[str, float], np.ndarray]] = [
        ("whitened_cosine", {"shrinkage": args.shrinkage}, baseline),
    ]
    for alpha in (0.1, 0.2, 0.3, 0.5, 1.0):
        methods.append((
            "robust_candidate_null",
            {"alpha": alpha},
            blend_scores(
                standardized_baseline,
                row_standardize(robust_z, eval_languages, profile_languages),
                alpha, eval_languages, profile_languages,
            ),
        ))
        methods.append((
            "mutual_proximity",
            {"alpha": alpha},
            blend_scores(
                standardized_baseline,
                row_standardize(mutual, eval_languages, profile_languages),
                alpha, eval_languages, profile_languages,
            ),
        ))
    for alpha in (0.25, 0.5, 0.75, 1.0):
        methods.append((
            "whitened_fnorm",
            {"alpha": alpha},
            blend_scores(
                standardized_baseline,
                row_standardize(fnorm, eval_languages, profile_languages),
                alpha, eval_languages, profile_languages,
            ),
        ))
        methods.append((
            "whitened_fnorm_mutual_proximity",
            {"alpha": alpha},
            blend_scores(
                standardized_baseline,
                row_standardize(fnorm_mutual, eval_languages, profile_languages),
                alpha, eval_languages, profile_languages,
            ),
        ))
    for penalty in (0.001, 0.0025, 0.005, 0.01):
        methods.append((
            "exposure_prior",
            {"penalty": penalty},
            same_language_mask(
                baseline - penalty * popularity[None, :], eval_languages, profile_languages
            ),
        ))
    for anchor_threshold in (0.95, 0.98):
        for popularity_threshold in (0.8, 0.9):
            for penalty in (0.0025, 0.005, 0.01):
                for bonus in (0.0, 0.0025):
                    parameters = {
                        "anchor": anchor_threshold,
                        "popularity": popularity_threshold,
                        "penalty": penalty,
                        "bonus": bonus,
                    }
                    methods.append((
                        "structural_expert",
                        parameters,
                        structural_scores(
                            baseline, reverse, eval_languages, profile_languages,
                            popularity, anchor_threshold, popularity_threshold,
                            penalty, bonus,
                        ),
                    ))

    dev_mask = evaluation["split"].eq("dev").to_numpy()
    test_mask = evaluation["split"].eq("test").to_numpy()
    dev_frame = evaluation[dev_mask].reset_index(drop=True)
    test_frame = evaluation[test_mask].reset_index(drop=True)
    dev_labels = labels[dev_mask]
    test_labels = labels[test_mask]
    baseline_dev, _ = evaluate(baseline[dev_mask], dev_labels, profiles, dev_frame)
    dev_rows = []
    dev_watched_rows = []
    selected = None
    selected_key = None
    for family, parameters, scores in methods:
        name = candidate_name(family, parameters)
        metrics, dev_exposure = evaluate(scores[dev_mask], dev_labels, profiles, dev_frame)
        watched_rows = dev_exposure[dev_exposure["profile"].isin(args.watch_profile)].copy()
        watched_rows.insert(0, "candidate", name)
        dev_watched_rows.append(watched_rows)
        passes_subgroups, _ = subgroup_pass(
            dev_frame, dev_labels, baseline[dev_mask], scores[dev_mask]
        )
        metrics["mrr_delta_vs_whitened"] = metrics["mrr"] - baseline_dev["mrr"]
        metrics["recall_at_3_delta_vs_whitened"] = (
            metrics["recall_at_3"] - baseline_dev["recall_at_3"]
        )
        metrics["worst_decile_delta_vs_whitened"] = (
            metrics["worst_decile_profile_recall_at_3"]
            - baseline_dev["worst_decile_profile_recall_at_3"]
        )
        concentration_names = (
            "source_balanced_maui_at_3", "false_top3_hhi",
            "false_top3_gini", "maximum_false_top3_share",
        )
        metrics["concentration_index_vs_whitened"] = float(np.mean([
            metrics[key] / max(baseline_dev[key], 1e-12) for key in concentration_names
        ]))
        eligible = (
            family != "whitened_cosine"
            and metrics["mrr_delta_vs_whitened"] >= -0.01
            and metrics["recall_at_3_delta_vs_whitened"] >= -0.01
            and metrics["worst_decile_delta_vs_whitened"] >= -0.02
            and metrics["source_balanced_maui_at_3"] < baseline_dev["source_balanced_maui_at_3"]
            and metrics["false_top3_hhi"] <= baseline_dev["false_top3_hhi"]
            and metrics["false_top3_gini"] <= baseline_dev["false_top3_gini"]
            and metrics["maximum_false_top3_share"] < baseline_dev["maximum_false_top3_share"]
            and passes_subgroups
        )
        dev_rows.append({
            "candidate": name,
            "family": family,
            "parameters": json.dumps(parameters, sort_keys=True),
            **metrics,
            "subgroup_non_degradation": passes_subgroups,
            "dev_eligible": eligible,
        })
        key = (metrics["concentration_index_vs_whitened"], -metrics["mrr"])
        if eligible and (selected_key is None or key < selected_key):
            selected = (name, family, parameters, scores)
            selected_key = key

    dev_results = pd.DataFrame(dev_rows).sort_values(
        ["dev_eligible", "concentration_index_vs_whitened", "mrr"],
        ascending=[False, True, False],
    )
    dev_results.to_csv(args.output_dir / "postwhitening_dev_search.csv", index=False)
    if dev_watched_rows:
        pd.concat(dev_watched_rows, ignore_index=True).to_csv(
            args.output_dir / "postwhitening_dev_watched_profiles.csv", index=False
        )
    geometry = profile_geometry(
        train, train_embeddings, profiles, popularity, exposure_mass, args.shrinkage
    )
    geometry.to_csv(args.output_dir / "postwhitening_profile_geometry.csv", index=False)

    if selected is None:
        report = {
            "status": "no_dev_candidate_passed_postwhitening_gate",
            "fixed_baseline": f"whitened_cosine:{args.shrinkage:g}",
            "seed": args.seed,
            "production_change_authorized": False,
        }
        (args.output_dir / "postwhitening_selection.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )
        print(json.dumps(report, indent=2))
        print(f"RETURN: {args.output_dir / 'postwhitening_selection.json'}")
        return

    name, family, parameters, selected_scores = selected
    comparison_rows = []
    exposure_rows = []
    subgroup_rows = []
    for position, (method, scores) in enumerate((
        (f"whitened_cosine:{args.shrinkage:g}", baseline[test_mask]),
        (name, selected_scores[test_mask]),
    )):
        metrics, table = evaluate(scores, test_labels, profiles, test_frame)
        estimates = intervals(scores, test_labels, args.bootstrap_runs, args.seed + 100 * position)
        comparison_rows.append({
            "method": method,
            **metrics,
            **{f"{key}_ci_low": value["ci_low"] for key, value in estimates.items()},
            **{f"{key}_ci_high": value["ci_high"] for key, value in estimates.items()},
        })
        table.insert(0, "method", method)
        exposure_rows.append(table)
        subgroup_rows.append(subgroup_table(method, test_frame, scores, test_labels))
    comparison = pd.DataFrame(comparison_rows)
    exposure_frame = pd.concat(exposure_rows, ignore_index=True)
    subgroups = pd.concat(subgroup_rows, ignore_index=True)
    test_subgroup_pass, _ = subgroup_pass(
        test_frame, test_labels, baseline[test_mask], selected_scores[test_mask]
    )
    deltas = {
        metric: delta_interval(
            baseline[test_mask], selected_scores[test_mask], test_labels,
            metric, args.bootstrap_runs, args.seed + 1000 + offset,
        )
        for offset, metric in enumerate(("mrr", "recall_at_3"))
    }
    watched = exposure_frame[exposure_frame["profile"].isin(args.watch_profile)].copy()
    watched_pivot = watched.pivot(
        index="profile", columns="method", values="source_balanced_false_top3_share"
    ) if len(watched) else pd.DataFrame()
    requested_watches = set(args.watch_profile)
    watched_non_worsening = requested_watches.issubset(set(watched_pivot.index)) and all(
        row.get(name, np.inf) <= row.get(f"whitened_cosine:{args.shrinkage:g}", -np.inf)
        for _, row in watched_pivot.iterrows()
    )
    old, new = comparison.iloc[0], comparison.iloc[1]
    diagnostic_gate = (
        deltas["mrr"]["ci_low"] >= -0.01
        and deltas["recall_at_3"]["ci_low"] >= -0.01
        and new["source_balanced_maui_at_3"] < old["source_balanced_maui_at_3"]
        and new["false_top3_hhi"] <= old["false_top3_hhi"]
        and new["false_top3_gini"] <= old["false_top3_gini"]
        and new["maximum_false_top3_share"] < old["maximum_false_top3_share"]
        and test_subgroup_pass
        and watched_non_worsening
    )
    report = {
        "status": "exploratory_existing_test_reused",
        "fixed_baseline": f"whitened_cosine:{args.shrinkage:g}",
        "seed": args.seed,
        "selection_unit": "dev only; watched authors are diagnostics, not tuning targets",
        "selected_candidate": name,
        "selected_family": family,
        "selected_parameters": parameters,
        "paired_test_deltas": deltas,
        "test_subgroup_non_degradation": test_subgroup_pass,
        "watched_profiles_non_worsening": watched_non_worsening,
        "watched_profiles_missing": sorted(requested_watches - set(watched_pivot.index)),
        "test_diagnostic_gate": bool(diagnostic_gate),
        "production_change_authorized": False,
        "reason": "the test split has already been opened; any passing method requires fresh-source confirmation",
    }
    comparison.to_csv(args.output_dir / "postwhitening_test_diagnostic.csv", index=False)
    exposure_frame.to_csv(args.output_dir / "postwhitening_author_exposure.csv", index=False)
    watched.to_csv(args.output_dir / "postwhitening_watched_profiles.csv", index=False)
    subgroups.to_csv(args.output_dir / "postwhitening_subgroup_metrics.csv", index=False)
    (args.output_dir / "postwhitening_selection.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    print(f"RETURN: {args.output_dir / 'postwhitening_selection.json'}")


if __name__ == "__main__":
    main()
