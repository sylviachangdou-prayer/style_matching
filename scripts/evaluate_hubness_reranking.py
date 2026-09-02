#!/usr/bin/env python3
"""Cross-fitted tests for concentrated author retrieval without re-encoding."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--bootstrap-runs", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260902)
    return parser.parse_args()


def source_key(frame: pd.DataFrame) -> pd.Series:
    identity = frame.get("independent_source_id", frame["source_id"])
    return frame["corpus"].astype(str) + "::" + identity.fillna("").astype(str)


def reciprocal_ranks(scores: np.ndarray, labels: np.ndarray) -> np.ndarray:
    order = np.argsort(scores, axis=1)[:, ::-1]
    return np.asarray([
        1.0 / (int(np.flatnonzero(row == true)[0]) + 1)
        for row, true in zip(order, labels)
    ])


def metric_values(scores: np.ndarray, labels: np.ndarray) -> dict[str, np.ndarray]:
    order = np.argsort(scores, axis=1)[:, ::-1]
    ranks = np.asarray([
        int(np.flatnonzero(row == true)[0]) + 1
        for row, true in zip(order, labels)
    ])
    values = {"mrr": 1.0 / ranks}
    for cutoff in (1, 3, 5, 20):
        values[f"recall_at_{cutoff}"] = (ranks <= cutoff).astype("float64")
    return values


def profile_summary(values: np.ndarray, labels: np.ndarray) -> np.ndarray:
    return np.asarray([values[labels == label].mean() for label in np.unique(labels)])


def intervals(
    scores: np.ndarray,
    labels: np.ndarray,
    runs: int,
    seed: int,
) -> dict[str, dict[str, float]]:
    rng = np.random.default_rng(seed)
    result = {}
    for position, (name, values) in enumerate(metric_values(scores, labels).items()):
        by_profile = profile_summary(values, labels)
        local_rng = np.random.default_rng(rng.integers(0, 2**32) + position)
        draws = by_profile[
            local_rng.integers(0, len(by_profile), (runs, len(by_profile)))
        ].mean(axis=1)
        result[name] = {
            "value": float(by_profile.mean()),
            "ci_low": float(np.quantile(draws, 0.025)),
            "ci_high": float(np.quantile(draws, 0.975)),
        }
    return result


def delta_interval(
    baseline: np.ndarray,
    challenger: np.ndarray,
    labels: np.ndarray,
    metric: str,
    runs: int,
    seed: int,
) -> dict[str, float]:
    old = profile_summary(metric_values(baseline, labels)[metric], labels)
    new = profile_summary(metric_values(challenger, labels)[metric], labels)
    delta = new - old
    rng = np.random.default_rng(seed)
    draws = delta[rng.integers(0, len(delta), (runs, len(delta)))].mean(axis=1)
    return {
        "delta": float(delta.mean()),
        "ci_low": float(np.quantile(draws, 0.025)),
        "ci_high": float(np.quantile(draws, 0.975)),
    }


def mask_languages(
    scores: np.ndarray,
    query_languages: np.ndarray,
    profile_languages: np.ndarray,
) -> np.ndarray:
    result = scores.copy()
    result[query_languages[:, None] != profile_languages[None, :]] = -np.inf
    return result


def rank_percentiles(
    scores: np.ndarray,
    query_languages: np.ndarray,
    profile_languages: np.ndarray,
) -> np.ndarray:
    result = np.zeros_like(scores, dtype="float64")
    for language in np.unique(query_languages):
        rows = np.flatnonzero(query_languages == language)
        columns = np.flatnonzero(profile_languages == language)
        if not len(rows) or not len(columns):
            continue
        order = np.argsort(np.argsort(scores[np.ix_(rows, columns)], axis=1), axis=1)
        result[np.ix_(rows, columns)] = (order + 1) / len(columns)
    return mask_languages(result, query_languages, profile_languages)


def empirical_percentiles(
    fit_scores: np.ndarray,
    fit_labels: np.ndarray,
    fit_languages: np.ndarray,
    fit_corpora: np.ndarray,
    valid_scores: np.ndarray,
    valid_languages: np.ndarray,
    valid_corpora: np.ndarray,
    profile_languages: np.ndarray,
) -> np.ndarray:
    result = np.zeros_like(valid_scores, dtype="float64")
    for candidate, language in enumerate(profile_languages):
        language_fit = (fit_languages == language) & (fit_labels != candidate)
        for corpus in np.unique(valid_corpora[valid_languages == language]):
            valid = (valid_languages == language) & (valid_corpora == corpus)
            reference = fit_scores[
                language_fit & (fit_corpora == corpus), candidate
            ]
            if len(reference) < 30:
                reference = fit_scores[language_fit, candidate]
            reference = np.sort(reference[np.isfinite(reference)])
            if not len(reference):
                continue
            result[valid, candidate] = (
                np.searchsorted(reference, valid_scores[valid, candidate], side="right") + 0.5
            ) / (len(reference) + 1.0)
    return mask_languages(result, valid_languages, profile_languages)


def local_density_bias(
    scores: np.ndarray,
    labels: np.ndarray,
    query_languages: np.ndarray,
    profile_languages: np.ndarray,
    top_k: int,
) -> np.ndarray:
    bias = np.zeros(scores.shape[1], dtype="float64")
    for language in np.unique(profile_languages):
        candidates = np.flatnonzero(profile_languages == language)
        rows = np.flatnonzero(query_languages == language)
        values = np.zeros(len(candidates), dtype="float64")
        for local, candidate in enumerate(candidates):
            eligible = scores[rows[labels[rows] != candidate], candidate]
            k = min(top_k, len(eligible))
            values[local] = np.sort(eligible)[-k:].mean() if k else 0.0
        scale = values.std()
        bias[candidates] = (values - values.mean()) / (scale if scale > 1e-12 else 1.0)
    return bias


def reciprocal_rank_fusion(
    first: np.ndarray,
    second: np.ndarray,
    query_languages: np.ndarray,
    profile_languages: np.ndarray,
    constant: int = 10,
) -> np.ndarray:
    result = np.full_like(first, -np.inf, dtype="float64")
    for language in np.unique(query_languages):
        rows = np.flatnonzero(query_languages == language)
        columns = np.flatnonzero(profile_languages == language)
        for row in rows:
            first_order = np.argsort(first[row, columns])[::-1]
            second_order = np.argsort(second[row, columns])[::-1]
            first_rank = np.empty(len(columns), dtype=int)
            second_rank = np.empty(len(columns), dtype=int)
            first_rank[first_order] = np.arange(1, len(columns) + 1)
            second_rank[second_order] = np.arange(1, len(columns) + 1)
            result[row, columns] = (
                1.0 / (constant + first_rank) + 1.0 / (constant + second_rank)
            )
    return result


def gini(values: np.ndarray) -> float:
    values = np.sort(values.astype("float64"))
    if not len(values) or values.sum() == 0:
        return 0.0
    positions = np.arange(1, len(values) + 1)
    return float((2 * np.sum(positions * values) / values.sum() - len(values) - 1) / len(values))


def exposure(
    scores: np.ndarray,
    labels: np.ndarray,
    profiles: np.ndarray,
    groups: np.ndarray | None = None,
    top_k: int = 3,
) -> tuple[dict[str, float], pd.DataFrame]:
    false_counts = np.zeros(len(profiles), dtype=int)
    all_counts = np.zeros(len(profiles), dtype=int)
    if groups is None:
        weights = np.ones(len(scores), dtype="float64")
    else:
        counts = pd.Series(groups).value_counts().to_dict()
        weights = np.asarray([1.0 / counts[group] for group in groups])
    balanced_false = np.zeros(len(profiles), dtype="float64")
    for values, true, weight in zip(scores, labels, weights):
        eligible = np.flatnonzero(np.isfinite(values))
        row = eligible[np.argsort(values[eligible])[::-1][:top_k]]
        all_counts[row] += 1
        false_counts[row[row != true]] += 1
        balanced_false[row[row != true]] += weight
    shares = balanced_false / max(balanced_false.sum(), 1)
    table = pd.DataFrame({
        "profile": profiles,
        "top3_count": all_counts,
        "false_top3_count": false_counts,
        "source_balanced_false_top3_mass": balanced_false,
        "source_balanced_false_top3_share": shares,
    }).sort_values(["false_top3_count", "top3_count"], ascending=False)
    return {
        "false_top3_gini": gini(balanced_false),
        "false_top3_hhi": float(np.square(shares).sum()),
        "maximum_false_top3_share": float(shares.max(initial=0.0)),
    }, table


def main() -> None:
    args = parse_args()
    frame = pd.read_parquet(args.input)
    payload = np.load(args.scores, allow_pickle=False)
    profiles = payload["profiles"].astype(str)
    labels = payload["y_true"].astype(int)
    chunk_ids = payload["chunk_ids"].astype(str)
    centroid = payload["single_centroid_scores"].astype("float64")
    prototype = payload["source_prototype_scores"].astype("float64")
    query_languages = payload["query_languages"].astype(str)
    query_corpora = payload["query_corpora"].astype(str)
    profile_languages = np.asarray([profile.split("::", 1)[0] for profile in profiles])

    lookup = frame.drop_duplicates("chunk_id").copy()
    lookup["chunk_id"] = lookup["chunk_id"].astype(str)
    aligned = lookup.set_index("chunk_id").loc[chunk_ids]
    groups = source_key(aligned).to_numpy()
    if len(np.unique(groups)) < args.folds:
        raise ValueError("Not enough independent sources for cross-fitting")

    centroid = mask_languages(centroid, query_languages, profile_languages)
    prototype = mask_languages(prototype, query_languages, profile_languages)
    methods = {
        "uncorrected_centroid": centroid,
        "source_prototype": prototype,
        "centroid_prototype_rrf": reciprocal_rank_fusion(
            centroid, prototype, query_languages, profile_languages
        ),
    }
    calibrated_centroid = np.full_like(centroid, -np.inf)
    calibrated_prototype = np.full_like(prototype, -np.inf)
    micro_lambdas = (0.00025, 0.0005, 0.001, 0.002, 0.004, 0.008, 0.012)
    micro_scores = {value: np.full_like(centroid, -np.inf) for value in micro_lambdas}
    splitter = GroupKFold(n_splits=args.folds)
    for fit, valid in splitter.split(centroid, labels, groups):
        calibrated_centroid[valid] = empirical_percentiles(
            centroid[fit], labels[fit], query_languages[fit], query_corpora[fit],
            centroid[valid], query_languages[valid], query_corpora[valid], profile_languages,
        )
        calibrated_prototype[valid] = empirical_percentiles(
            prototype[fit], labels[fit], query_languages[fit], query_corpora[fit],
            prototype[valid], query_languages[valid], query_corpora[valid], profile_languages,
        )
        bias = local_density_bias(
            centroid[fit], labels[fit], query_languages[fit], profile_languages, args.top_k
        )
        for value in micro_lambdas:
            micro_scores[value][valid] = mask_languages(
                centroid[valid] - value * bias[None, :],
                query_languages[valid], profile_languages,
            )

    query_percentile = rank_percentiles(centroid, query_languages, profile_languages)
    methods["candidate_empirical_null"] = calibrated_centroid
    methods["mutual_proximity"] = mask_languages(
        np.sqrt(np.clip(calibrated_centroid, 0, 1) * np.clip(query_percentile, 0, 1)),
        query_languages, profile_languages,
    )
    methods["calibrated_multiview"] = mask_languages(
        np.sqrt(np.clip(calibrated_centroid, 0, 1) * np.clip(calibrated_prototype, 0, 1)),
        query_languages, profile_languages,
    )
    methods.update({f"micro_density_{value:g}": scores for value, scores in micro_scores.items()})

    baseline = methods["uncorrected_centroid"]
    report = {
        "design": "exploratory source-grouped cross-fitting on the existing evaluation pool; no re-encoding",
        "seed": args.seed,
        "n_queries": int(len(labels)),
        "n_profiles": int(len(profiles)),
        "methods": {},
    }
    exposure_tables = []
    for position, (name, scores) in enumerate(methods.items()):
        concentration, table = exposure(scores, labels, profiles, groups)
        metrics = intervals(scores, labels, args.bootstrap_runs, args.seed + position)
        comparison = {
            metric: delta_interval(
                baseline, scores, labels, metric, args.bootstrap_runs,
                args.seed + 100 + 10 * position + offset,
            )
            for offset, metric in enumerate(("mrr", "recall_at_3"))
        }
        by_language = {}
        for language in np.unique(query_languages):
            rows = np.flatnonzero(query_languages == language)
            by_language[language] = {
                "n_queries": int(len(rows)),
                "n_profiles": int(len(np.unique(labels[rows]))),
                "metrics": intervals(
                    scores[rows], labels[rows], args.bootstrap_runs,
                    args.seed + 1000 + position,
                ),
            }
        report["methods"][name] = {
            "metrics": metrics,
            "delta_vs_uncorrected": comparison,
            "concentration": concentration,
            "by_language": by_language,
        }
        table.insert(0, "method", name)
        exposure_tables.append(table)

    baseline_concentration = report["methods"]["uncorrected_centroid"]["concentration"]
    eligible = []
    for name, result in report["methods"].items():
        mrr = result["delta_vs_uncorrected"]["mrr"]
        recall = result["delta_vs_uncorrected"]["recall_at_3"]
        concentration = result["concentration"]
        if (
            mrr["ci_low"] >= -0.01
            and recall["ci_low"] >= -0.01
            and concentration["false_top3_hhi"] <= baseline_concentration["false_top3_hhi"]
        ):
            eligible.append(name)
    selected = min(
        eligible,
        key=lambda name: (
            report["methods"][name]["concentration"]["false_top3_hhi"],
            -report["methods"][name]["metrics"]["mrr"]["value"],
        ),
    ) if eligible else "uncorrected_centroid"
    report["selection_rule"] = (
        "minimize false-top3 HHI subject to paired-profile MRR and Recall@3 "
        "95% CI lower bounds >= -0.01 versus the uncorrected centroid"
    )
    report["eligible_methods"] = eligible
    report["selected_exploratory_method"] = selected

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "hubness_reranking_metrics.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    pd.concat(exposure_tables, ignore_index=True).to_csv(
        args.output_dir / "hubness_reranking_author_exposure.csv", index=False
    )
    print(json.dumps({
        "selected_exploratory_method": selected,
        "eligible_methods": eligible,
        "selected": report["methods"][selected],
    }, indent=2))
    print(f"RETURN: {output}")


if __name__ == "__main__":
    main()
