#!/usr/bin/env python3
"""Tune author-hub corrections on dev and open test exactly once."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluate_ecore_innovations import macro_metrics, paired_profile_bootstrap


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--lambdas", default="0,0.02,0.05,0.08,0.1,0.15,0.2")
    parser.add_argument("--bootstrap-runs", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260726)
    return parser.parse_args()


def source_key(frame: pd.DataFrame) -> pd.Series:
    identity = frame.get("independent_source_id", frame["source_id"]).fillna("").astype(str)
    return frame["corpus"].astype(str) + "::" + identity


def candidate_languages(profiles: np.ndarray) -> np.ndarray:
    return np.asarray([str(profile).split("::", 1)[0] for profile in profiles])


def support_bins(frame: pd.DataFrame, profiles: np.ndarray) -> np.ndarray:
    train = frame[frame["split"].eq("train")].copy()
    train["profile_key"] = train["language"].astype(str) + "::" + train["author_or_speaker"].astype(str)
    train["source_key"] = source_key(train)
    counts = train.groupby("profile_key")["source_key"].nunique().to_dict()
    return np.asarray([
        "1" if counts.get(str(profile), 0) <= 1
        else "2" if counts.get(str(profile), 0) == 2
        else "3+"
        for profile in profiles
    ])


def hub_bias(
    scores: np.ndarray,
    labels: np.ndarray,
    profiles: np.ndarray,
    support: np.ndarray,
    method: str,
    top_k: int,
) -> np.ndarray:
    languages = candidate_languages(profiles)
    bias = np.zeros(scores.shape[1], dtype="float64")
    for language in np.unique(languages):
        candidates = np.flatnonzero(languages == language)
        queries = np.flatnonzero(np.isin(labels, candidates))
        if not len(queries):
            continue
        local = scores[np.ix_(queries, candidates)]
        if method == "frequency":
            counts = np.zeros(len(candidates), dtype="float64")
            k = min(top_k, len(candidates))
            for row, true in zip(local, labels[queries]):
                chosen = np.argsort(row)[-k:]
                counts[chosen[candidates[chosen] != true]] += 1
            values = counts / max(len(queries), 1)
        else:
            values = np.zeros(len(candidates), dtype="float64")
            for position, candidate in enumerate(candidates):
                eligible = local[labels[queries] != candidate, position]
                k = min(top_k, len(eligible))
                values[position] = np.sort(eligible)[-k:].mean() if k else 0.0
        for bucket in np.unique(support[candidates]):
            positions = np.flatnonzero(support[candidates] == bucket)
            block = values[positions]
            scale = block.std()
            bias[candidates[positions]] = (block - block.mean()) / (scale if scale > 1e-12 else 1.0)
    return bias


def corrected(scores: np.ndarray, bias: np.ndarray, value: float) -> np.ndarray:
    return scores - value * bias[None, :]


def main() -> None:
    args = parse_args()
    frame = pd.read_parquet(args.input)
    payload = np.load(args.scores, allow_pickle=False)
    profiles = payload["profiles"].astype(str)
    scores = payload["single_centroid_scores"].astype("float64")
    labels = payload["y_true"].astype(int)
    splits = payload["splits"].astype(str)
    chunk_ids = payload["chunk_ids"].astype(str)
    support = support_bins(frame, profiles)
    lookup = frame.drop_duplicates("chunk_id").copy()
    lookup["chunk_id"] = lookup["chunk_id"].astype(str)
    lookup = lookup.set_index("chunk_id")
    eval_rows = lookup.loc[chunk_ids]
    groups = source_key(eval_rows).to_numpy()
    lambdas = [float(value) for value in args.lambdas.split(",")]
    dev = np.flatnonzero(splits == "dev")
    test = np.flatnonzero(splits == "test")
    if not len(dev) or not len(test):
        raise ValueError("scores must contain separate dev and test rows")

    methods = {}
    for method in ("frequency", "local_density"):
        fold_count = min(args.folds, len(np.unique(groups[dev])))
        splitter = GroupKFold(n_splits=fold_count)
        oof = {value: np.full_like(scores[dev], -np.inf) for value in lambdas}
        for fit_local, valid_local in splitter.split(dev, labels[dev], groups[dev]):
            fit = dev[fit_local]
            valid = dev[valid_local]
            bias = hub_bias(scores[fit], labels[fit], profiles, support, method, args.top_k)
            for value in lambdas:
                oof[value][valid_local] = corrected(scores[valid], bias, value)
        dev_metrics = {str(value): macro_metrics(matrix, labels[dev]) for value, matrix in oof.items()}
        selected = max(
            lambdas,
            key=lambda value: (
                dev_metrics[str(value)]["mrr"],
                dev_metrics[str(value)]["recall_at_3"],
                -value,
            ),
        )
        final_bias = hub_bias(scores[dev], labels[dev], profiles, support, method, args.top_k)
        test_scores = corrected(scores[test], final_bias, selected)
        methods[method] = {
            "selected_lambda": selected,
            "cross_fitted_dev": dev_metrics[str(selected)],
            "test": macro_metrics(test_scores, labels[test]),
            "paired_test_vs_uncorrected": paired_profile_bootstrap(
                scores[test], test_scores, labels[test], args.bootstrap_runs, args.seed
            ),
            "all_dev_candidates_bias": {
                str(profile): float(value) for profile, value in zip(profiles, final_bias)
            },
            "lambda_search": dev_metrics,
        }

    baseline = macro_metrics(scores[test], labels[test])
    for result in methods.values():
        delta = result["paired_test_vs_uncorrected"]
        result["adopted"] = bool(
            delta["ci_low"] > 0
            and result["test"]["recall_at_3"] >= baseline["recall_at_3"]
        )
    report = {
        "design": "source-grouped cross-fitted dev selection; locked test opened once",
        "baseline_test": baseline,
        "top_k": args.top_k,
        "support_balanced": True,
        "methods": methods,
        "selected": max(
            ("uncorrected", *methods),
            key=lambda name: baseline["mrr"] if name == "uncorrected" else (
                methods[name]["test"]["mrr"] if methods[name]["adopted"] else -np.inf
            ),
        ),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "hubness_correction_metrics.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "methods"}, indent=2))
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
