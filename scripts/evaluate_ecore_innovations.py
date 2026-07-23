"""Falsifiable evaluation of the first three ECoRe innovations.

The encoder stays frozen.  The script tests, in order:
1. source-balanced soft multi-prototypes versus one author centroid;
2. environment-centred cohort-relative geometry versus absolute prototypes;
3. a candidate-identity-free episodic pairwise scorer on wholly held-out authors.

No output from this script is a calibrated probability.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import normalize

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.style_embedding_recall import balanced_train, profile_key


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--embedding-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--temperature", type=float, default=0.08)
    parser.add_argument("--author-folds", type=int, default=5)
    parser.add_argument("--hard-negatives", type=int, default=12)
    parser.add_argument("--bootstrap-runs", type=int, default=5000)
    parser.add_argument("--train-cap", type=int, default=300)
    parser.add_argument(
        "--embedding-seed",
        type=int,
        default=20260701,
        help="Seed used by style_embedding_recall when the reused train embeddings were written.",
    )
    parser.add_argument("--seed", type=int, default=20260725)
    return parser.parse_args()


def source_keys(frame: pd.DataFrame) -> pd.Series:
    identity = (
        frame["independent_source_id"]
        if "independent_source_id" in frame
        else frame["source_id"]
    )
    return frame["corpus"].astype(str) + "::" + identity.fillna("").astype(str)


def environment(frame: pd.DataFrame) -> pd.Series:
    register = (
        frame["register"].fillna("").astype(str)
        if "register" in frame
        else frame["corpus"].fillna("").astype(str)
    )
    register = register.mask(register.eq(""), frame["corpus"].fillna("").astype(str))
    return frame["language"].astype(str) + "::" + register


def logmeanexp(values: np.ndarray, temperature: float) -> np.ndarray:
    scaled = values / temperature
    maximum = scaled.max(axis=1, keepdims=True)
    return temperature * (
        maximum[:, 0]
        + np.log(np.exp(scaled - maximum).mean(axis=1).clip(1e-12))
    )


def ranking_metrics(scores: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    order = np.argsort(scores, axis=1)[:, ::-1]
    ranks = np.asarray(
        [int(np.flatnonzero(row == true)[0]) + 1 for row, true in zip(order, labels)]
    )
    return {
        "mrr": float(np.mean(1.0 / ranks)),
        "recall_at_1": float(np.mean(ranks <= 1)),
        "recall_at_3": float(np.mean(ranks <= 3)),
        "recall_at_5": float(np.mean(ranks <= 5)),
    }


def macro_metrics(scores: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    rows = [
        ranking_metrics(scores[labels == label], labels[labels == label])
        for label in np.unique(labels)
    ]
    return {
        key: float(np.mean([row[key] for row in rows]))
        for key in rows[0]
    }


def paired_profile_bootstrap(
    baseline: np.ndarray,
    challenger: np.ndarray,
    labels: np.ndarray,
    runs: int,
    seed: int,
) -> dict[str, float]:
    def reciprocal_ranks(matrix: np.ndarray) -> np.ndarray:
        order = np.argsort(matrix, axis=1)[:, ::-1]
        return np.asarray(
            [1.0 / (int(np.flatnonzero(row == true)[0]) + 1) for row, true in zip(order, labels)]
        )

    delta = reciprocal_ranks(challenger) - reciprocal_ranks(baseline)
    by_profile = np.asarray(
        [delta[labels == label].mean() for label in np.unique(labels)]
    )
    rng = np.random.default_rng(seed)
    draws = by_profile[
        rng.integers(0, len(by_profile), size=(runs, len(by_profile)))
    ].mean(axis=1)
    return {
        "mrr_delta": float(by_profile.mean()),
        "ci_low": float(np.quantile(draws, 0.025)),
        "ci_high": float(np.quantile(draws, 0.975)),
    }


def make_prototypes(
    train: pd.DataFrame, embeddings: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    work = train.copy().reset_index(drop=True)
    work["source_key"] = source_keys(work)
    work["environment"] = environment(work)
    vectors, labels, envs, supports = [], [], [], []
    for (profile, key), positions in work.groupby(
        ["profile_key", "source_key"], sort=True
    ).indices.items():
        vectors.append(embeddings[np.asarray(positions)].mean(axis=0))
        labels.append(profile)
        envs.append(work.iloc[np.asarray(positions)]["environment"].mode().iat[0])
        supports.append(len(positions))
    return (
        normalize(np.vstack(vectors)),
        np.asarray(labels),
        np.asarray(envs),
        np.asarray(supports),
    )


def cohort_centres(
    prototypes: np.ndarray, prototype_envs: np.ndarray, profile_languages: np.ndarray
) -> dict[str, np.ndarray]:
    centres = {}
    for env in np.unique(prototype_envs):
        centres[str(env)] = normalize(
            prototypes[prototype_envs == env].mean(axis=0, keepdims=True)
        )[0]
    for language in np.unique(profile_languages):
        mask = np.char.startswith(prototype_envs.astype(str), f"{language}::")
        centres[f"{language}::__fallback__"] = normalize(
            prototypes[mask].mean(axis=0, keepdims=True)
        )[0]
    return centres


def score_views(
    queries: np.ndarray,
    query_envs: np.ndarray,
    profiles: np.ndarray,
    prototypes: np.ndarray,
    prototype_profiles: np.ndarray,
    prototype_envs: np.ndarray,
    temperature: float,
    centres: dict[str, np.ndarray],
    support_cap: int | None = None,
) -> dict[str, np.ndarray]:
    n_query, n_profile = len(queries), len(profiles)
    centroid = np.full((n_query, n_profile), -1e9, dtype="float64")
    hard = centroid.copy()
    soft = centroid.copy()
    cohort = centroid.copy()
    dispersion = np.zeros((n_query, n_profile), dtype="float64")
    query_languages = np.asarray([str(env).split("::", 1)[0] for env in query_envs])
    profile_languages = np.asarray([str(p).split("::", 1)[0] for p in profiles])

    for label, profile in enumerate(profiles):
        valid_rows = np.flatnonzero(query_languages == profile_languages[label])
        if not len(valid_rows):
            continue
        modes = prototypes[prototype_profiles == profile]
        if support_cap is not None:
            modes = modes[:support_cap]
        absolute = queries[valid_rows] @ modes.T
        centroid[valid_rows, label] = (
            queries[valid_rows] @ normalize(modes.mean(axis=0, keepdims=True))[0]
        )
        hard[valid_rows, label] = absolute.max(axis=1)
        soft[valid_rows, label] = logmeanexp(absolute, temperature)
        dispersion[valid_rows, label] = (
            absolute.std(axis=1) if len(modes) > 1 else 0.0
        )
        for env in np.unique(query_envs[valid_rows].astype(str)):
            rows = valid_rows[query_envs[valid_rows].astype(str) == env]
            centre = centres.get(
                env,
                centres[f"{profile_languages[label]}::__fallback__"],
            )
            q_resid = normalize(queries[rows] - centre)
            p_resid = normalize(modes - centre)
            cohort[rows, label] = logmeanexp(q_resid @ p_resid.T, temperature)
    return {
        "centroid": centroid,
        "hard_prototype": hard,
        "soft_prototype": soft,
        "cohort_relative": cohort,
        "prototype_dispersion": dispersion,
    }


def shuffled_environments(
    prototype_envs: np.ndarray, seed: int
) -> np.ndarray:
    shuffled = prototype_envs.copy()
    rng = np.random.default_rng(seed)
    languages = np.asarray([str(env).split("::", 1)[0] for env in prototype_envs])
    for language in np.unique(languages):
        positions = np.flatnonzero(languages == language)
        shuffled[positions] = rng.permutation(shuffled[positions])
    return shuffled


def aggregate_by_source(
    frame: pd.DataFrame,
    labels: np.ndarray,
    views: dict[str, np.ndarray],
) -> tuple[pd.DataFrame, np.ndarray, dict[str, np.ndarray]]:
    work = frame.copy().reset_index(drop=True)
    work["_source_key"] = source_keys(work)
    groups = list(work.groupby("_source_key", sort=True).indices.items())
    source_frame = pd.DataFrame([
        {
            "source_key": key,
            "chunk_id": str(work.iloc[np.asarray(positions)]["chunk_id"].iat[0]),
        }
        for key, positions in groups
    ])
    source_labels = np.asarray([
        int(np.unique(labels[np.asarray(positions)])[0]) for _, positions in groups
    ])
    source_views = {
        name: np.vstack([
            matrix[np.asarray(positions)].mean(axis=0) for _, positions in groups
        ])
        for name, matrix in views.items()
    }
    return source_frame, source_labels, source_views


def episodic_crossfit(
    dev_views: dict[str, np.ndarray],
    test_views: dict[str, np.ndarray],
    dev_labels: np.ndarray,
    test_labels: np.ndarray,
    profiles: np.ndarray,
    folds: int,
    hard_negatives: int,
    seed: int,
) -> tuple[np.ndarray, list[dict]]:
    feature_names = (
        "centroid", "hard_prototype", "soft_prototype",
        "cohort_relative", "prototype_dispersion",
    )
    dev_x = np.stack([dev_views[name] for name in feature_names], axis=-1)
    test_x = np.stack([test_views[name] for name in feature_names], axis=-1)
    unique = np.unique(test_labels)
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(unique)
    buckets = [shuffled[index::min(folds, len(unique))] for index in range(min(folds, len(unique)))]
    output = np.full(test_views["centroid"].shape, -1e9, dtype="float64")
    audit = []
    profile_languages = np.asarray([str(profile).split("::", 1)[0] for profile in profiles])

    for fold, heldout in enumerate(buckets):
        print(
            f"episodic fold {fold + 1}/{len(buckets)}: "
            f"{len(heldout)} wholly held-out profiles",
            flush=True,
        )
        train_rows = np.flatnonzero(~np.isin(dev_labels, heldout))
        diffs = []
        for row in train_rows:
            true = int(dev_labels[row])
            language = profile_languages[true]
            candidates = np.flatnonzero(
                (profile_languages == language)
                & (~np.isin(np.arange(len(profiles)), heldout))
                & (np.arange(len(profiles)) != true)
            )
            if not len(candidates):
                continue
            hardest = candidates[
                np.argsort(dev_views["soft_prototype"][row, candidates])[::-1][
                    :hard_negatives
                ]
            ]
            diffs.extend(dev_x[row, true] - dev_x[row, negative] for negative in hardest)
        if not diffs:
            raise ValueError(f"Fold {fold} has no author-disjoint episodic comparisons")
        positive = np.vstack(diffs)
        pair_x = np.vstack([positive, -positive])
        pair_y = np.concatenate([np.ones(len(positive)), np.zeros(len(positive))])
        model = LogisticRegression(
            fit_intercept=False, C=0.25, max_iter=2000, random_state=seed + fold
        ).fit(pair_x, pair_y)
        heldout_rows = np.flatnonzero(np.isin(test_labels, heldout))
        output[heldout_rows] = (
            test_x[heldout_rows] @ model.coef_[0]
        )
        audit.append({
            "fold": fold,
            "heldout_profiles": profiles[heldout].tolist(),
            "n_pairwise_differences": int(len(positive)),
            "feature_weights": dict(zip(feature_names, model.coef_[0].tolist())),
        })
    return output, audit


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.read_parquet(args.input).copy()
    frame["profile_key"] = profile_key(frame)
    train = balanced_train(frame, args.train_cap, args.embedding_seed)
    eval_frame = frame[frame["split"].isin(["dev", "test"])].copy().reset_index(drop=True)
    print(
        f"loaded frozen split: {len(train)} balanced train chunks; "
        f"{len(eval_frame)} dev/test chunks; {frame['profile_key'].nunique()} profiles",
        flush=True,
    )
    train_embeddings = np.load(args.embedding_dir / "style_embedding_train_embeddings.npy")
    eval_embeddings = np.load(args.embedding_dir / "style_embedding_eval_embeddings.npy")
    if len(train) != len(train_embeddings) or len(eval_frame) != len(eval_embeddings):
        raise ValueError("Embedding rows do not align with the frozen split; regenerate this embedding artifact")
    # style_embedding_recall historically stored string metadata as object
    # arrays. This is a trusted, locally generated Part 7 artifact; NumPy 2.x
    # requires an explicit opt-in to read those arrays.
    reference = np.load(
        args.embedding_dir / "style_embedding_scores.npz",
        allow_pickle=True,
    )
    if not np.array_equal(
        reference["chunk_ids"].astype(str),
        eval_frame["chunk_id"].astype(str).to_numpy(),
    ):
        raise ValueError("Evaluation embedding order does not match style_embedding_scores.npz")
    print("Part 7 embedding alignment verified", flush=True)

    profiles = np.asarray(sorted(frame["profile_key"].unique()))
    profile_to_label = {profile: index for index, profile in enumerate(profiles)}
    eval_labels = eval_frame["profile_key"].map(profile_to_label).to_numpy()
    prototype_vectors, prototype_profiles, prototype_envs, _ = make_prototypes(
        train, train_embeddings
    )
    print(
        f"built {len(prototype_vectors)} independent-source prototypes; "
        "scoring centroid, hard, soft, and cohort-relative views",
        flush=True,
    )
    centres = cohort_centres(
        prototype_vectors,
        prototype_envs,
        np.asarray([profile.split("::", 1)[0] for profile in profiles]),
    )
    views = score_views(
        eval_embeddings, environment(eval_frame).to_numpy(), profiles,
        prototype_vectors, prototype_profiles, prototype_envs,
        args.temperature, centres,
    )
    support_views = {
        cap: score_views(
            eval_embeddings, environment(eval_frame).to_numpy(), profiles,
            prototype_vectors, prototype_profiles, prototype_envs,
            args.temperature, centres, support_cap=cap,
        )
        for cap in (1, 2)
    }
    print("fixed geometry and variable-support views scored", flush=True)
    shuffled_envs = shuffled_environments(prototype_envs, args.seed)
    shuffled_centres = cohort_centres(
        prototype_vectors,
        shuffled_envs,
        np.asarray([profile.split("::", 1)[0] for profile in profiles]),
    )
    views["shuffled_cohort_control"] = score_views(
        eval_embeddings, environment(eval_frame).to_numpy(), profiles,
        prototype_vectors, prototype_profiles, shuffled_envs,
        args.temperature, shuffled_centres,
    )["cohort_relative"]
    print("shuffled-environment negative control scored", flush=True)

    dev = eval_frame["split"].eq("dev").to_numpy()
    test = eval_frame["split"].eq("test").to_numpy()
    dev_frame, dev_labels, dev_views_full = aggregate_by_source(
        eval_frame.loc[dev].reset_index(drop=True),
        eval_labels[dev],
        {key: value[dev] for key, value in views.items()},
    )
    test_frame, test_labels, test_views = aggregate_by_source(
        eval_frame.loc[test].reset_index(drop=True),
        eval_labels[test],
        {key: value[test] for key, value in views.items()},
    )
    support_dev = {}
    support_test = {}
    for cap, variant in support_views.items():
        _, _, support_dev[cap] = aggregate_by_source(
            eval_frame.loc[dev].reset_index(drop=True),
            eval_labels[dev],
            {key: value[dev] for key, value in variant.items()},
        )
        _, _, support_test[cap] = aggregate_by_source(
            eval_frame.loc[test].reset_index(drop=True),
            eval_labels[test],
            {key: value[test] for key, value in variant.items()},
        )
    episodic_feature_names = (
        "centroid", "hard_prototype", "soft_prototype",
        "cohort_relative", "prototype_dispersion",
    )
    dev_variants = [dev_views_full, support_dev[1], support_dev[2]]
    episodic_dev = {
        key: np.concatenate([variant[key] for variant in dev_variants], axis=0)
        for key in episodic_feature_names
    }
    episodic_dev_labels = np.tile(dev_labels, len(dev_variants))
    episodic_by_support = {}
    fold_audit = None
    for support_label, variant in (("one_source", support_test[1]), ("two_sources", support_test[2]), ("all_sources", test_views)):
        print(f"author-heldout episodic transfer: {support_label}", flush=True)
        scores, audit = episodic_crossfit(
            episodic_dev,
            {key: variant[key] for key in episodic_feature_names},
            episodic_dev_labels, test_labels, profiles,
            args.author_folds, args.hard_negatives, args.seed,
        )
        episodic_by_support[support_label] = scores
        if support_label == "all_sources":
            fold_audit = audit
    episodic_scores = episodic_by_support["all_sources"]
    test_views["episodic_author_heldout"] = episodic_scores

    metrics = {
        name: macro_metrics(matrix, test_labels)
        for name, matrix in test_views.items()
        if name != "prototype_dispersion"
    }
    h1 = paired_profile_bootstrap(
        test_views["centroid"], test_views["soft_prototype"], test_labels,
        args.bootstrap_runs, args.seed,
    )
    h2 = paired_profile_bootstrap(
        test_views["soft_prototype"], test_views["cohort_relative"], test_labels,
        args.bootstrap_runs, args.seed + 1,
    )
    h2_control = paired_profile_bootstrap(
        test_views["shuffled_cohort_control"], test_views["cohort_relative"], test_labels,
        args.bootstrap_runs, args.seed + 2,
    )
    h3 = paired_profile_bootstrap(
        test_views["cohort_relative"], test_views["episodic_author_heldout"], test_labels,
        args.bootstrap_runs, args.seed + 3,
    )
    report = {
        "status": "research_evaluation_not_calibrated",
        "input": str(args.input),
        "embedding_dir": str(args.embedding_dir),
        "embedding_alignment": {
            "train_cap": args.train_cap,
            "embedding_seed": args.embedding_seed,
            "n_train_rows": int(len(train)),
            "n_eval_rows": int(len(eval_frame)),
        },
        "environment": "language × register with language fallback",
        "candidate_identity_parameters": False,
        "test_metrics": metrics,
        "innovation_1_distributional_profile": {
            "contrast": "source-balanced soft multi-prototype minus centroid",
            "paired_profile_bootstrap": h1,
            "supported": h1["ci_low"] > 0,
        },
        "innovation_2_cohort_relative_geometry": {
            "contrast": "environment-centred residual soft prototype minus absolute soft prototype",
            "paired_profile_bootstrap": h2,
            "shuffled_environment_control": h2_control,
            "supported": h2["ci_low"] > 0 and h2_control["ci_low"] > 0,
        },
        "innovation_3_episodic_transfer": {
            "contrast": "whole-author cross-fitted episodic scorer minus fixed cohort energy",
            "paired_profile_bootstrap": h3,
            "supported": h3["ci_low"] > 0,
            "variable_support_test_metrics": {
                label: macro_metrics(scores, test_labels)
                for label, scores in episodic_by_support.items()
            },
            "variable_support_training": "dev episodes pooled deterministic 1-source, 2-source, and all-source support conditions",
            "folds": fold_audit,
        },
        "interpretation_rule": (
            "An innovation is supported only when its paired profile-bootstrap "
            "95% interval is wholly above zero; innovation 2 must also beat the "
            "shuffled-environment control."
        ),
    }
    (args.output_dir / "ecore_innovation_metrics.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    np.savez_compressed(
        args.output_dir / "ecore_innovation_scores.npz",
        profiles=profiles,
        chunk_ids=test_frame["chunk_id"].astype(str).to_numpy(),
        y_true=test_labels,
        **{name: matrix.astype("float32") for name, matrix in test_views.items()},
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
