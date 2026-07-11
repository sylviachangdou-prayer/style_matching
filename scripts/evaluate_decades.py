from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import normalize


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate decade style with author-heldout splits.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--min-authors", type=int, default=5)
    parser.add_argument("--min-sources", type=int, default=20)
    parser.add_argument("--test-author-fraction", type=float, default=0.25)
    parser.add_argument("--bootstrap-runs", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260711)
    return parser.parse_args()


def encode(model_name: str, texts: list[str], batch_size: int, device: str) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name, device=device)
    return model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    ).astype("float32")


def bootstrap_balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray, runs: int, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    scores = []
    for _ in range(runs):
        positions = rng.integers(0, len(y_true), len(y_true))
        scores.append(balanced_accuracy_score(y_true[positions], y_pred[positions]))
    return float(np.quantile(scores, 0.025)), float(np.quantile(scores, 0.975))


def main() -> None:
    args = parse_args()
    df = pd.read_parquet(args.input)
    required = {"language", "corpus", "author_or_speaker", "source_id", "decade", "text"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    df = df[df["decade"].fillna("").astype(str).ne("")].copy().reset_index(drop=True)
    if df.empty:
        raise ValueError("No verified dated sources available")
    identity = df["independent_source_id"] if "independent_source_id" in df else df["source_id"]
    df["source_key"] = df["corpus"].astype(str) + "::" + identity.fillna("").astype(str)
    support = (
        df.groupby(["language", "corpus", "decade"])
        .agg(n_authors=("author_or_speaker", "nunique"), n_sources=("source_key", "nunique"))
        .reset_index()
    )
    eligible_classes = {
        (str(row.language), str(row.corpus), str(row.decade))
        for row in support.itertuples(index=False)
        if row.n_authors >= args.min_authors and row.n_sources >= args.min_sources
    }
    df = df[
        df.apply(lambda row: (str(row.language), str(row.corpus), str(row.decade)) in eligible_classes, axis=1)
    ].reset_index(drop=True)
    if df.empty:
        raise ValueError("No language-register-decade class passes support thresholds")
    embeddings = encode(args.model_name, df["text"].fillna("").tolist(), args.batch_size, args.device)
    rng = np.random.default_rng(args.seed)
    reports = {}
    for (language, corpus), group in df.groupby(["language", "corpus"], sort=True):
        decades = sorted(group["decade"].astype(str).unique())
        authors = np.asarray(sorted(group["author_or_speaker"].astype(str).unique()))
        if len(decades) < 2 or len(authors) < args.min_authors:
            continue
        shuffled = authors.copy()
        rng.shuffle(shuffled)
        n_test = max(1, int(len(shuffled) * args.test_author_fraction))
        test_authors = set(shuffled[:n_test])
        train_positions = group.index[~group["author_or_speaker"].isin(test_authors)].to_numpy()
        test_positions = group.index[group["author_or_speaker"].isin(test_authors)].to_numpy()
        if not len(train_positions) or not len(test_positions):
            continue
        if any(
            not df.loc[train_positions, "decade"].astype(str).eq(decade).any()
            or not df.loc[test_positions, "decade"].astype(str).eq(decade).any()
            for decade in decades
        ):
            reports[f"{language}::{corpus}"] = {
                "decades": decades,
                "status": "insufficient_author_heldout_class_coverage",
                "display_eligible": False,
            }
            continue
        centroids = normalize(np.vstack([
            embeddings[train_positions[df.loc[train_positions, "decade"].astype(str).eq(decade)]].mean(axis=0)
            for decade in decades
        ]))
        scores = embeddings[test_positions] @ centroids.T
        labels = np.asarray([decades.index(str(value)) for value in df.loc[test_positions, "decade"]])
        predictions = scores.argmax(axis=1)
        accuracy = float(balanced_accuracy_score(labels, predictions))
        ci_low, ci_high = bootstrap_balanced_accuracy(labels, predictions, args.bootstrap_runs, args.seed)
        chance = 1.0 / len(decades)
        reports[f"{language}::{corpus}"] = {
            "decades": decades,
            "n_train_authors": len(set(authors) - test_authors),
            "n_test_authors": len(test_authors),
            "n_test_chunks": len(test_positions),
            "balanced_accuracy": accuracy,
            "chance": chance,
            "bootstrap_ci_low": ci_low,
            "bootstrap_ci_high": ci_high,
            "display_eligible": accuracy >= 2.0 * chance and ci_low > chance,
        }
    report = {
        "model_name": args.model_name,
        "min_authors_per_class": args.min_authors,
        "min_sources_per_class": args.min_sources,
        "groups": reports,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "decade_metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    support.to_csv(args.out_dir / "decade_support.csv", index=False)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
