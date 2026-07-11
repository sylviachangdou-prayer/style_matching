from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import normalize

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.retrieval_metrics import misattribution_unfairness, open_set_metrics, ranking_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate author-heldout open-set rejection.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--language", required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20260711)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--per-profile-cap", type=int, default=50)
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


def profile_key(df: pd.DataFrame) -> pd.Series:
    return df["language"].astype(str) + "::" + df["author_or_speaker"].astype(str)


def cap_profiles(df: pd.DataFrame, cap: int, seed: int) -> pd.DataFrame:
    return (
        df.sample(frac=1, random_state=seed)
        .groupby("profile_key", group_keys=False)
        .head(cap)
        .reset_index(drop=True)
    )


def main() -> None:
    args = parse_args()
    df = pd.read_parquet(args.input)
    required = {"language", "author_or_speaker", "split", "text"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    df = df[df["language"].astype(str).eq(args.language)].copy()
    if df.empty:
        raise ValueError(f"No rows for language: {args.language}")
    df["profile_key"] = profile_key(df)
    profiles = np.asarray(sorted(df["profile_key"].unique()))
    if len(profiles) < 10:
        raise ValueError("Open-set evaluation requires at least 10 author-language profiles")
    rng = np.random.default_rng(args.seed)
    shuffled = profiles.copy()
    rng.shuffle(shuffled)
    n_unknown_dev = max(1, int(len(shuffled) * 0.15))
    n_unknown_test = max(1, int(len(shuffled) * 0.15))
    unknown_dev = set(shuffled[:n_unknown_dev])
    unknown_test = set(shuffled[n_unknown_dev:n_unknown_dev + n_unknown_test])
    known = set(shuffled[n_unknown_dev + n_unknown_test:])

    train = cap_profiles(df[df["profile_key"].isin(known) & df["split"].eq("train")], args.per_profile_cap, args.seed)
    known_dev = cap_profiles(df[df["profile_key"].isin(known) & df["split"].eq("dev")], args.per_profile_cap, args.seed)
    known_test = cap_profiles(df[df["profile_key"].isin(known) & df["split"].eq("test")], args.per_profile_cap, args.seed)
    unknown_dev_df = cap_profiles(df[df["profile_key"].isin(unknown_dev)], args.per_profile_cap, args.seed)
    unknown_test_df = cap_profiles(df[df["profile_key"].isin(unknown_test)], args.per_profile_cap, args.seed)
    candidate_profiles = sorted(known)
    candidate_index = {profile: index for index, profile in enumerate(candidate_profiles)}
    if any(frame.empty for frame in (train, known_dev, known_test, unknown_dev_df, unknown_test_df)):
        raise ValueError("Open-set train/dev/test partition is empty")

    combined = pd.concat([train, known_dev, known_test, unknown_dev_df, unknown_test_df], ignore_index=True)
    embeddings = encode(args.model_name, combined["text"].fillna("").tolist(), args.batch_size, args.device)
    offsets = np.cumsum([0, len(train), len(known_dev), len(known_test), len(unknown_dev_df), len(unknown_test_df)])
    train_emb, known_dev_emb, known_test_emb, unknown_dev_emb, unknown_test_emb = [
        embeddings[offsets[i]:offsets[i + 1]] for i in range(5)
    ]
    train_keys = train["profile_key"].tolist()
    centroids = normalize(np.vstack([
        train_emb[np.asarray([key == profile for key in train_keys])].mean(axis=0)
        for profile in candidate_profiles
    ]))

    known_dev_scores = known_dev_emb @ centroids.T
    known_test_scores = known_test_emb @ centroids.T
    unknown_dev_scores = unknown_dev_emb @ centroids.T
    unknown_test_scores = unknown_test_emb @ centroids.T
    dev_confidence = np.concatenate([known_dev_scores.max(axis=1), unknown_dev_scores.max(axis=1)])
    dev_labels = np.concatenate([np.ones(len(known_dev_scores)), np.zeros(len(unknown_dev_scores))])
    calibrator = LogisticRegression().fit(dev_confidence.reshape(-1, 1), dev_labels)
    test_confidence = np.concatenate([known_test_scores.max(axis=1), unknown_test_scores.max(axis=1)])
    test_labels = np.concatenate([np.ones(len(known_test_scores)), np.zeros(len(unknown_test_scores))])
    probabilities = calibrator.predict_proba(test_confidence.reshape(-1, 1))[:, 1]
    y_known = np.asarray([candidate_index[key] for key in known_test["profile_key"]])
    predicted_known = known_test_scores.argmax(axis=1)
    predicted_all = np.concatenate([predicted_known, unknown_test_scores.argmax(axis=1)])
    correct_all = np.concatenate([predicted_known == y_known, np.zeros(len(unknown_test_scores), dtype=bool)])
    top_half = np.argsort(test_confidence)[::-1][: max(1, len(test_confidence) // 2)]

    report = {
        "model_name": args.model_name,
        "language": args.language,
        "n_candidate_profiles": len(candidate_profiles),
        "n_unknown_dev_profiles": len(unknown_dev),
        "n_unknown_test_profiles": len(unknown_test),
        "known_ranking": ranking_metrics(known_test_scores, y_known),
        "open_set": open_set_metrics(test_labels, test_confidence, probabilities),
        "selective_top1_precision_at_50pct_coverage": float(correct_all[top_half].mean()),
        "maui_at_3": misattribution_unfairness(known_test_scores, y_known, 3),
        "calibrator": {
            "coefficient": float(calibrator.coef_[0, 0]),
            "intercept": float(calibrator.intercept_[0]),
        },
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "open_set_metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    pd.DataFrame(confusion_matrix(y_known, predicted_known)).to_csv(
        args.out_dir / "known_confusion_matrix.csv", index=False
    )
    pd.DataFrame({
        "known": test_labels.astype(bool),
        "max_similarity": test_confidence,
        "known_probability": probabilities,
        "predicted_candidate": [candidate_profiles[index] for index in predicted_all],
        "correct": correct_all,
    }).to_parquet(args.out_dir / "open_set_predictions.parquet", index=False)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
