from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.retrieval_metrics import ranking_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate direct original-text retrieval by ordered language pair.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--train-cap", type=int, default=300)
    parser.add_argument("--eval-cap", type=int, default=100)
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


def cap(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    return (
        df.sample(frac=1, random_state=seed)
        .groupby(["language", "author_or_speaker"], group_keys=False)
        .head(n)
        .reset_index(drop=True)
    )


def main() -> None:
    args = parse_args()
    df = pd.read_parquet(args.input)
    required = {"language", "author_or_speaker", "split", "text"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    train = cap(df[df["split"].eq("train")], args.train_cap, args.seed)
    evaluation = cap(df[df["split"].isin(["dev", "test"])], args.eval_cap, args.seed)
    combined = pd.concat([train, evaluation], ignore_index=True)
    embeddings = encode(args.model_name, combined["text"].fillna("").tolist(), args.batch_size, args.device)
    train_embeddings = embeddings[:len(train)]
    eval_embeddings = embeddings[len(train):]

    target_profiles = []
    target_centroids = []
    for (language, author), indices in train.groupby(["language", "author_or_speaker"], sort=True).indices.items():
        target_profiles.append((str(language), str(author)))
        target_centroids.append(train_embeddings[indices].mean(axis=0))
    target_centroids = normalize(np.vstack(target_centroids))
    profile_index = {profile: index for index, profile in enumerate(target_profiles)}
    languages = sorted(df["language"].astype(str).unique())
    reports = {}
    prediction_rows = []
    for source_language in languages:
        source_rows = evaluation[evaluation["language"].eq(source_language)]
        if source_rows.empty:
            continue
        source_positions = source_rows.index.to_numpy()
        for target_language in languages:
            if source_language == target_language:
                continue
            candidates = [index for index, profile in enumerate(target_profiles) if profile[0] == target_language]
            eligible_positions = [
                position for position in source_positions
                if (target_language, str(evaluation.iloc[position]["author_or_speaker"])) in profile_index
            ]
            shared_authors = sorted({str(evaluation.iloc[position]["author_or_speaker"]) for position in eligible_positions})
            key = f"{source_language}->{target_language}"
            if len(shared_authors) < 2 or not eligible_positions:
                reports[key] = {
                    "status": "insufficient_shared_authors",
                    "n_shared_authors": len(shared_authors),
                    "n_queries": len(eligible_positions),
                }
                continue
            pair_scores = eval_embeddings[eligible_positions] @ target_centroids[candidates].T
            candidate_lookup = {target_profiles[index][1]: local for local, index in enumerate(candidates)}
            labels = np.asarray([
                candidate_lookup[str(evaluation.iloc[position]["author_or_speaker"])]
                for position in eligible_positions
            ])
            reports[key] = {
                "status": "evaluated",
                "n_shared_authors": len(shared_authors),
                "n_target_candidates": len(candidates),
                "n_queries": len(eligible_positions),
                **ranking_metrics(pair_scores, labels),
            }
            for row_position, score_row in zip(eligible_positions, pair_scores):
                best = int(np.argmax(score_row))
                prediction_rows.append({
                    "chunk_id": evaluation.iloc[row_position].get("chunk_id", ""),
                    "source_language": source_language,
                    "target_language": target_language,
                    "true_author": evaluation.iloc[row_position]["author_or_speaker"],
                    "predicted_author": target_profiles[candidates[best]][1],
                    "top1_similarity": float(score_row[best]),
                })

    output = {"model_name": args.model_name, "ordered_language_pairs": reports}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "cross_language_metrics.json").write_text(
        json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    pd.DataFrame(prediction_rows).to_parquet(args.out_dir / "cross_language_predictions.parquet", index=False)
    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
