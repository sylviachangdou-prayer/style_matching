from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder, normalize


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run style embedding centroid recall.")
    parser.add_argument("--input", type=Path, required=True, help="Split parquet with text column.")
    parser.add_argument("--out-dir", type=Path, required=True, help="Output artifact directory.")
    parser.add_argument("--model-name", default="StyleDistance/styledistance")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--train-cap", type=int, default=300)
    parser.add_argument("--eval-splits", default="dev,test")
    parser.add_argument("--seed", type=int, default=20260701)
    return parser.parse_args()


def validate(df: pd.DataFrame) -> None:
    required = {"chunk_id", "author_or_speaker", "split", "text"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if df["chunk_id"].duplicated().any():
        raise ValueError("chunk_id must be unique")


def balanced_train(df: pd.DataFrame, train_cap: int, seed: int) -> pd.DataFrame:
    train = df[df["split"] == "train"].copy()
    return (
        train.sample(frac=1, random_state=seed)
        .groupby("author_or_speaker", group_keys=False)
        .head(train_cap)
        .reset_index(drop=True)
    )


def encode_texts(model_name: str, texts: list[str], batch_size: int) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return embeddings.astype("float32")


def mrr_from_scores(scores: np.ndarray, y_true: np.ndarray) -> float:
    ranks = []
    for row, true_label in zip(scores, y_true):
        order = np.argsort(row)[::-1]
        rank = int(np.where(order == true_label)[0][0]) + 1
        ranks.append(1.0 / rank)
    return float(np.mean(ranks))


def topk_accuracy(scores: np.ndarray, y_true: np.ndarray, k: int) -> float:
    topk = np.argsort(scores, axis=1)[:, -k:]
    return float(np.mean([true in row for true, row in zip(y_true, topk)]))


def prediction_frame(eval_df: pd.DataFrame, label_encoder: LabelEncoder, scores: np.ndarray) -> pd.DataFrame:
    top3 = np.argsort(scores, axis=1)[:, -3:][:, ::-1]
    pred = top3[:, 0]
    output = eval_df[["chunk_id", "author_or_speaker", "split"]].copy()
    output["predicted_author"] = label_encoder.inverse_transform(pred)
    output["top1_score"] = np.max(scores, axis=1)
    for idx in range(3):
        output[f"rank{idx + 1}_author"] = label_encoder.inverse_transform(top3[:, idx])
        output[f"rank{idx + 1}_score"] = scores[np.arange(scores.shape[0]), top3[:, idx]]
    return output


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(args.input)
    validate(df)

    train = balanced_train(df, args.train_cap, args.seed)
    eval_splits = {split.strip() for split in args.eval_splits.split(",") if split.strip()}
    eval_df = df[df["split"].isin(eval_splits)].copy().reset_index(drop=True)
    if eval_df.empty:
        raise ValueError(f"No rows found for eval splits: {sorted(eval_splits)}")

    label_encoder = LabelEncoder()
    label_encoder.fit(df["author_or_speaker"])
    y_train = label_encoder.transform(train["author_or_speaker"])
    y_eval = label_encoder.transform(eval_df["author_or_speaker"])

    train_emb = encode_texts(args.model_name, train["text"].fillna("").tolist(), args.batch_size)
    eval_emb = encode_texts(args.model_name, eval_df["text"].fillna("").tolist(), args.batch_size)

    centroids = []
    for label in range(len(label_encoder.classes_)):
        centroid = train_emb[y_train == label].mean(axis=0)
        centroids.append(centroid)
    centroid_matrix = normalize(np.vstack(centroids), norm="l2")
    scores = cosine_similarity(eval_emb, centroid_matrix)
    pred = np.argmax(scores, axis=1)

    metrics = {
        "input": str(args.input),
        "model_name": args.model_name,
        "n_train": int(len(train)),
        "n_eval": int(len(eval_df)),
        "n_authors": int(len(label_encoder.classes_)),
        "eval_splits": sorted(eval_splits),
        "top1_accuracy": float(topk_accuracy(scores, y_eval, 1)),
        "top3_accuracy": float(topk_accuracy(scores, y_eval, 3)),
        "top5_accuracy": float(topk_accuracy(scores, y_eval, 5)),
        "mrr": mrr_from_scores(scores, y_eval),
        "accuracy": float(accuracy_score(y_eval, pred)),
    }

    predictions = prediction_frame(eval_df, label_encoder, scores)
    pred_path = args.out_dir / "style_embedding_predictions.parquet"
    metrics_path = args.out_dir / "style_embedding_metrics.json"
    centroid_path = args.out_dir / "style_embedding_author_centroids.npz"
    train_emb_path = args.out_dir / "style_embedding_train_embeddings.npy"
    eval_emb_path = args.out_dir / "style_embedding_eval_embeddings.npy"

    predictions.to_parquet(pred_path, index=False)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    np.savez_compressed(
        centroid_path,
        authors=label_encoder.classes_,
        centroids=centroid_matrix.astype("float32"),
    )
    np.save(train_emb_path, train_emb)
    np.save(eval_emb_path, eval_emb)

    print(json.dumps(metrics, indent=2))
    print(f"Wrote {metrics_path}")
    print(f"Wrote {pred_path}")
    print(f"Wrote {centroid_path}")


if __name__ == "__main__":
    main()
