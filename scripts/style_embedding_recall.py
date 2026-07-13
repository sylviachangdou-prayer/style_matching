from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder, normalize

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.retrieval_metrics import misattribution_unfairness


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run style embedding centroid recall.")
    parser.add_argument("--input", type=Path, required=True, help="Split parquet with text column.")
    parser.add_argument("--out-dir", type=Path, required=True, help="Output artifact directory.")
    parser.add_argument("--model-name", default="StyleDistance/styledistance")
    parser.add_argument("--model-revision", help="Optional immutable Hugging Face commit hash.")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--train-cap", type=int, default=300)
    parser.add_argument("--eval-splits", default="dev,test")
    parser.add_argument(
        "--language",
        help="Optional single-language diagnostic; keeps its candidate universe separate from multilingual runs.",
    )
    parser.add_argument("--seed", type=int, default=20260701)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or mps")
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip encoding when this out-dir already holds scores and metrics; lets an interrupted eval batch resume without re-running the GPU.",
    )
    return parser.parse_args()


def validate(df: pd.DataFrame) -> None:
    required = {"chunk_id", "author_or_speaker", "split", "text"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if df["chunk_id"].duplicated().any():
        raise ValueError("chunk_id must be unique")


def profile_key(df: pd.DataFrame) -> pd.Series:
    if "language" not in df.columns:
        return df["author_or_speaker"].astype(str)
    return df["language"].astype(str) + "::" + df["author_or_speaker"].astype(str)


def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def balanced_train(df: pd.DataFrame, train_cap: int, seed: int) -> pd.DataFrame:
    train = df[df["split"] == "train"].copy()
    train["profile_key"] = profile_key(train)
    return (
        train.sample(frac=1, random_state=seed)
        .groupby("profile_key", group_keys=False)
        .head(train_cap)
        .reset_index(drop=True)
    )


def encode_texts(
    model_name: str,
    texts: list[str],
    batch_size: int,
    device: str,
    revision: str | None = None,
) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name, device=device, revision=revision)
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


def mask_cross_language_candidates(
    scores: np.ndarray, query_languages: pd.Series, profiles: np.ndarray
) -> np.ndarray:
    masked = scores.copy()
    profile_languages = np.asarray([str(profile).split("::", 1)[0] for profile in profiles])
    for position, language in enumerate(query_languages.astype(str)):
        masked[position, profile_languages != language] = -1e9
    return masked


def prediction_frame(
    eval_df: pd.DataFrame,
    label_encoder: LabelEncoder,
    scores: np.ndarray,
    profile_to_author: dict[str, str],
    strategy: str = "single_centroid",
) -> pd.DataFrame:
    top3 = np.argsort(scores, axis=1)[:, -3:][:, ::-1]
    pred = top3[:, 0]
    output_columns = ["chunk_id", "author_or_speaker", "split"]
    for column in ("language", "corpus", "source_id"):
        if column in eval_df.columns:
            output_columns.append(column)
    output = eval_df[output_columns].copy()
    output["strategy"] = strategy
    predicted_profiles = label_encoder.inverse_transform(pred)
    output["predicted_profile"] = predicted_profiles
    output["predicted_author"] = [profile_to_author[profile] for profile in predicted_profiles]
    output["top1_score"] = np.max(scores, axis=1)
    for idx in range(3):
        profiles = label_encoder.inverse_transform(top3[:, idx])
        output[f"rank{idx + 1}_profile"] = profiles
        output[f"rank{idx + 1}_author"] = [profile_to_author[profile] for profile in profiles]
        output[f"rank{idx + 1}_score"] = scores[np.arange(scores.shape[0]), top3[:, idx]]
    return output


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.skip_existing and (args.out_dir / "style_embedding_scores.npz").exists() and (
        args.out_dir / "style_embedding_metrics.json"
    ).exists():
        print(f"skip existing eval: {args.out_dir} already has scores and metrics", flush=True)
        return
    df = pd.read_parquet(args.input)
    validate(df)
    if args.language:
        if "language" not in df.columns:
            raise ValueError("--language requires a language column")
        df = df[df["language"].astype(str).eq(args.language)].copy()
        if df.empty:
            raise ValueError(f"No rows found for language: {args.language}")

    train = balanced_train(df, args.train_cap, args.seed)
    eval_splits = {split.strip() for split in args.eval_splits.split(",") if split.strip()}
    eval_df = df[df["split"].isin(eval_splits)].copy().reset_index(drop=True)
    if eval_df.empty:
        raise ValueError(f"No rows found for eval splits: {sorted(eval_splits)}")

    df["profile_key"] = profile_key(df)
    eval_df["profile_key"] = profile_key(eval_df)
    label_encoder = LabelEncoder()
    label_encoder.fit(df["profile_key"])
    y_train = label_encoder.transform(train["profile_key"])
    y_eval = label_encoder.transform(eval_df["profile_key"])

    device = resolve_device(args.device)
    train_emb = encode_texts(
        args.model_name, train["text"].fillna("").tolist(), args.batch_size, device, args.model_revision
    )
    eval_emb = encode_texts(
        args.model_name, eval_df["text"].fillna("").tolist(), args.batch_size, device, args.model_revision
    )

    centroids = []
    for label in range(len(label_encoder.classes_)):
        centroid = train_emb[y_train == label].mean(axis=0)
        centroids.append(centroid)
    centroid_matrix = normalize(np.vstack(centroids), norm="l2")
    scores = cosine_similarity(eval_emb, centroid_matrix)
    scores = mask_cross_language_candidates(scores, eval_df["language"], label_encoder.classes_)
    pred = np.argmax(scores, axis=1)

    train_source_keys = (
        train["corpus"].astype(str) + "::" + (
            train["independent_source_id"] if "independent_source_id" in train else train["source_id"]
        ).fillna("").astype(str)
        if {"corpus", "source_id"}.issubset(train.columns)
        else pd.Series([f"row::{index}" for index in range(len(train))])
    )
    prototype_centroids = []
    prototype_labels = []
    prototype_frame = train.assign(source_key=train_source_keys)
    for (profile, _), indices in prototype_frame.groupby(["profile_key", "source_key"]).indices.items():
        prototype_centroids.append(train_emb[indices].mean(axis=0))
        prototype_labels.append(int(label_encoder.transform([profile])[0]))
    prototype_centroids = normalize(np.vstack(prototype_centroids), norm="l2")
    raw_prototype_scores = cosine_similarity(eval_emb, prototype_centroids)
    prototype_scores = np.full_like(scores, -1.0)
    prototype_labels_array = np.asarray(prototype_labels)
    for label in range(len(label_encoder.classes_)):
        candidate = raw_prototype_scores[:, prototype_labels_array == label]
        top = np.sort(candidate, axis=1)[:, ::-1][:, :3]
        prototype_scores[:, label] = top.mean(axis=1)
    prototype_scores = mask_cross_language_candidates(
        prototype_scores, eval_df["language"], label_encoder.classes_
    )

    metrics = {
        "input": str(args.input),
        "model_name": args.model_name,
        "model_revision": args.model_revision,
        "n_train": int(len(train)),
        "n_eval": int(len(eval_df)),
        "n_authors": int(df["author_or_speaker"].nunique()),
        "n_author_language_profiles": int(df["profile_key"].nunique()),
        "eval_splits": sorted(eval_splits),
        "top1_accuracy": float(topk_accuracy(scores, y_eval, 1)),
        "top3_accuracy": float(topk_accuracy(scores, y_eval, 3)),
        "top5_accuracy": float(topk_accuracy(scores, y_eval, 5)),
        "top20_accuracy": float(topk_accuracy(scores, y_eval, 20)),
        "mrr": mrr_from_scores(scores, y_eval),
        "accuracy": float(accuracy_score(y_eval, pred)),
        "device": device,
        "maui_at_3": misattribution_unfairness(scores, y_eval, 3),
        "maui_candidate_profiles": label_encoder.classes_.tolist(),
        "strategies": {
            "single_centroid": {
                "top1_accuracy": float(topk_accuracy(scores, y_eval, 1)),
                "top3_accuracy": float(topk_accuracy(scores, y_eval, 3)),
                "top5_accuracy": float(topk_accuracy(scores, y_eval, 5)),
                "top20_accuracy": float(topk_accuracy(scores, y_eval, 20)),
                "mrr": mrr_from_scores(scores, y_eval),
            },
            "source_prototype_top3_mean": {
                "top1_accuracy": float(topk_accuracy(prototype_scores, y_eval, 1)),
                "top3_accuracy": float(topk_accuracy(prototype_scores, y_eval, 3)),
                "top5_accuracy": float(topk_accuracy(prototype_scores, y_eval, 5)),
                "top20_accuracy": float(topk_accuracy(prototype_scores, y_eval, 20)),
                "mrr": mrr_from_scores(prototype_scores, y_eval),
            },
        },
    }

    def breakdown(column: str) -> dict[str, dict[str, float]]:
        if column not in eval_df.columns:
            return {}
        result = {}
        for value, subset in eval_df.groupby(column, sort=True):
            positions = subset.index.to_numpy()
            subset_scores = scores[positions]
            subset_labels = y_eval[positions]
            result[str(value)] = {
                "n_eval": int(len(subset)),
                "n_candidates": int(sum(
                    str(profile).startswith(f"{value}::") for profile in label_encoder.classes_
                )) if column == "language" else int(len(label_encoder.classes_)),
                "top1_accuracy": topk_accuracy(subset_scores, subset_labels, 1),
                "top3_accuracy": topk_accuracy(subset_scores, subset_labels, 3),
                "top5_accuracy": topk_accuracy(subset_scores, subset_labels, 5),
                "top20_accuracy": topk_accuracy(subset_scores, subset_labels, 20),
                "mrr": mrr_from_scores(subset_scores, subset_labels),
            }
        return result

    metrics["by_language"] = breakdown("language")
    metrics["by_corpus"] = breakdown("corpus")

    profile_to_author = dict(zip(df["profile_key"], df["author_or_speaker"]))
    predictions = pd.concat([
        prediction_frame(eval_df, label_encoder, scores, profile_to_author, "single_centroid"),
        prediction_frame(
            eval_df,
            label_encoder,
            prototype_scores,
            profile_to_author,
            "source_prototype_top3_mean",
        ),
    ], ignore_index=True)
    pred_path = args.out_dir / "style_embedding_predictions.parquet"
    metrics_path = args.out_dir / "style_embedding_metrics.json"
    centroid_path = args.out_dir / "style_embedding_author_centroids.npz"
    train_emb_path = args.out_dir / "style_embedding_train_embeddings.npy"
    eval_emb_path = args.out_dir / "style_embedding_eval_embeddings.npy"
    score_path = args.out_dir / "style_embedding_scores.npz"

    predictions.to_parquet(pred_path, index=False)
    pd.DataFrame(
        confusion_matrix(y_eval, pred, labels=np.arange(len(label_encoder.classes_))),
        index=label_encoder.classes_,
        columns=label_encoder.classes_,
    ).to_csv(args.out_dir / "style_embedding_confusion_matrix.csv")
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    np.savez_compressed(
        centroid_path,
        authors=np.asarray([profile_to_author[profile] for profile in label_encoder.classes_]),
        profiles=label_encoder.classes_,
        centroids=centroid_matrix.astype("float32"),
    )
    np.save(train_emb_path, train_emb)
    np.save(eval_emb_path, eval_emb)
    np.savez_compressed(
        score_path,
        chunk_ids=eval_df["chunk_id"].astype(str).to_numpy(),
        splits=eval_df["split"].astype(str).to_numpy(),
        query_languages=eval_df["language"].astype(str).to_numpy(),
        query_corpora=eval_df["corpus"].astype(str).to_numpy(),
        profiles=label_encoder.classes_,
        y_true=y_eval,
        single_centroid_scores=scores.astype("float32"),
        source_prototype_scores=prototype_scores.astype("float32"),
    )

    print(json.dumps(metrics, indent=2))
    print(f"Wrote {metrics_path}")
    print(f"Wrote {pred_path}")
    print(f"Wrote {centroid_path}")


if __name__ == "__main__":
    main()
