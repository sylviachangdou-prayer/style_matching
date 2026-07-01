from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import LinearSVC


FUNCTION_WORDS = [
    "the", "and", "of", "to", "in", "a", "that", "is", "it", "for",
    "as", "with", "was", "on", "be", "by", "not", "he", "this", "are",
    "or", "his", "from", "at", "which", "but", "have", "an", "had", "they",
    "you", "were", "their", "one", "all", "we", "can", "her", "has", "there",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run literary corpus sanity baselines.")
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Balanced split parquet with text and split columns.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Directory for metrics and prediction artifacts.",
    )
    return parser.parse_args()


def validate_input(df: pd.DataFrame) -> None:
    required = {"chunk_id", "author_or_speaker", "split", "text"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if df["chunk_id"].duplicated().any():
        raise ValueError("chunk_id must be unique")
    splits = set(df["split"])
    if not {"train", "dev", "test"}.issubset(splits):
        raise ValueError(f"Expected train/dev/test splits, got {sorted(splits)}")


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


def evaluate_scores(scores: np.ndarray, y_true: np.ndarray) -> dict[str, float]:
    return {
        "top1_accuracy": float(topk_accuracy(scores, y_true, 1)),
        "top3_accuracy": float(topk_accuracy(scores, y_true, 3)),
        "top5_accuracy": float(topk_accuracy(scores, y_true, 5)),
        "mrr": mrr_from_scores(scores, y_true),
    }


def train_linear_svc(train: pd.DataFrame, eval_df: pd.DataFrame, label_encoder: LabelEncoder) -> tuple[dict[str, float], pd.DataFrame]:
    vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(3, 5),
        min_df=3,
        max_features=250_000,
        sublinear_tf=True,
    )
    model = make_pipeline(vectorizer, LinearSVC(C=1.0))
    y_train = label_encoder.transform(train["author_or_speaker"])
    y_eval = label_encoder.transform(eval_df["author_or_speaker"])
    model.fit(train["text"], y_train)
    scores = model.decision_function(eval_df["text"])
    if scores.ndim == 1:
        scores = np.column_stack([-scores, scores])
    pred = np.argmax(scores, axis=1)
    metrics = evaluate_scores(scores, y_eval)
    metrics["accuracy"] = float(accuracy_score(y_eval, pred))
    predictions = prediction_frame(eval_df, label_encoder, scores, pred, "char_tfidf_linear_svc")
    return metrics, predictions


def train_centroid(train: pd.DataFrame, eval_df: pd.DataFrame, label_encoder: LabelEncoder) -> tuple[dict[str, float], pd.DataFrame]:
    vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(3, 5),
        min_df=3,
        max_features=250_000,
        sublinear_tf=True,
    )
    x_train = vectorizer.fit_transform(train["text"])
    x_eval = vectorizer.transform(eval_df["text"])
    y_train = label_encoder.transform(train["author_or_speaker"])
    y_eval = label_encoder.transform(eval_df["author_or_speaker"])

    centroids = []
    for label in range(len(label_encoder.classes_)):
        centroid = x_train[y_train == label].mean(axis=0)
        centroids.append(np.asarray(centroid).ravel())
    centroid_matrix = np.vstack(centroids)
    scores = cosine_similarity(x_eval, centroid_matrix)
    pred = np.argmax(scores, axis=1)
    metrics = evaluate_scores(scores, y_eval)
    metrics["accuracy"] = float(accuracy_score(y_eval, pred))
    predictions = prediction_frame(eval_df, label_encoder, scores, pred, "char_tfidf_centroid")
    return metrics, predictions


def stylometric_features(texts: pd.Series) -> pd.DataFrame:
    rows = []
    for text in texts.fillna(""):
        words = re.findall(r"[A-Za-z']+", text.lower())
        sentences = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]
        word_count = max(len(words), 1)
        unique_count = len(set(words))
        sentence_lengths = [
            len(re.findall(r"[A-Za-z']+", sentence))
            for sentence in sentences
            if re.findall(r"[A-Za-z']+", sentence)
        ]
        avg_sentence = float(np.mean(sentence_lengths)) if sentence_lengths else 0.0
        sd_sentence = float(np.std(sentence_lengths)) if sentence_lengths else 0.0
        row = {
            "word_count": len(words),
            "type_token_ratio": unique_count / word_count,
            "avg_word_length": float(np.mean([len(word) for word in words])) if words else 0.0,
            "avg_sentence_length": avg_sentence,
            "sd_sentence_length": sd_sentence,
            "comma_rate": text.count(",") / word_count,
            "semicolon_rate": text.count(";") / word_count,
            "colon_rate": text.count(":") / word_count,
            "quote_rate": (text.count('"') + text.count("'")) / word_count,
            "question_rate": text.count("?") / word_count,
            "exclamation_rate": text.count("!") / word_count,
            "dash_rate": (text.count("-") + text.count("—")) / word_count,
        }
        counts = pd.Series(words).value_counts() if words else pd.Series(dtype=int)
        for word in FUNCTION_WORDS:
            row[f"fw_{word}"] = int(counts.get(word, 0)) / word_count
        rows.append(row)
    return pd.DataFrame(rows)


def train_stylometric(train: pd.DataFrame, eval_df: pd.DataFrame, label_encoder: LabelEncoder) -> tuple[dict[str, float], pd.DataFrame]:
    x_train = stylometric_features(train["text"])
    x_eval = stylometric_features(eval_df["text"])
    y_train = label_encoder.transform(train["author_or_speaker"])
    y_eval = label_encoder.transform(eval_df["author_or_speaker"])
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, class_weight="balanced", n_jobs=-1),
    )
    model.fit(x_train, y_train)
    scores = model.predict_proba(x_eval)
    pred = np.argmax(scores, axis=1)
    metrics = evaluate_scores(scores, y_eval)
    metrics["accuracy"] = float(accuracy_score(y_eval, pred))
    predictions = prediction_frame(eval_df, label_encoder, scores, pred, "stylometric_logreg")
    return metrics, predictions


def prediction_frame(
    eval_df: pd.DataFrame,
    label_encoder: LabelEncoder,
    scores: np.ndarray,
    pred: np.ndarray,
    model_name: str,
) -> pd.DataFrame:
    top3 = np.argsort(scores, axis=1)[:, -3:][:, ::-1]
    output = eval_df[["chunk_id", "author_or_speaker", "split"]].copy()
    output["model"] = model_name
    output["predicted_author"] = label_encoder.inverse_transform(pred)
    output["top1_score"] = np.max(scores, axis=1)
    for idx in range(3):
        output[f"rank{idx + 1}_author"] = label_encoder.inverse_transform(top3[:, idx])
        output[f"rank{idx + 1}_score"] = scores[np.arange(scores.shape[0]), top3[:, idx]]
    return output


def classification_summary(y_true: np.ndarray, y_pred: np.ndarray, labels: list[str]) -> dict:
    report = classification_report(
        y_true,
        y_pred,
        target_names=labels,
        output_dict=True,
        zero_division=0,
    )
    return {
        "macro_f1": report["macro avg"]["f1-score"],
        "weighted_f1": report["weighted avg"]["f1-score"],
    }


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(args.input)
    validate_input(df)
    train = df[df["split"] == "train"].copy()
    eval_df = df[df["split"].isin(["dev", "test"])].copy()

    label_encoder = LabelEncoder()
    label_encoder.fit(df["author_or_speaker"])

    metrics = {
        "input": str(args.input),
        "n_rows": int(len(df)),
        "n_train": int(len(train)),
        "n_eval": int(len(eval_df)),
        "n_authors": int(df["author_or_speaker"].nunique()),
        "split_counts": df["split"].value_counts().to_dict(),
        "models": {},
    }
    predictions = []

    for name, runner in [
        ("char_tfidf_centroid", train_centroid),
        ("char_tfidf_linear_svc", train_linear_svc),
        ("stylometric_logreg", train_stylometric),
    ]:
        model_metrics, model_predictions = runner(train, eval_df, label_encoder)
        y_true = label_encoder.transform(model_predictions["author_or_speaker"])
        y_pred = label_encoder.transform(model_predictions["predicted_author"])
        model_metrics.update(classification_summary(y_true, y_pred, list(label_encoder.classes_)))
        metrics["models"][name] = model_metrics
        predictions.append(model_predictions)

    pred_df = pd.concat(predictions, ignore_index=True)
    pred_path = args.out_dir / "literary_baseline_predictions.parquet"
    metrics_path = args.out_dir / "literary_baseline_metrics.json"
    pred_df.to_parquet(pred_path, index=False)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(json.dumps(metrics, indent=2))
    print(f"Wrote {metrics_path}")
    print(f"Wrote {pred_path}")


if __name__ == "__main__":
    main()
