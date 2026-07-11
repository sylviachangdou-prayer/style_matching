from __future__ import annotations

import argparse
import json
import re
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import LinearSVC


FUNCTION_WORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are",
    "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but",
    "by", "can", "could", "did", "do", "does", "doing", "down", "during", "each", "few", "for",
    "from", "further", "had", "has", "have", "having", "he", "her", "here", "hers", "herself",
    "him", "himself", "his", "how", "i", "if", "in", "into", "is", "it", "its", "itself", "me",
    "more", "most", "my", "myself", "nor", "not", "of", "off", "on", "once", "only", "or",
    "other", "our", "ours", "ourselves", "out", "over", "own", "same", "she", "should", "so",
    "some", "such", "than", "that", "the", "their", "theirs", "them", "themselves", "then",
    "there", "these", "they", "this", "those", "through", "to", "too", "under", "until", "up",
    "very", "was", "we", "were", "what", "when", "where", "which", "while", "who", "whom",
    "why", "will", "with", "would", "you", "your", "yours", "yourself", "yourselves",
}
HEDGES = {"apparently", "perhaps", "possibly", "probably", "seem", "seems", "suggest", "suggests"}
MODALS = {"can", "could", "may", "might", "must", "shall", "should", "will", "would"}
FIRST_PERSON = {"i", "me", "my", "mine", "we", "us", "our", "ours"}
SECOND_PERSON = {"you", "your", "yours"}
THIRD_PERSON = {"he", "him", "his", "she", "her", "hers", "they", "them", "their", "theirs"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run content-masked style baselines.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--eval-splits", default="dev,test")
    parser.add_argument("--max-features", type=int, default=250_000)
    return parser.parse_args()


def validate(df: pd.DataFrame) -> None:
    required = {"chunk_id", "author_or_speaker", "language", "split", "text"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if df["chunk_id"].duplicated().any():
        raise ValueError("chunk_id must be unique")


def word_shape(word: str) -> str:
    if word.lower() in FUNCTION_WORDS:
        return word.lower()
    if word.isdigit():
        return "<NUM>"
    length = len(word)
    if length <= 3:
        size = "S"
    elif length <= 7:
        size = "M"
    else:
        size = "L"
    if word.isupper():
        case = "UP"
    elif word[:1].isupper():
        case = "CAP"
    else:
        case = "LOW"
    suffix = word[-3:].lower() if length >= 5 else ""
    if suffix in {"ing", "ion", "ity", "ous", "ive", "ent", "ant", "est", "ful", "less", "ment"}:
        return f"<W_{size}_{case}_{suffix}>"
    return f"<W_{size}_{case}>"


def delexicalize(text: str) -> str:
    tokens = re.findall(r"[A-Za-z]+|\d+|[^\w\s]", text)
    return " ".join(word_shape(token) if re.match(r"[A-Za-z0-9]+$", token) else token for token in tokens)


def function_word_stream(text: str) -> str:
    tokens = re.findall(r"[A-Za-z]+|[^\w\s]", text.lower())
    return " ".join(token if token in FUNCTION_WORDS or re.match(r"[^\w\s]$", token) else "<X>" for token in tokens)


def stylometric_features(texts: pd.Series) -> pd.DataFrame:
    rows = []
    for text in texts.fillna(""):
        words = re.findall(r"[A-Za-z']+", text.lower())
        sentences = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]
        word_count = max(len(words), 1)
        sentence_lengths = [
            len(re.findall(r"[A-Za-z']+", sentence))
            for sentence in sentences
            if re.findall(r"[A-Za-z']+", sentence)
        ]
        counts = pd.Series(words).value_counts() if words else pd.Series(dtype=int)
        sentence_openings = [
            match.group(0).lower()
            for sentence in sentences
            if (match := re.search(r"[A-Za-z']+", sentence))
        ]
        repeated_openings = len(sentence_openings) - len(set(sentence_openings))
        row = {
            "word_count": len(words),
            "type_token_ratio": len(set(words)) / word_count,
            "avg_word_length": float(np.mean([len(word) for word in words])) if words else 0.0,
            "avg_sentence_length": float(np.mean(sentence_lengths)) if sentence_lengths else 0.0,
            "sd_sentence_length": float(np.std(sentence_lengths)) if sentence_lengths else 0.0,
            "comma_rate": text.count(",") / word_count,
            "semicolon_rate": text.count(";") / word_count,
            "colon_rate": text.count(":") / word_count,
            "quote_rate": (text.count('"') + text.count("'")) / word_count,
            "question_rate": text.count("?") / word_count,
            "exclamation_rate": text.count("!") / word_count,
            "dash_rate": (text.count("-") + text.count("—")) / word_count,
            "paren_rate": (text.count("(") + text.count(")")) / word_count,
            "hedge_rate": sum(int(counts.get(word, 0)) for word in HEDGES) / word_count,
            "modal_rate": sum(int(counts.get(word, 0)) for word in MODALS) / word_count,
            "first_person_rate": sum(int(counts.get(word, 0)) for word in FIRST_PERSON) / word_count,
            "second_person_rate": sum(int(counts.get(word, 0)) for word in SECOND_PERSON) / word_count,
            "third_person_rate": sum(int(counts.get(word, 0)) for word in THIRD_PERSON) / word_count,
            "parallel_opening_rate": repeated_openings / max(len(sentence_openings), 1),
            "compression_ratio": len(zlib.compress(text.encode("utf-8"))) / max(len(text.encode("utf-8")), 1),
        }
        for word in sorted(FUNCTION_WORDS):
            row[f"fw_{word}"] = int(counts.get(word, 0)) / word_count
        rows.append(row)
    return pd.DataFrame(rows)


def softmax(scores: np.ndarray) -> np.ndarray:
    shifted = scores - scores.max(axis=1, keepdims=True)
    exp_scores = np.exp(shifted)
    return exp_scores / exp_scores.sum(axis=1, keepdims=True)


def evaluate_scores(scores: np.ndarray, y_true: np.ndarray) -> dict[str, float]:
    order = np.argsort(scores, axis=1)[:, ::-1]
    ranks = [int(np.where(row == true)[0][0]) + 1 for row, true in zip(order, y_true)]
    return {
        "top1_accuracy": float(np.mean([rank <= 1 for rank in ranks])),
        "top3_accuracy": float(np.mean([rank <= 3 for rank in ranks])),
        "top5_accuracy": float(np.mean([rank <= 5 for rank in ranks])),
        "top20_accuracy": float(np.mean([rank <= 20 for rank in ranks])),
        "mrr": float(np.mean([1.0 / rank for rank in ranks])),
    }


def prediction_frame(eval_df: pd.DataFrame, label_encoder: LabelEncoder, scores: np.ndarray, model_name: str) -> pd.DataFrame:
    pred = np.argmax(scores, axis=1)
    top3 = np.argsort(scores, axis=1)[:, -3:][:, ::-1]
    output = eval_df[["chunk_id", "author_or_speaker", "split"]].copy()
    output["model"] = model_name
    output["predicted_profile"] = label_encoder.inverse_transform(pred)
    output["predicted_author"] = output["predicted_profile"].str.split("::", n=1).str[-1]
    output["top1_score"] = np.max(scores, axis=1)
    for idx in range(3):
        output[f"rank{idx + 1}_profile"] = label_encoder.inverse_transform(top3[:, idx])
        output[f"rank{idx + 1}_author"] = output[f"rank{idx + 1}_profile"].str.split("::", n=1).str[-1]
        output[f"rank{idx + 1}_score"] = scores[np.arange(scores.shape[0]), top3[:, idx]]
    return output


def classification_summary(y_true: np.ndarray, scores: np.ndarray, labels: list[str]) -> dict[str, float]:
    pred = np.argmax(scores, axis=1)
    report = classification_report(
        y_true,
        pred,
        labels=np.arange(len(labels)),
        target_names=labels,
        output_dict=True,
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(y_true, pred)),
        "macro_f1": float(report["macro avg"]["f1-score"]),
        "weighted_f1": float(report["weighted avg"]["f1-score"]),
    }


def mask_cross_language_candidates(
    scores: np.ndarray, query_languages: pd.Series, profiles: np.ndarray
) -> np.ndarray:
    masked = scores.copy()
    profile_languages = np.asarray([str(profile).split("::", 1)[0] for profile in profiles])
    for position, language in enumerate(query_languages.astype(str)):
        masked[position, profile_languages != language] = -1e9
    return masked


def train_text_svc(train_text: pd.Series, eval_text: pd.Series, y_train: np.ndarray, max_features: int) -> np.ndarray:
    model = make_pipeline(
        TfidfVectorizer(analyzer="char", ngram_range=(3, 5), min_df=3, max_features=max_features, sublinear_tf=True),
        LinearSVC(C=1.0),
    )
    model.fit(train_text, y_train)
    scores = model.decision_function(eval_text)
    if scores.ndim == 1:
        scores = np.column_stack([-scores, scores])
    return scores


def train_stylometric(train: pd.DataFrame, eval_df: pd.DataFrame, y_train: np.ndarray) -> np.ndarray:
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, class_weight="balanced", n_jobs=-1),
    )
    model.fit(stylometric_features(train["text"]), y_train)
    return model.predict_proba(stylometric_features(eval_df["text"]))


def compression_distance_scores(
    train: pd.DataFrame,
    eval_df: pd.DataFrame,
    y_train: np.ndarray,
    n_profiles: int,
    reference_char_cap: int = 12_000,
) -> np.ndarray:
    references = []
    for label in range(n_profiles):
        reference = "\n".join(train.iloc[y_train == label]["text"].fillna("").astype(str))
        references.append(reference[:reference_char_cap].encode("utf-8"))
    reference_sizes = np.asarray([len(zlib.compress(value)) for value in references], dtype=float)
    scores = np.empty((len(eval_df), n_profiles), dtype=float)
    for row_index, text in enumerate(eval_df["text"].fillna("").astype(str)):
        query = text.encode("utf-8")
        query_size = float(len(zlib.compress(query)))
        for label, reference in enumerate(references):
            combined_size = float(len(zlib.compress(reference + b"\n" + query)))
            denominator = max(reference_sizes[label], query_size, 1.0)
            distance = (combined_size - min(reference_sizes[label], query_size)) / denominator
            scores[row_index, label] = 1.0 - distance
    return scores


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(args.input)
    validate(df)

    eval_splits = {split.strip() for split in args.eval_splits.split(",") if split.strip()}
    train = df[df["split"].eq("train")].copy()
    eval_df = df[df["split"].isin(eval_splits)].copy()
    if train.empty or eval_df.empty:
        raise ValueError("train/eval split is empty")

    label_encoder = LabelEncoder()
    df["profile_key"] = df["language"].astype(str) + "::" + df["author_or_speaker"].astype(str)
    train["profile_key"] = train["language"].astype(str) + "::" + train["author_or_speaker"].astype(str)
    eval_df["profile_key"] = eval_df["language"].astype(str) + "::" + eval_df["author_or_speaker"].astype(str)
    label_encoder.fit(df["profile_key"])
    y_train = label_encoder.transform(train["profile_key"])
    y_eval = label_encoder.transform(eval_df["profile_key"])

    train_delex = train["text"].map(delexicalize)
    eval_delex = eval_df["text"].map(delexicalize)
    train_fw = train["text"].map(function_word_stream)
    eval_fw = eval_df["text"].map(function_word_stream)

    model_scores = {
        "raw_char_svc_topic_leakage_ceiling": train_text_svc(train["text"], eval_df["text"], y_train, args.max_features),
        "delex_char_svc": train_text_svc(train_delex, eval_delex, y_train, args.max_features),
        "function_word_char_svc": train_text_svc(train_fw, eval_fw, y_train, args.max_features),
        "stylometric_logreg": train_stylometric(train, eval_df, y_train),
        "compression_distance": compression_distance_scores(
            train, eval_df, y_train, len(label_encoder.classes_)
        ),
    }
    model_scores = {
        name: mask_cross_language_candidates(scores, eval_df["language"], label_encoder.classes_)
        for name, scores in model_scores.items()
    }
    model_scores["style_only_fusion"] = (
        softmax(model_scores["delex_char_svc"])
        + softmax(model_scores["function_word_char_svc"])
        + softmax(model_scores["stylometric_logreg"])
        + softmax(model_scores["compression_distance"])
    ) / 4.0

    metrics = {
        "input": str(args.input),
        "n_rows": int(len(df)),
        "n_train": int(len(train)),
        "n_eval": int(len(eval_df)),
        "n_authors": int(df["author_or_speaker"].nunique()),
        "n_author_language_profiles": int(df["profile_key"].nunique()),
        "eval_splits": sorted(eval_splits),
        "models": {},
        "interpretation": {
            "raw_char_svc_topic_leakage_ceiling": "High score here may reflect lexical/topic overlap, not only style.",
            "style_only_fusion": "Primary v1 robustness score: delexicalized character patterns + function-word stream + stylometric rhythm/discourse + compression distance.",
        },
    }
    predictions = []
    labels = list(label_encoder.classes_)
    for name, scores in model_scores.items():
        model_metrics = evaluate_scores(scores, y_eval)
        model_metrics.update(classification_summary(y_eval, scores, labels))
        metrics["models"][name] = model_metrics
        predictions.append(prediction_frame(eval_df, label_encoder, scores, name))

    pred_df = pd.concat(predictions, ignore_index=True)
    pred_path = args.out_dir / "style_robust_predictions.parquet"
    metrics_path = args.out_dir / "style_robust_metrics.json"
    pred_df.to_parquet(pred_path, index=False)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    np.savez_compressed(
        args.out_dir / "style_robust_scores.npz",
        chunk_ids=eval_df["chunk_id"].astype(str).to_numpy(),
        splits=eval_df["split"].astype(str).to_numpy(),
        query_languages=eval_df["language"].astype(str).to_numpy(),
        query_corpora=eval_df["corpus"].astype(str).to_numpy(),
        profiles=label_encoder.classes_,
        y_true=y_eval,
        **{name: scores.astype("float32") for name, scores in model_scores.items()},
    )
    print(json.dumps(metrics, indent=2))
    print(f"Wrote {metrics_path}")
    print(f"Wrote {pred_path}")


if __name__ == "__main__":
    main()
