from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.retrieval_metrics import ranking_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a lexical-content-free UPOS/dependency authorship baseline."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--eval-splits", default="dev,test")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--download-missing", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def profile_key(frame: pd.DataFrame) -> pd.Series:
    return frame["language"].astype(str) + "::" + frame["author_or_speaker"].astype(str)


def document_stream(document: object) -> str:
    tokens = []
    for sentence in document.sentences:
        words = sentence.words
        for index, word in enumerate(words):
            upos = word.upos or "X"
            dependency = word.deprel or "dep"
            if word.head and 0 < int(word.head) <= len(words):
                head_upos = words[int(word.head) - 1].upos or "X"
                direction = "L" if int(word.head) < int(word.id) else "R"
            else:
                head_upos = "ROOT"
                direction = "ROOT"
            tokens.extend(
                [
                    f"P_{upos}",
                    f"R_{dependency}",
                    f"D_{upos}_{dependency}_{head_upos}_{direction}",
                ]
            )
            if index:
                previous = words[index - 1].upos or "X"
                tokens.append(f"B_{previous}_{upos}")
    return " ".join(tokens)


def parse_syntax(df: pd.DataFrame, cache_path: Path, device: str, download_missing: bool) -> pd.DataFrame:
    if cache_path.exists():
        cache = pd.read_parquet(cache_path)
    else:
        cache = pd.DataFrame(columns=["chunk_id", "syntax_stream"])
    parsed = set(cache["chunk_id"].astype(str))
    missing = df[~df["chunk_id"].astype(str).isin(parsed)]
    if missing.empty:
        return cache
    import stanza

    cache_rows = [cache]
    for language, group in missing.groupby("language", sort=True):
        try:
            pipeline = stanza.Pipeline(
                lang=str(language),
                processors="tokenize,pos,depparse",
                use_gpu=device == "cuda",
                verbose=False,
            )
        except Exception:
            if not download_missing:
                raise
            stanza.download(str(language), processors="tokenize,pos,depparse", verbose=False)
            pipeline = stanza.Pipeline(
                lang=str(language),
                processors="tokenize,pos,depparse",
                use_gpu=device == "cuda",
                verbose=False,
            )
        rows = []
        for row in group.itertuples(index=False):
            rows.append(
                {
                    "chunk_id": str(row.chunk_id),
                    "syntax_stream": document_stream(pipeline(str(row.text or ""))),
                }
            )
        cache_rows.append(pd.DataFrame(rows))
        cache = pd.concat(cache_rows, ignore_index=True).drop_duplicates("chunk_id", keep="last")
        cache.to_parquet(cache_path, index=False)
        cache_rows = [cache]
    return cache


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    score_path = args.out_dir / "style_syntax_scores.npz"
    metrics_path = args.out_dir / "style_syntax_metrics.json"
    if args.skip_existing and score_path.exists() and metrics_path.exists():
        print(f"skip existing syntax baseline: {args.out_dir}")
        return
    df = pd.read_parquet(args.input)
    required = {"chunk_id", "author_or_speaker", "language", "corpus", "split", "text"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    cache = parse_syntax(
        df, args.out_dir / "syntax_streams.parquet", args.device, args.download_missing
    )
    df = df.merge(cache, on="chunk_id", how="left", validate="one_to_one")
    if df["syntax_stream"].isna().any():
        raise ValueError("Syntax cache is incomplete")
    df["profile_key"] = profile_key(df)
    encoder = LabelEncoder().fit(df["profile_key"])
    eval_splits = {value.strip() for value in args.eval_splits.split(",") if value.strip()}
    eval_df = df[df["split"].isin(eval_splits)].copy().reset_index(drop=True)
    scores = np.full((len(eval_df), len(encoder.classes_)), -1e9, dtype="float64")
    for language in sorted(df["language"].astype(str).unique()):
        train = df[df["language"].astype(str).eq(language) & df["split"].eq("train")]
        evaluation = eval_df[eval_df["language"].astype(str).eq(language)]
        if train.empty or evaluation.empty:
            continue
        y_train = encoder.transform(train["profile_key"])
        classes = np.unique(y_train)
        row_indices = evaluation.index.to_numpy()
        if len(classes) == 1:
            scores[row_indices, classes[0]] = 0.0
            continue
        vectorizer = TfidfVectorizer(
            analyzer="word", ngram_range=(1, 2), min_df=3, max_features=250_000, sublinear_tf=True
        )
        train_matrix = vectorizer.fit_transform(train["syntax_stream"].astype(str))
        eval_matrix = vectorizer.transform(evaluation["syntax_stream"].astype(str))
        model = LinearSVC(C=1.0).fit(train_matrix, y_train)
        decision = model.decision_function(eval_matrix)
        if decision.ndim == 1:
            decision = np.column_stack([-decision, decision])
        scores[np.ix_(row_indices, model.classes_.astype(int))] = decision
    y_eval = encoder.transform(eval_df["profile_key"])
    report = {
        "protocol": {
            "view": "UPOS, dependency relations, dependency direction, POS transitions",
            "lexical_forms_used": False,
            "evaluation_level": "chunk diagnostic; source aggregation occurs in comparison notebook",
        },
        "overall": ranking_metrics(scores, y_eval),
        "by_language": {},
    }
    for language, group in eval_df.groupby("language", sort=True):
        positions = group.index.to_numpy()
        report["by_language"][str(language)] = {
            "n_chunks": int(len(group)),
            **ranking_metrics(scores[positions], y_eval[positions]),
        }
    np.savez_compressed(
        score_path,
        chunk_ids=eval_df["chunk_id"].astype(str).to_numpy(),
        splits=eval_df["split"].astype(str).to_numpy(),
        query_languages=eval_df["language"].astype(str).to_numpy(),
        query_corpora=eval_df["corpus"].astype(str).to_numpy(),
        profiles=encoder.classes_,
        y_true=y_eval,
        syntax_scores=scores.astype("float32"),
    )
    metrics_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
