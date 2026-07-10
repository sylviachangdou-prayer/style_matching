from __future__ import annotations

import argparse
import csv
import html
import json
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build chunk parquet directly from source texts without per-chunk files.")
    parser.add_argument("--corpus", choices=["literary", "rhetorical", "both"], default="literary")
    parser.add_argument("--language")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--coverage-output", type=Path)
    parser.add_argument("--min-words", type=int, default=75)
    parser.add_argument("--max-words", type=int, default=150)
    parser.add_argument("--min-cjk-chars", type=int, default=250)
    parser.add_argument("--max-cjk-chars", type=int, default=500)
    parser.add_argument("--min-sources", type=int, default=3)
    parser.add_argument("--min-chunks", type=int, default=30)
    return parser.parse_args()


def safe_id(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def normalize_source_text(text: str) -> str:
    """Remove residual HTML wrappers from plain-text source downloads."""
    text = html.unescape(text).replace("\xa0", " ").replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def chunk_words(text: str, min_words: int, max_words: int) -> list[str]:
    words = re.findall(r"\S+", text)
    chunks = []
    for start in range(0, len(words), max_words):
        chunk = words[start:start + max_words]
        if len(chunk) >= min_words:
            chunks.append(" ".join(chunk))
    return chunks


def chunk_cjk(text: str, min_chars: int, max_chars: int) -> list[str]:
    text = re.sub(r"[ \t\r\f\v]+", "", text)
    text = re.sub(r"\n{2,}", "\n", text).strip()
    sentences = [part for part in re.split(r"(?<=[。！？!?])", text) if part]
    chunks: list[str] = []
    current = ""
    for sentence in sentences:
        while len(sentence) > max_chars:
            room = max_chars - len(current)
            if room > 0:
                current += sentence[:room]
                sentence = sentence[room:]
            if len(current) >= min_chars:
                chunks.append(current)
            current = ""
        if current and len(current) + len(sentence) > max_chars:
            if len(current) >= min_chars:
                chunks.append(current)
            current = ""
        current += sentence
    if len(current) >= min_chars:
        chunks.append(current)
    return chunks


def chunk_text(text: str, language: str, args: argparse.Namespace) -> list[str]:
    if language in {"zh", "ja"}:
        return chunk_cjk(text, args.min_cjk_chars, args.max_cjk_chars)
    return chunk_words(text, args.min_words, args.max_words)


def read_sources(corpus: str) -> list[dict[str, str]]:
    path = ROOT / "data" / corpus / "meta" / "sources.csv"
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def build_rows(corpus: str, args: argparse.Namespace) -> list[dict[str, str | int]]:
    rows = []
    for source in read_sources(corpus):
        language = source.get("language") or source.get("original_language") or "und"
        if args.language and language != args.language:
            continue
        source_id = source.get("source_id") or source.get("gutenberg_id")
        if not source_id:
            raise ValueError(f"missing source_id for {source.get('title', '')}")
        raw_path = ROOT / source["raw_text_path"]
        if not raw_path.exists():
            raise FileNotFoundError(raw_path)
        text = normalize_source_text(raw_path.read_text(encoding="utf-8", errors="replace"))
        for index, chunk in enumerate(chunk_text(text, language, args), start=1):
            rows.append({
                "chunk_id": f"{corpus}_{safe_id(source_id)}_{index:04d}",
                "corpus": corpus,
                "author_or_speaker": source["author_or_speaker"],
                "title": source.get("title", ""),
                "source_id": source_id,
                "language": language,
                "word_count": len(chunk.split()),
                "char_count": len(chunk),
                "text": chunk,
            })
    return rows


def coverage(rows: list[dict[str, str | int]], args: argparse.Namespace) -> dict:
    source_ids: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    chunk_counts: dict[tuple[str, str, str], int] = defaultdict(int)
    source_corpora: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    for row in rows:
        key = (str(row["language"]), str(row["author_or_speaker"]), "profile")
        source_ids[key].add(f"{row['corpus']}::{row['source_id']}")
        chunk_counts[key] += 1
        source_corpora[key].add(str(row["corpus"]))

    people = []
    for language, name, _ in sorted(set(source_ids) | set(chunk_counts)):
        key = (language, name, "profile")
        source_count = len(source_ids[key])
        chunk_count = chunk_counts[key]
        ready = source_count >= args.min_sources and chunk_count >= args.min_chunks
        people.append({
            "language": language,
            "author_or_speaker": name,
            "source_count": source_count,
            "chunk_count": chunk_count,
            "source_corpora": sorted(source_corpora[key]),
            "source_heldout_ready": ready,
        })
    people.sort(key=lambda row: (row["source_heldout_ready"], row["language"], row["author_or_speaker"]))
    source_keys = {
        (str(row["language"]), str(row["corpus"]), str(row["source_id"]))
        for row in rows
    }
    source_count_by_language = defaultdict(int)
    source_count_by_language_corpus = defaultdict(int)
    for language, corpus, _ in source_keys:
        source_count_by_language[language] += 1
        source_count_by_language_corpus[f"{language}::{corpus}"] += 1
    return {
        "corpus": args.corpus,
        "language": args.language,
        "min_sources": args.min_sources,
        "min_chunks": args.min_chunks,
        "n_chunks": len(rows),
        "n_authors": len({row["author_or_speaker"] for row in people}),
        "n_people": len({row["author_or_speaker"] for row in people}),
        "n_author_language_profiles": len(people),
        "n_sources": len(source_keys),
        "source_count_by_language": dict(sorted(source_count_by_language.items())),
        "source_count_by_language_corpus": dict(sorted(source_count_by_language_corpus.items())),
        "ready_people": sum(1 for row in people if row["source_heldout_ready"]),
        "people": people,
        "not_ready": [row for row in people if not row["source_heldout_ready"]],
    }


def main() -> None:
    args = parse_args()
    corpora = ["literary", "rhetorical"] if args.corpus == "both" else [args.corpus]
    rows = []
    for corpus in corpora:
        rows.extend(build_rows(corpus, args))
    if not rows:
        raise ValueError("No chunks produced.")

    df = pd.DataFrame(rows)
    if df["chunk_id"].duplicated().any():
        duplicated = df.loc[df["chunk_id"].duplicated(), "chunk_id"].head().tolist()
        raise ValueError(f"duplicate chunk_id values: {duplicated}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.output, index=False)

    report = coverage(rows, args)
    if args.coverage_output:
        args.coverage_output.parent.mkdir(parents=True, exist_ok=True)
        args.coverage_output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({key: report[key] for key in ["n_chunks", "n_authors", "n_author_language_profiles", "ready_people"]}, indent=2))
    print(f"Wrote {args.output}")
    if args.coverage_output:
        print(f"Wrote {args.coverage_output}")


if __name__ == "__main__":
    main()
