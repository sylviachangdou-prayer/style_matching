from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def chunk_words(text: str, min_words: int = 75, max_words: int = 150) -> list[str]:
    words = re.findall(r"\S+", text)
    chunks: list[str] = []

    for start in range(0, len(words), max_words):
        chunk = words[start:start + max_words]
        if len(chunk) >= min_words:
            chunks.append(" ".join(chunk))

    return chunks


def chunk_corpus(corpus: str, min_words: int, max_words: int, allow_missing: bool = False) -> None:
    source_path = ROOT / "data" / corpus / "meta" / "sources.csv"
    if not source_path.exists():
        if allow_missing:
            print(f"{corpus}: missing {source_path}; skipped")
            return
        raise FileNotFoundError(source_path)

    out_dir = ROOT / "data" / corpus / "chunks"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    with source_path.open(encoding="utf-8") as handle:
        for source in csv.DictReader(handle):
            text = (ROOT / source["raw_text_path"]).read_text(encoding="utf-8", errors="replace")
            source_id = source.get("source_id") or source.get("gutenberg_id")
            if not source_id:
                raise ValueError(f"missing source_id for {source.get('title', '')}")
            safe_source_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", source_id).strip("_")
            language = source.get("language") or source.get("original_language") or "und"
            for index, chunk in enumerate(chunk_words(text, min_words=min_words, max_words=max_words), start=1):
                chunk_id = f"{corpus}_{safe_source_id}_{index:04d}"
                chunk_path = out_dir / f"{chunk_id}.txt"
                chunk_path.write_text(chunk, encoding="utf-8")
                rows.append({
                    "chunk_id": chunk_id,
                    "corpus": corpus,
                    "author_or_speaker": source["author_or_speaker"],
                    "title": source["title"],
                    "source_id": source_id,
                    "language": language,
                    "word_count": str(len(chunk.split())),
                    "chunk_path": str(chunk_path.relative_to(ROOT)),
                })

    fields = ["chunk_id", "corpus", "author_or_speaker", "title", "source_id", "language", "word_count", "chunk_path"]
    with (ROOT / "data" / corpus / "meta" / "chunks.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    print(f"{corpus}: wrote {len(rows)} chunks")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", choices=["literary", "rhetorical", "both"], default="both")
    parser.add_argument("--min-words", type=int, default=75)
    parser.add_argument("--max-words", type=int, default=150)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    corpora = ["literary", "rhetorical"] if args.corpus == "both" else [args.corpus]
    for corpus in corpora:
        chunk_corpus(corpus, min_words=args.min_words, max_words=args.max_words, allow_missing=args.corpus == "both")


if __name__ == "__main__":
    main()
