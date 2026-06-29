from __future__ import annotations

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


def chunk_corpus(corpus: str) -> None:
    source_path = ROOT / "data" / corpus / "meta" / "sources.csv"
    if not source_path.exists():
        raise FileNotFoundError(source_path)

    out_dir = ROOT / "data" / corpus / "chunks"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    with source_path.open(encoding="utf-8") as handle:
        for source in csv.DictReader(handle):
            text = (ROOT / source["raw_text_path"]).read_text(encoding="utf-8", errors="replace")
            for index, chunk in enumerate(chunk_words(text), start=1):
                chunk_id = f"{corpus}_{source['gutenberg_id']}_{index:04d}"
                chunk_path = out_dir / f"{chunk_id}.txt"
                chunk_path.write_text(chunk, encoding="utf-8")
                rows.append({
                    "chunk_id": chunk_id,
                    "corpus": corpus,
                    "author_or_speaker": source["author_or_speaker"],
                    "title": source["title"],
                    "source_id": source["gutenberg_id"],
                    "language": "en",
                    "word_count": str(len(chunk.split())),
                    "chunk_path": str(chunk_path.relative_to(ROOT)),
                })

    fields = ["chunk_id", "corpus", "author_or_speaker", "title", "source_id", "language", "word_count", "chunk_path"]
    with (ROOT / "data" / corpus / "meta" / "chunks.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    print(f"{corpus}: wrote {len(rows)} chunks")


def main() -> None:
    chunk_corpus("literary")
    chunk_corpus("rhetorical")


if __name__ == "__main__":
    main()
