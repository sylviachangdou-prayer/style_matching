from __future__ import annotations

import csv
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def chunk_words(text: str, min_words: int = 75, max_words: int = 150) -> list[str]:
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: list[str] = []
    buffer: list[str] = []

    for paragraph in paragraphs:
        words = paragraph.split()
        if len(words) > max_words * 2:
            sentences = re.split(r"(?<=[.!?])\s+", paragraph)
            units = [s for s in sentences if s.strip()]
        else:
            units = [paragraph]

        for unit in units:
            buffer.extend(unit.split())
            if len(buffer) >= min_words:
                chunks.append(" ".join(buffer[:max_words]))
                buffer = buffer[max_words:]

    if len(buffer) >= min_words:
        chunks.append(" ".join(buffer))

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

