from __future__ import annotations

import argparse
import csv
import re
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FIELDS = [
    "corpus",
    "author_or_speaker",
    "title",
    "source_id",
    "source_url",
    "source_text_rule",
    "language",
    "raw_text_path",
]
REQUIRED = {"corpus", "name", "original_language", "title", "source_id", "local_text_path"}


def slug(value: str) -> str:
    value = value.lower().replace("&", "and")
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return value.strip("_")


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = REQUIRED - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"manifest missing columns: {sorted(missing)}")
        return list(reader)


def import_rows(rows: list[dict[str, str]], manifest_dir: Path) -> dict[str, list[dict[str, str]]]:
    by_corpus: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        corpus = row["corpus"]
        name = row["name"]
        language = row["original_language"]
        source_id = row["source_id"]
        local_path = Path(row["local_text_path"])
        if not local_path.is_absolute():
            local_path = manifest_dir / local_path
        if not local_path.exists():
            raise FileNotFoundError(local_path)

        out_dir = ROOT / "data" / corpus / "raw" / slug(name)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{slug(source_id)}.txt"
        shutil.copy2(local_path, out_path)

        by_corpus.setdefault(corpus, []).append({
            "corpus": corpus,
            "author_or_speaker": name,
            "title": row["title"],
            "source_id": source_id,
            "source_url": row.get("source_url", ""),
            "source_text_rule": "original-language source text only",
            "language": language,
            "raw_text_path": str(out_path.relative_to(ROOT)),
        })
    return by_corpus


def write_sources(corpus: str, rows: list[dict[str, str]], append: bool) -> None:
    meta_dir = ROOT / "data" / corpus / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    out_path = meta_dir / "sources.csv"
    existing: list[dict[str, str]] = []
    if append and out_path.exists():
        with out_path.open(encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                row["source_id"] = row.get("source_id") or row.get("gutenberg_id", "")
                existing.append({field: row.get(field, "") for field in FIELDS})
    merged = existing + rows
    seen = set()
    deduped = []
    for row in merged:
        key = (row["corpus"], row["author_or_speaker"], row["language"], row["source_id"])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(deduped)
    print(f"{corpus}: wrote {len(deduped)} source rows")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--append", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_manifest(args.manifest)
    by_corpus = import_rows(rows, args.manifest.resolve().parent)
    for corpus, corpus_rows in sorted(by_corpus.items()):
        write_sources(corpus, corpus_rows, args.append)


if __name__ == "__main__":
    main()
