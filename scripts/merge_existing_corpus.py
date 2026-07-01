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
    "word_count",
    "raw_text_path",
]


def slug(value: str) -> str:
    value = value.lower().replace("&", "and")
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return value.strip("_")


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def normalize_source_id(row: dict[str, str]) -> str:
    source_id = row.get("source_id") or row.get("gutenberg_id") or ""
    if not source_id:
        raise ValueError(f"missing source_id: {row}")
    return source_id


def import_sources(source_root: Path, corpus: str) -> list[dict[str, str]]:
    source_meta = source_root / "data" / corpus / "meta" / "sources.csv"
    imported = []
    for row in read_rows(source_meta):
        source_id = normalize_source_id(row)
        name = row["author_or_speaker"]
        raw_path = source_root / row["raw_text_path"]
        if not raw_path.exists():
            raise FileNotFoundError(raw_path)

        out_dir = ROOT / "data" / corpus / "raw" / slug(name)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{slug(source_id)}.txt"
        if raw_path.resolve() != out_path.resolve():
            shutil.copy2(raw_path, out_path)
        text = out_path.read_text(encoding="utf-8", errors="replace")

        imported.append({
            "corpus": corpus,
            "author_or_speaker": name,
            "title": row.get("title", ""),
            "source_id": source_id,
            "source_url": row.get("source_url", ""),
            "source_text_rule": row.get("source_text_rule", "original-language source text only"),
            "language": row.get("language") or row.get("original_language") or "und",
            "word_count": str(len(re.findall(r"\S+", text))),
            "raw_text_path": str(out_path.relative_to(ROOT)),
        })
    return imported


def write_merged(corpus: str, rows: list[dict[str, str]]) -> None:
    out_path = ROOT / "data" / corpus / "meta" / "sources.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    existing = [{field: row.get(field, "") for field in FIELDS} for row in read_rows(out_path)]
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
    print(f"{corpus}: merged {len(rows)} imported rows; wrote {len(deduped)} total source rows")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True, help="Root containing data/<corpus>/meta/sources.csv")
    parser.add_argument("--corpus", choices=["literary", "rhetorical", "both"], default="both")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    corpora = ["literary", "rhetorical"] if args.corpus == "both" else [args.corpus]
    for corpus in corpora:
        rows = import_sources(args.source_root, corpus)
        if rows:
            write_merged(corpus, rows)
        else:
            print(f"{corpus}: no source rows found under {args.source_root}")


if __name__ == "__main__":
    main()
