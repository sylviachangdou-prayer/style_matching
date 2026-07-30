#!/usr/bin/env python3
"""Verify a short author list against Gutendex without scanning its full catalog."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.fetch_gutendex import (
    author_match,
    fetch_json,
    independent_title_key,
    text_url,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-works", type=int, default=3)
    parser.add_argument("--max-pages-per-author", type=int, default=8)
    parser.add_argument("--workers", type=int, default=6)
    return parser.parse_args()


def verified_works(name: str, query_name: str, language: str, max_pages: int) -> list[dict]:
    url = "https://gutendex.com/books/?" + urllib.parse.urlencode({
        "languages": language,
        "search": query_name,
        "copyright": "false",
        "mime_type": "text/plain",
    })
    works: dict[str, dict] = {}
    pages = 0
    while url and pages < max_pages:
        page = fetch_json(url)
        pages += 1
        for book in page.get("results", []):
            if (
                book.get("copyright") is True
                or book.get("translators")
                or len(book.get("authors", [])) != 1
                or language not in book.get("languages", [])
                or not author_match(book, query_name)
                or not text_url(book)
            ):
                continue
            title_key = independent_title_key(str(book.get("title", "")))
            if title_key:
                works.setdefault(title_key, {
                    "id": int(book["id"]),
                    "title": str(book.get("title", "")),
                    "download_count": int(book.get("download_count", 0)),
                })
        url = page.get("next")
    return sorted(works.values(), key=lambda row: (-row["download_count"], row["id"]))


def main() -> None:
    args = parse_args()
    with args.candidates.open(encoding="utf-8") as handle:
        candidates = list(csv.DictReader(handle))

    rows_by_position = {}
    failures_by_position = {}

    def verify(position: int, candidate: dict) -> tuple[int, dict, list[dict]]:
        name = candidate["name"].strip()
        query_name = (candidate.get("gutendex_query") or name).strip()
        language = candidate["original_language"].strip()
        print(
            f"[{position}/{len(candidates)}] verify: {language}: {name} [{query_name}]",
            flush=True,
        )
        works = verified_works(name, query_name, language, args.max_pages_per_author)
        return position, candidate, works

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(verify, position, candidate): (position, candidate)
            for position, candidate in enumerate(candidates, start=1)
        }
        for future in as_completed(futures):
            position, candidate = futures[future]
            name = candidate["name"].strip()
            try:
                _, candidate, works = future.result()
            except Exception as error:
                failures_by_position[position] = {
                    "name": name,
                    "language": candidate["original_language"].strip(),
                    "error": f"{type(error).__name__}: {error}",
                }
                print(f"WARNING verification failed: {name}: {error}", flush=True)
                continue
            if len(works) < args.min_works:
                print(
                    f"reject: {name}: {len(works)} verified independent works",
                    flush=True,
                )
                continue
            row = dict(candidate)
            row["verified_independent_works"] = str(len(works))
            row["gutenberg_ids"] = "|".join(str(work["id"]) for work in works)
            row["titles"] = " | ".join(work["title"] for work in works)
            row["eligible"] = "true"
            rows_by_position[position] = row
            print(f"accept: {name}: {len(works)} verified independent works", flush=True)

    rows = [rows_by_position[position] for position in sorted(rows_by_position)]
    failures = [
        failures_by_position[position] for position in sorted(failures_by_position)
    ]

    fields = list(candidates[0]) + [
        "verified_independent_works", "gutenberg_ids", "titles", "eligible"
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    report = {
        "candidate_file": str(args.candidates),
        "output": str(args.output),
        "candidates": len(candidates),
        "eligible": len(rows),
        "min_works": args.min_works,
        "max_pages_per_author": args.max_pages_per_author,
        "workers": args.workers,
        "failures": failures,
    }
    args.output.with_suffix(".report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
