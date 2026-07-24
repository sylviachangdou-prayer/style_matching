#!/usr/bin/env python3
"""Discover Project Gutenberg authors with enough independent original-language works."""

from __future__ import annotations

import argparse
import csv
import json
import time
import urllib.parse
import urllib.request
from collections import defaultdict
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.fetch_gutendex import independent_title_key, text_url


def fetch(url: str) -> dict:
    request = urllib.request.Request(
        url, headers={"User-Agent": "StyleMatch Gutenberg coverage audit"}
    )
    with urllib.request.urlopen(request, timeout=45) as response:
        return json.loads(response.read().decode("utf-8"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--language", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-works", type=int, default=3)
    parser.add_argument(
        "--earliest-author-year",
        type=int,
        default=1800,
        help="Exclude authors born before this year; estimate birth as death year minus 70 when needed.",
    )
    parser.add_argument("--max-pages", type=int, default=0)
    parser.add_argument("--sleep", type=float, default=0.15)
    return parser.parse_args()


def estimated_birth_year(author: dict) -> int | None:
    birth = author.get("birth_year")
    if birth is not None:
        return int(birth)
    death = author.get("death_year")
    return int(death) - 70 if death is not None else None


def display_name(gutenberg_name: str) -> str:
    parts = [part.strip() for part in gutenberg_name.split(",")]
    if len(parts) >= 2 and parts[0] and parts[1] and not any(char.isdigit() for char in parts[1]):
        return f"{parts[1]} {parts[0]}".strip()
    return gutenberg_name.strip()


def main() -> None:
    args = parse_args()
    works: dict[tuple[str, str], dict[str, dict]] = defaultdict(dict)
    pages_scanned = 0
    for language in args.language:
        language_pages = 0
        language_books = 0
        print(f"scan start: {language}", flush=True)
        url = "https://gutendex.com/books/?" + urllib.parse.urlencode({
            "languages": language,
            "author_year_start": args.earliest_author_year,
            "copyright": "false",
            "mime_type": "text/plain",
        })
        while url and (not args.max_pages or language_pages < args.max_pages):
            if language_pages == 0:
                print(f"request: {language}: first page", flush=True)
            page = fetch(url)
            pages_scanned += 1
            language_pages += 1
            language_books += len(page.get("results", []))
            for book in page.get("results", []):
                if (
                    book.get("copyright") is True
                    or language not in book.get("languages", [])
                    or book.get("translators")
                ):
                    continue
                if not text_url(book) or len(book.get("authors", [])) != 1:
                    continue
                author_record = book["authors"][0]
                author_year = estimated_birth_year(author_record)
                if author_year is None or author_year < args.earliest_author_year:
                    continue
                gutenberg_name = str(author_record.get("name", "")).strip()
                author = display_name(gutenberg_name)
                title_key = independent_title_key(str(book.get("title", "")))
                if not author or not title_key:
                    continue
                works[(language, author)].setdefault(
                    title_key,
                    {
                        "id": int(book["id"]),
                        "title": str(book.get("title", "")),
                        "download_count": int(book.get("download_count", 0)),
                        "gutenberg_name": gutenberg_name,
                        "estimated_birth_year": author_year,
                    },
                )
            url = page.get("next")
            if language_pages == 1 or language_pages % 25 == 0 or not url:
                qualifying = sum(
                    len(titles) >= args.min_works
                    for (candidate_language, _), titles in works.items()
                    if candidate_language == language
                )
                print(
                    f"scan progress: {language}: pages={language_pages} "
                    f"books={language_books} qualifying_profiles={qualifying}",
                    flush=True,
                )
            if url:
                time.sleep(args.sleep)
        print(
            f"scan complete: {language}: pages={language_pages} books={language_books}",
            flush=True,
        )

    rows = []
    for (language, author), titles in sorted(works.items()):
        ranked = sorted(
            titles.values(), key=lambda row: (-row["download_count"], row["id"])
        )
        if len(ranked) < args.min_works:
            continue
        rows.append({
            "name": author,
            "gutenberg_name": str(next(iter(titles.values())).get("gutenberg_name", author)),
            "estimated_birth_year": min(
                int(row["estimated_birth_year"]) for row in titles.values()
            ),
            "corpus": "literary",
            "original_language": language,
            "independent_works": len(ranked),
            "gutenberg_ids": "|".join(str(row["id"]) for row in ranked),
            "titles": " | ".join(row["title"] for row in ranked),
            "eligible": "true",
        })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "name", "gutenberg_name", "estimated_birth_year", "corpus",
        "original_language", "independent_works",
        "gutenberg_ids", "titles", "eligible",
    ]
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({
        "output": str(args.output),
        "pages_scanned": pages_scanned,
        "eligible_author_language_profiles": len(rows),
        "languages": sorted(set(args.language)),
        "min_works": args.min_works,
        "earliest_author_year": args.earliest_author_year,
    }, indent=2))


if __name__ == "__main__":
    main()
