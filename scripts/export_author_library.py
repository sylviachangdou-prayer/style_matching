#!/usr/bin/env python3
"""Export the canonical people registry for the static Author Library page."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


FIELDS = (
    "name",
    "corpus",
    "original_language",
    "era",
    "photo_url",
    "profile",
    "style_traits",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/source_registry/all_people.csv")
    parser.add_argument("--output", default="web/static/authors-data.js")
    args = parser.parse_args()

    with Path(args.input).open(encoding="utf-8", newline="") as source:
        profiles = [
            {field: row.get(field, "").strip() for field in FIELDS}
            for row in csv.DictReader(source)
        ]

    if any(not row["name"] or not row["profile"] or not row["style_traits"] for row in profiles):
        raise ValueError("Every registry row must have a name, profile, and unique style traits")

    authors: dict[str, dict] = {}
    for profile in profiles:
        author = authors.setdefault(
            profile["name"],
            {
                "name": profile["name"],
                "photo_url": profile["photo_url"],
                "profile": profile["profile"],
                "style_traits": profile["style_traits"],
                "original_languages": [],
                "corpora": [],
                "eras": [],
            },
        )
        for source_field, target_field in (
            ("original_language", "original_languages"),
            ("corpus", "corpora"),
            ("era", "eras"),
        ):
            value = profile[source_field]
            if value not in author[target_field]:
                author[target_field].append(value)

    payload = json.dumps(list(authors.values()), ensure_ascii=False, separators=(",", ":"))
    Path(args.output).write_text(
        "// Generated from data/source_registry/all_people.csv. Do not edit by hand.\n"
        f"window.STYLEMATCH_AUTHORS = {payload};\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
