#!/usr/bin/env python3
"""Export the canonical people registry for the static Author Library page."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import pandas as pd


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
    parser.add_argument(
        "--profiles",
        help="Optional profiles.parquet; when set, export only authors present in the live index.",
    )
    parser.add_argument("--coverage-output")
    args = parser.parse_args()

    with Path(args.input).open(encoding="utf-8", newline="") as source:
        profiles = [
            {field: row.get(field, "").strip() for field in FIELDS}
            for row in csv.DictReader(source)
        ]

    if any(not row["name"] or not row["profile"] or not row["style_traits"] for row in profiles):
        raise ValueError("Every registry row must have a name, profile, and unique style traits")

    matchable = None
    if args.profiles:
        profile_frame = pd.read_parquet(args.profiles)
        matchable = set(profile_frame["author_or_speaker"].astype(str))

    authors: dict[str, dict] = {}
    for profile in profiles:
        if matchable is not None and profile["name"] not in matchable:
            continue
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
    if args.coverage_output:
        registry_names = {profile["name"] for profile in profiles}
        Path(args.coverage_output).write_text(json.dumps({
            "registry_authors": len(registry_names),
            "indexed_authors": len(matchable or registry_names),
            "exported_authors": len(authors),
            "indexed_missing_registry_metadata": sorted((matchable or set()) - registry_names),
            "registry_hidden_without_index_profile": sorted(registry_names - (matchable or registry_names)),
        }, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
