#!/usr/bin/env python3
"""Append reviewed author metadata without changing existing registry rows."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path, default=Path("data/source_registry/all_people.csv"))
    parser.add_argument("--additions", type=Path, required=True)
    return parser.parse_args()


def read_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def main() -> None:
    args = parse_args()
    fields, existing = read_rows(args.registry)
    addition_fields, additions = read_rows(args.additions)
    if addition_fields != fields:
        raise ValueError("Addition columns must exactly match the canonical registry")

    existing_names = {row["name"].strip() for row in existing}
    existing_rows = {
        tuple(row.get(field, "") for field in fields) for row in existing
    }
    addition_names = [row["name"].strip() for row in additions]
    if len(addition_names) != len(set(addition_names)):
        raise ValueError("Addition file contains duplicate author names")
    to_append = []
    for row in additions:
        encoded = tuple(row.get(field, "") for field in fields)
        if encoded in existing_rows:
            continue
        if row["name"].strip() in existing_names:
            raise ValueError(f"Refusing to overwrite existing author metadata: {row['name']}")
        to_append.append(row)
    for row in additions:
        traits = [value.strip() for value in row["style_traits"].split(",") if value.strip()]
        if row["profile"].count(";") != 2 or not 3 <= len(traits) <= 5:
            raise ValueError(f"Invalid profile/traits contract: {row['name']}")

    if not to_append:
        print(
            f"Appended 0 authors; reused {len(additions)}; "
            f"preserved {len(existing)} existing rows"
        )
        return

    with args.registry.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writerows(to_append)
    print(
        f"Appended {len(to_append)} authors; reused {len(additions) - len(to_append)}; "
        f"preserved {len(existing)} existing rows"
    )


if __name__ == "__main__":
    main()
