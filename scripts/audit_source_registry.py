from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = ROOT / "data" / "source_registry" / "all_people.csv"
OUT_PATH = ROOT / "data" / "source_registry" / "registry_audit.json"


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def count_by(rows: list[dict[str, str]], *keys: str) -> dict[str, int]:
    counts = Counter(tuple(row.get(key, "") for key in keys) for row in rows)
    return {" | ".join(key): value for key, value in sorted(counts.items())}


def duplicate_keys(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["name"], row["corpus"], row["original_language"])].append(row)
    return [
        {"name": key[0], "corpus": key[1], "original_language": key[2], "count": len(group)}
        for key, group in grouped.items()
        if len(group) > 1
    ]


def main() -> None:
    rows = read_rows(REGISTRY_PATH)
    audit = {
        "input": str(REGISTRY_PATH.relative_to(ROOT)),
        "n_records": len(rows),
        "by_corpus": count_by(rows, "corpus"),
        "by_corpus_language": count_by(rows, "corpus", "original_language"),
        "by_batch": count_by(rows, "batch"),
        "by_status": count_by(rows, "modeling_status"),
        "duplicates": duplicate_keys(rows),
        "registry_only": [
            {
                "name": row["name"],
                "corpus": row["corpus"],
                "original_language": row["original_language"],
                "batch": row["batch"],
                "source_family": row["source_family"],
            }
            for row in rows
            if row["modeling_status"] == "registry_only"
        ],
    }
    OUT_PATH.write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({key: audit[key] for key in ["n_records", "by_corpus", "by_corpus_language", "by_status", "duplicates"]}, indent=2))
    print(f"Wrote {OUT_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
