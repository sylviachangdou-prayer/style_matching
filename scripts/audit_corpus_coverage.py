from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", choices=["literary", "rhetorical", "both"], default="both")
    parser.add_argument("--language")
    parser.add_argument("--min-sources", type=int, default=3)
    parser.add_argument("--min-chunks", type=int, default=30)
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "coverage_audit.json")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def row_key(row: dict[str, str]) -> tuple[str, str, str]:
    return (row.get("corpus", ""), row.get("language", ""), row.get("author_or_speaker", ""))


def main() -> None:
    args = parse_args()
    corpora = ["literary", "rhetorical"] if args.corpus == "both" else [args.corpus]
    sources: list[dict[str, str]] = []
    chunks: list[dict[str, str]] = []
    for corpus in corpora:
        sources.extend(read_csv(ROOT / "data" / corpus / "meta" / "sources.csv"))
        chunks.extend(read_csv(ROOT / "data" / corpus / "meta" / "chunks.csv"))

    if args.language:
        sources = [row for row in sources if row.get("language") == args.language]
        chunks = [row for row in chunks if row.get("language") == args.language]

    source_ids: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    chunk_counts: dict[tuple[str, str, str], int] = defaultdict(int)
    for row in sources:
        source_ids[row_key(row)].add(row.get("source_id") or row.get("gutenberg_id") or "")
    for row in chunks:
        chunk_counts[row_key(row)] += 1

    keys = sorted(set(source_ids) | set(chunk_counts))
    coverage = []
    for corpus, language, name in keys:
        source_count = len(source_ids[(corpus, language, name)])
        chunk_count = chunk_counts[(corpus, language, name)]
        ready = source_count >= args.min_sources and chunk_count >= args.min_chunks
        coverage.append({
            "corpus": corpus,
            "language": language,
            "author_or_speaker": name,
            "source_count": source_count,
            "chunk_count": chunk_count,
            "source_heldout_ready": ready,
        })
    coverage.sort(key=lambda row: (row["source_heldout_ready"], row["corpus"], row["language"], row["chunk_count"]))

    report = {
        "corpus": args.corpus,
        "language": args.language,
        "min_sources": args.min_sources,
        "min_chunks": args.min_chunks,
        "n_sources": len(sources),
        "n_chunks": len(chunks),
        "n_people": len({row["author_or_speaker"] for row in coverage}),
        "ready_people": sum(1 for row in coverage if row["source_heldout_ready"]),
        "not_ready": [row for row in coverage if not row["source_heldout_ready"]],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({key: report[key] for key in ["n_sources", "n_chunks", "n_people", "ready_people"]}, indent=2))
    for row in coverage:
        print(row)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
