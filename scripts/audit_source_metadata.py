from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


REQUIRED_FIELDS = {
    "corpus",
    "author_or_speaker",
    "title",
    "source_id",
    "independent_source_id",
    "language",
    "year",
    "topic",
    "domain",
    "register",
    "source_type",
    "delivered_language",
    "license_status",
    "display_allowed",
    "canonical_url",
    "raw_text_path",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit source metadata needed for evaluation and display.")
    parser.add_argument("--corpus", choices=["literary", "rhetorical", "both"], default="both")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args()


def audit(paths: list[Path]) -> dict:
    rows = []
    schema_errors = []
    missing_files = []
    for path in paths:
        if not path.exists():
            missing_files.append(str(path))
            continue
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            missing_schema = REQUIRED_FIELDS - set(reader.fieldnames or [])
            if missing_schema:
                schema_errors.append({"path": str(path), "missing_fields": sorted(missing_schema)})
            rows.extend(list(reader))
    incomplete = []
    invalid_display = []
    for row in rows:
        missing = sorted(field for field in REQUIRED_FIELDS if not str(row.get(field, "")).strip())
        if missing:
            incomplete.append({
                "corpus": row.get("corpus", ""),
                "author_or_speaker": row.get("author_or_speaker", ""),
                "source_id": row.get("source_id", ""),
                "missing_fields": missing,
            })
        display = str(row.get("display_allowed", "")).lower() == "true"
        if display and row.get("license_status", "") not in {"public_domain", "licensed", "permission_granted"}:
            invalid_display.append({
                "source_id": row.get("source_id", ""),
                "license_status": row.get("license_status", ""),
            })
    return {
        "n_sources": len(rows),
        "missing_files": missing_files,
        "schema_errors": schema_errors,
        "incomplete_sources": incomplete,
        "invalid_display_sources": invalid_display,
        "complete": bool(rows) and not missing_files and not schema_errors and not incomplete and not invalid_display,
    }


def main() -> None:
    args = parse_args()
    corpora = ["literary", "rhetorical"] if args.corpus == "both" else [args.corpus]
    report = audit([Path("data") / corpus / "meta" / "sources.csv" for corpus in corpora])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("n_sources", "complete")}, indent=2))
    if args.strict and not report["complete"]:
        raise SystemExit("Source metadata audit failed; inspect the JSON report")


if __name__ == "__main__":
    main()
