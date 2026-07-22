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
    "independent_source_id",
    "source_url",
    "source_text_rule",
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
    "word_count",
    "raw_text_path",
]
REQUIRED = {"corpus", "name", "original_language", "title", "source_id", "local_text_path"}
REGISTRY_PATH = ROOT / "data" / "source_registry" / "all_people.csv"
REGISTRY_DIR = REGISTRY_PATH.parent


def slug(value: str) -> str:
    value = value.lower().replace("&", "and")
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return value.strip("_")


def source_identity(corpus: str, title: str, year: str, fallback: str) -> str:
    title_key = re.sub(r"[^\w]+", "_", title.casefold(), flags=re.UNICODE).strip("_")
    date_key = year.strip() if corpus == "rhetorical" else ""
    return f"{date_key}_{title_key}".strip("_") or fallback


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = REQUIRED - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"manifest missing columns: {sorted(missing)}")
        return list(reader)


def registry_keys() -> set[tuple[str, str, str]]:
    if not REGISTRY_PATH.exists():
        return set()
    with REGISTRY_PATH.open(encoding="utf-8") as handle:
        return {
            (row["corpus"], row["name"], row["original_language"])
            for row in csv.DictReader(handle)
        }


def resolve_local_path(value: str, manifest_dir: Path) -> tuple[Path, list[Path]]:
    path = Path(value)
    if path.is_absolute():
        return path, [path]
    candidates = [manifest_dir / path, REGISTRY_DIR / path, ROOT / path]
    return next((candidate for candidate in candidates if candidate.exists()), candidates[0]), candidates


def import_rows(
    rows: list[dict[str, str]],
    manifest_dir: Path,
    dry_run: bool,
    skip_missing: bool = False,
) -> dict[str, list[dict[str, str]]]:
    by_corpus: dict[str, list[dict[str, str]]] = {}
    allowed = registry_keys()
    for row in rows:
        corpus = row["corpus"]
        name = row["name"]
        language = row["original_language"]
        if allowed and (corpus, name, language) not in allowed:
            raise ValueError(f"manifest row not in source registry: {corpus} | {name} | {language}")
        source_id = row["source_id"]
        independent_source_id = row.get("independent_source_id", "").strip()
        if not independent_source_id:
            independent_source_id = source_identity(
                corpus, row["title"], row.get("year", ""), source_id
            )
        local_path, checked_paths = resolve_local_path(row["local_text_path"], manifest_dir)
        if not local_path.exists():
            checked = ", ".join(str(path) for path in checked_paths)
            if skip_missing:
                print(f"skip missing {source_id}: checked {checked}")
                continue
            raise FileNotFoundError(f"{source_id}: checked {checked}")

        text = local_path.read_text(encoding="utf-8", errors="replace")
        word_count = len(re.findall(r"\S+", text))
        out_dir = ROOT / "data" / corpus / "raw" / slug(name)
        out_path = out_dir / f"{slug(source_id)}.txt"
        if not dry_run:
            out_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(local_path, out_path)

        source_url = row.get("source_url", "")
        public_archive = any(
            domain in source_url for domain in ("gutenberg.org", "aozora.gr.jp", "wikisource.org")
        )
        by_corpus.setdefault(corpus, []).append({
            "corpus": corpus,
            "author_or_speaker": name,
            "title": row["title"],
            "source_id": source_id,
            "independent_source_id": independent_source_id,
            "source_url": source_url,
            "source_text_rule": "original-language source text only",
            "language": language,
            "year": row.get("year", ""),
            "topic": row.get("topic", ""),
            "domain": row.get("domain") or ("literature" if corpus == "literary" else "public_rhetoric"),
            "register": row.get("register") or ("literary_prose" if corpus == "literary" else "formal_public_address"),
            "source_type": row.get("source_type") or ("work" if corpus == "literary" else "speech_or_document"),
            "delivered_language": row.get("delivered_language", language),
            "license_status": row.get("license_status") or ("public_domain" if public_archive else "unknown"),
            "display_allowed": row.get("display_allowed") or ("true" if public_archive else "false"),
            "canonical_url": row.get("canonical_url") or source_url,
            "word_count": str(word_count),
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
    merged: dict[tuple[str, str, str, str], dict[str, str]] = {}
    order: list[tuple[str, str, str, str]] = []
    for row in existing + rows:
        key = (row["corpus"], row["author_or_speaker"], row["language"], row["source_id"])
        if key not in merged:
            merged[key] = {field: row.get(field, "") for field in FIELDS}
            order.append(key)
        else:
            merged[key].update({
                field: value for field, value in row.items() if field in FIELDS and str(value).strip()
            })
    deduped = [merged[key] for key in order]
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(deduped)
    print(f"{corpus}: wrote {len(deduped)} source rows")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--append", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--skip-missing",
        action="store_true",
        help="Import successful manifest rows and report stale or failed local paths.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_manifest(args.manifest)
    by_corpus = import_rows(
        rows,
        args.manifest.resolve().parent,
        args.dry_run,
        skip_missing=args.skip_missing,
    )
    for corpus, corpus_rows in sorted(by_corpus.items()):
        if args.dry_run:
            print(f"{corpus}: validated {len(corpus_rows)} source rows")
        else:
            write_sources(corpus, corpus_rows, args.append)


if __name__ == "__main__":
    main()
