from __future__ import annotations

import argparse
import csv
import http.client
import json
import re
import time
import unicodedata
import urllib.parse
import urllib.request
from urllib.error import HTTPError, URLError
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REGISTRY_DIR = ROOT / "data" / "source_registry"


def slug(value: str) -> str:
    value = value.lower().replace("&", "and")
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return value.strip("_")


def read_registry_names(
    path: Path,
    corpus: str,
    language: str,
    batches: set[str] | None,
) -> list[tuple[str, str]]:
    with path.open(encoding="utf-8") as handle:
        names = []
        for row in csv.DictReader(handle):
            if row.get("corpus") != corpus:
                continue
            if row.get("original_language") != language:
                continue
            if batches and row.get("batch") not in batches:
                continue
            names.append((row["name"], row.get("gutendex_query") or row["name"]))
        return names


def fetch_json(url: str) -> dict:
    request = urllib.request.Request(url, headers={"User-Agent": "StyleMatch corpus builder (academic prototype)"})
    for attempt in range(5):
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                return json.loads(response.read().decode("utf-8"))
        except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as error:
            if attempt == 4:
                raise error
            time.sleep(2 ** (attempt + 1))
    raise RuntimeError(f"unreachable fetch_json retry state: {url}")


def fetch_text(url: str) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": "StyleMatch corpus builder (academic prototype)"})
    for attempt in range(3):
        try:
            chunks = []
            started = time.monotonic()
            with urllib.request.urlopen(request, timeout=10) as response:
                while True:
                    if time.monotonic() - started > 60:
                        raise TimeoutError(f"download exceeded 60s: {url}")
                    chunk = response.read(65536)
                    if not chunk:
                        break
                    chunks.append(chunk)
            return b"".join(chunks).decode("utf-8", errors="replace")
        except (HTTPError, URLError, TimeoutError, http.client.IncompleteRead) as error:
            if attempt == 2:
                raise error
            time.sleep(1 + attempt)
    raise RuntimeError(f"unreachable fetch_text retry state: {url}")


def text_url(book: dict) -> str | None:
    formats = book.get("formats", {})
    for key, value in formats.items():
        if key.startswith("text/plain") and not value.endswith(".zip"):
            return value
    return None


def clean_gutenberg(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    start = re.search(r"\*\*\* START OF (?:THE|THIS) PROJECT GUTENBERG EBOOK .*?\*\*\*", text, flags=re.I | re.S)
    if start:
        text = text[start.end():]
    end = re.search(r"\*\*\* END OF (?:THE|THIS) PROJECT GUTENBERG EBOOK .*?\*\*\*", text, flags=re.I | re.S)
    if end:
        text = text[:end.start()]
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def looks_like_literary(book: dict) -> bool:
    title = book.get("title", "").lower()
    subjects = " ".join(book.get("subjects", [])).lower()
    excluded = ["dictionary", "bibliography", "catalog", "index", "manual"]
    if any(word in title for word in excluded):
        return False
    return any(word in subjects for word in ["fiction", "novel", "stories", "literature"]) or True


def looks_like_rhetorical(book: dict) -> bool:
    title = book.get("title", "").lower()
    subjects = " ".join(book.get("subjects", [])).lower()
    keep = ["speech", "speeches", "address", "addresses", "state of the union", "inaugural", "messages and papers"]
    return any(word in title or word in subjects for word in keep)


def name_tokens(value: str) -> set[str]:
    folded = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii").lower()
    return set(re.findall(r"[a-z]+", folded)) - {"de", "del", "la", "von", "van"}


def author_match(book: dict, name: str) -> bool:
    wanted_tokens = name_tokens(name)
    for author in book.get("authors", []):
        author_tokens = name_tokens(author.get("name", ""))
        if wanted_tokens and wanted_tokens.issubset(author_tokens):
            return True
    return False


def independent_title_key(title: str) -> str:
    value = unicodedata.normalize("NFKD", title).encode("ascii", "ignore").decode("ascii").lower()
    value = re.sub(r"\b(?:volume|vol|part|tome|tom|band|libro|book)\s*[ivxlcdm\d]+\b", "", value)
    return re.sub(r"[^a-z0-9]+", "_", value).strip("_")


def fetch_for_name(
    name: str,
    corpus: str,
    language: str,
    max_works: int,
    existing_ids: set[str] | None = None,
    query_name: str | None = None,
) -> list[dict[str, str]]:
    # Announce before the first API call: gutendex can be slow, and silent
    # minutes-long queries are indistinguishable from a hang in Colab.
    query_name = query_name or name
    print(f"query: {corpus}: {name} [{query_name}]", flush=True)
    query = urllib.parse.urlencode({"languages": language, "search": query_name})
    url = f"https://gutendex.com/books/?{query}"
    rows = []
    independent_titles: set[str] = set()
    while url and (max_works <= 0 or len(rows) < max_works):
        page = fetch_json(url)
        for book in page.get("results", []):
            if max_works > 0 and len(rows) >= max_works:
                break
            if not author_match(book, query_name):
                continue
            if language not in book.get("languages", []):
                continue
            if book.get("copyright") is True or book.get("translators"):
                continue
            if len(book.get("authors", [])) != 1:
                continue
            title_key = independent_title_key(book.get("title", ""))
            if not title_key or title_key in independent_titles:
                continue
            source_id = f"gutenberg_{book['id']}"
            if existing_ids and source_id in existing_ids:
                continue
            if corpus == "rhetorical" and not looks_like_rhetorical(book):
                continue
            if corpus == "literary" and not looks_like_literary(book):
                continue

            download_url = text_url(book)
            if not download_url:
                continue

            try:
                print(f"download: {corpus}: {name}: {book.get('id')} {book.get('title', '')[:80]}", flush=True)
                raw = fetch_text(download_url)
            except (HTTPError, URLError, TimeoutError, http.client.IncompleteRead) as error:
                print(f"skip download: {name}: {book.get('id')}: {error}", flush=True)
                continue
            cleaned = clean_gutenberg(raw)
            if len(cleaned.split()) < 1000:
                continue

            out_dir = ROOT / "data" / corpus / "raw" / slug(name)
            out_dir.mkdir(parents=True, exist_ok=True)
            text_path = out_dir / f"gutenberg_{book['id']}.txt"
            meta_path = out_dir / f"gutenberg_{book['id']}.json"
            text_path.write_text(cleaned, encoding="utf-8")
            meta = {
                "corpus": corpus,
                "author_or_speaker": name,
                "title": book.get("title", ""),
                "source_id": source_id,
                "independent_source_id": title_key,
                "gutenberg_id": str(book["id"]),
                "source_url": download_url,
                "source_text_rule": "original-language source text only",
                "language": language,
                "year": "",
                "topic": "",
                "domain": "literature" if corpus == "literary" else "public_rhetoric",
                "register": "literary_prose" if corpus == "literary" else "formal_public_address",
                "source_type": "work" if corpus == "literary" else "speech_or_document",
                "delivered_language": language,
                "license_status": "public_domain",
                "display_allowed": "true",
                "canonical_url": f"https://www.gutenberg.org/ebooks/{book['id']}",
                "raw_text_path": str(text_path.relative_to(ROOT)),
            }
            meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
            rows.append(meta)
            independent_titles.add(title_key)
            if existing_ids is not None:
                existing_ids.add(source_id)
            time.sleep(0.25)
        url = page.get("next")

    return rows


def read_existing_sources(corpus: str) -> list[dict[str, str]]:
    path = ROOT / "data" / corpus / "meta" / "sources.csv"
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def existing_work_counts(rows: list[dict[str, str]]) -> dict[str, set[str]]:
    counts: dict[str, set[str]] = {}
    for row in rows:
        author = row.get("author_or_speaker", "")
        work = row.get("independent_source_id") or row.get("source_id", "")
        if author and work:
            counts.setdefault(author, set()).add(work)
    return counts


def write_metadata(corpus: str, rows: list[dict[str, str]]) -> None:
    meta_dir = ROOT / "data" / corpus / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    path = meta_dir / "sources.csv"
    fields = ["corpus", "author_or_speaker", "title", "source_id", "independent_source_id", "gutenberg_id", "source_url", "source_text_rule", "language", "year", "topic", "domain", "register", "source_type", "delivered_language", "license_status", "display_allowed", "canonical_url", "word_count", "raw_text_path"]
    existing = []
    if path.exists():
        with path.open(encoding="utf-8") as handle:
            existing = list(csv.DictReader(handle))
    merged = []
    seen = set()
    for row in existing + rows:
        for field in fields:
            row.setdefault(field, "")
        row["domain"] = row["domain"] or ("literature" if corpus == "literary" else "public_rhetoric")
        row["register"] = row["register"] or ("literary_prose" if corpus == "literary" else "formal_public_address")
        row["independent_source_id"] = row["independent_source_id"] or slug(row.get("title", "")) or row.get("source_id", "")
        row["source_type"] = row["source_type"] or ("work" if corpus == "literary" else "speech_or_document")
        row["delivered_language"] = row["delivered_language"] or row.get("language", "en")
        row["license_status"] = row["license_status"] or "public_domain"
        row["display_allowed"] = row["display_allowed"] or "true"
        gutenberg_id = row.get("gutenberg_id") or str(row.get("source_id", "")).removeprefix("gutenberg_")
        row["canonical_url"] = row["canonical_url"] or f"https://www.gutenberg.org/ebooks/{gutenberg_id}"
        key = (row.get("corpus", corpus), row.get("author_or_speaker", ""), row.get("language", ""), row.get("source_id", ""))
        if key in seen:
            continue
        seen.add(key)
        merged.append(row)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(merged)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", choices=["literary", "rhetorical", "both"], default="literary")
    parser.add_argument("--language", default="en")
    parser.add_argument("--batch", action="append", help="Registry batch to include; may be repeated.")
    parser.add_argument("--max-works", type=int, default=0, help="Maximum works per author; 0 means all available works.")
    parser.add_argument("--min-works", type=int, default=1, help="Discard authors below this independent-work count.")
    parser.add_argument("--max-authors", type=int, default=0, help="Stop after this many qualifying authors; 0 means no cap.")
    parser.add_argument("--registry", type=Path, help="Candidate registry CSV; defaults to the corpus registry.")
    parser.add_argument(
        "--skip-covered",
        action="store_true",
        help="Skip querying authors who already have at least one source; makes reruns pick up only newly registered authors.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    batches = set(args.batch) if args.batch else None
    literary = []
    rhetorical = []
    if args.corpus in {"literary", "both"}:
        literary = read_registry_names(
            args.registry or REGISTRY_DIR / "literary_authors.csv",
            "literary",
            args.language,
            batches,
        )
    if args.corpus in {"rhetorical", "both"}:
        rhetorical = read_registry_names(
            args.registry or REGISTRY_DIR / "rhetorical_speakers.csv",
            "rhetorical",
            args.language,
            batches,
        )

    literary_rows = []
    literary_sources = read_existing_sources("literary")
    literary_existing = {row.get("source_id", "") for row in literary_sources}
    literary_work_counts = existing_work_counts(literary_sources)
    unreachable: list[str] = []
    qualified_authors = 0
    for name, query_name in literary:
        if args.max_authors and qualified_authors >= args.max_authors:
            break
        existing_count = len(literary_work_counts.get(name, set()))
        if args.skip_covered and existing_count >= args.min_works:
            print(f"skip covered: literary: {name}", flush=True)
            continue
        try:
            author_rows = fetch_for_name(
                name,
                "literary",
                args.language,
                max_works=args.max_works,
                existing_ids=literary_existing,
                query_name=query_name,
            )
        except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as error:
            # A flaky gutendex query must not kill the batch; coverage audits
            # downstream surface any author left without sources.
            unreachable.append(f"literary: {name}")
            print(f"WARNING query failed: literary: {name}: {error}", flush=True)
            continue
        total_count = existing_count + len({
            row.get("independent_source_id") or row.get("source_id", "")
            for row in author_rows
        })
        if total_count >= args.min_works:
            literary_rows.extend(author_rows)
            qualified_authors += 1
        else:
            print(f"discard: literary: {name}: only {total_count} independent works", flush=True)
        print(f"literary: {name}: {total_count} total works ({len(author_rows)} new)", flush=True)

    rhetorical_rows = []
    rhetorical_sources = read_existing_sources("rhetorical")
    rhetorical_existing = {row.get("source_id", "") for row in rhetorical_sources}
    rhetorical_work_counts = existing_work_counts(rhetorical_sources)
    for name, query_name in rhetorical:
        existing_count = len(rhetorical_work_counts.get(name, set()))
        if args.skip_covered and existing_count >= args.min_works:
            print(f"skip covered: rhetorical: {name}", flush=True)
            continue
        try:
            author_rows = fetch_for_name(
                name,
                "rhetorical",
                args.language,
                max_works=args.max_works,
                existing_ids=rhetorical_existing,
                query_name=query_name,
            )
        except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as error:
            unreachable.append(f"rhetorical: {name}")
            print(f"WARNING query failed: rhetorical: {name}: {error}", flush=True)
            continue
        total_count = existing_count + len({
            row.get("independent_source_id") or row.get("source_id", "")
            for row in author_rows
        })
        if total_count >= args.min_works:
            rhetorical_rows.extend(author_rows)
        else:
            print(f"discard: rhetorical: {name}: only {total_count} independent works", flush=True)
        print(f"rhetorical: {name}: {total_count} total works ({len(author_rows)} new)", flush=True)

    write_metadata("literary", literary_rows)
    write_metadata("rhetorical", rhetorical_rows)
    print(f"Wrote {len(literary_rows)} literary sources and {len(rhetorical_rows)} rhetorical sources")
    if unreachable:
        print(f"WARNING {len(unreachable)} author(s) could not be queried this run; rerun later to retry:")
        for entry in unreachable:
            print(f"  {entry}")


if __name__ == "__main__":
    main()
