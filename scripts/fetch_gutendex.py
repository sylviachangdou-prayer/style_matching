from __future__ import annotations

import csv
import http.client
import json
import re
import time
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


def read_names(path: Path, column: str) -> list[str]:
    with path.open(encoding="utf-8") as handle:
        rows = []
        for row in csv.DictReader(handle):
            if row.get("original_language", "en") == "en":
                rows.append(row[column])
        return rows


def fetch_json(url: str) -> dict:
    request = urllib.request.Request(url, headers={"User-Agent": "StyleMatch corpus builder (academic prototype)"})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                return json.loads(response.read().decode("utf-8"))
        except (HTTPError, URLError, TimeoutError) as error:
            if attempt == 2:
                raise error
            time.sleep(1 + attempt)
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
    end = re.search(r"\*\*\* END OF (?:THE|THIS) PROJECT GUTENBERG EBOOK .*?\*\*\*", text, flags=re.I | re.S)
    if start:
        text = text[start.end():]
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


def author_match(book: dict, name: str) -> bool:
    wanted_tokens = set(re.findall(r"[a-z]+", name.lower()))
    for author in book.get("authors", []):
        author_tokens = set(re.findall(r"[a-z]+", author.get("name", "").lower()))
        if wanted_tokens and wanted_tokens.issubset(author_tokens):
            return True
    return False


def fetch_for_name(name: str, corpus: str, max_works: int) -> list[dict[str, str]]:
    query = urllib.parse.urlencode({"languages": "en", "search": name})
    url = f"https://gutendex.com/books/?{query}"
    rows = []
    page = fetch_json(url)
    books = page.get("results", [])

    for book in books:
        if len(rows) >= max_works:
            break
        if not author_match(book, name):
            continue
        if corpus == "rhetorical" and not looks_like_rhetorical(book):
            continue
        if corpus == "literary" and not looks_like_literary(book):
            continue

        url = text_url(book)
        if not url:
            continue

        try:
            print(f"download: {corpus}: {name}: {book.get('id')} {book.get('title', '')[:80]}", flush=True)
            raw = fetch_text(url)
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
            "gutenberg_id": str(book["id"]),
            "source_url": url,
            "source_text_rule": "original-language source text only",
            "language": "en",
            "raw_text_path": str(text_path.relative_to(ROOT)),
        }
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        rows.append(meta)
        time.sleep(0.25)

    return rows


def write_metadata(corpus: str, rows: list[dict[str, str]]) -> None:
    meta_dir = ROOT / "data" / corpus / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    path = meta_dir / "sources.csv"
    fields = ["corpus", "author_or_speaker", "title", "gutenberg_id", "source_url", "source_text_rule", "language", "raw_text_path"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    literary = read_names(REGISTRY_DIR / "literary_authors.csv", "author")
    rhetorical = read_names(REGISTRY_DIR / "rhetorical_speakers.csv", "speaker")

    literary_rows = []
    for name in literary:
        literary_rows.extend(fetch_for_name(name, "literary", max_works=1))
        print(f"literary: {name}: {len([r for r in literary_rows if r['author_or_speaker'] == name])} works", flush=True)

    rhetorical_rows = []
    for name in rhetorical:
        rhetorical_rows.extend(fetch_for_name(name, "rhetorical", max_works=1))
        print(f"rhetorical: {name}: {len([r for r in rhetorical_rows if r['author_or_speaker'] == name])} works", flush=True)

    write_metadata("literary", literary_rows)
    write_metadata("rhetorical", rhetorical_rows)
    print(f"Wrote {len(literary_rows)} literary sources and {len(rhetorical_rows)} rhetorical sources")


if __name__ == "__main__":
    main()
