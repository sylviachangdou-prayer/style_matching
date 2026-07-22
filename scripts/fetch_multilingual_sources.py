from __future__ import annotations

import argparse
import csv
import json
import re
import time
import urllib.parse
from pathlib import Path

import requests
from bs4 import BeautifulSoup, UnicodeDammit


ROOT = Path(__file__).resolve().parents[1]
REGISTRY_DIR = ROOT / "data" / "source_registry"
CATALOG = REGISTRY_DIR / "multilingual_source_catalog.csv"
MANIFEST = REGISTRY_DIR / "source_manifest.csv"
MANIFEST_FIELDS = [
    "corpus",
    "name",
    "original_language",
    "year",
    "title",
    "source_id",
    "independent_source_id",
    "source_url",
    "topic",
    "domain",
    "register",
    "source_type",
    "delivered_language",
    "license_status",
    "display_allowed",
    "canonical_url",
    "local_text_path",
]
# Wikimedia's robot policy degrades service for generic user agents; a
# descriptive UA with a contact address gets normal treatment.
USER_AGENT = (
    "StyleMatchCorpusBuilder/0.2 "
    "(https://github.com/sylviachangdou-prayer/style_matching; sylvia.chang.dou@gmail.com) "
    "python-requests"
)


def slug(value: str) -> str:
    value = value.lower().replace("&", "and")
    return re.sub(r"[^a-z0-9]+", "_", value).strip("_")


def source_identity(row: dict[str, str]) -> str:
    explicit = row.get("independent_source_id", "").strip()
    title_key = re.sub(r"[^\w]+", "_", row["title"].casefold(), flags=re.UNICODE).strip("_")
    date_key = row.get("year", "").strip() if row.get("corpus") == "rhetorical" else ""
    return explicit or f"{date_key}_{title_key}".strip("_") or row["source_id"]


def fetch_bytes(url: str) -> bytes:
    data = bytearray()
    total_length: int | None = None
    last_error: Exception | None = None
    for attempt in range(12):
        # Force identity encoding: with gzip, Content-Length counts compressed
        # bytes while iter_content yields decompressed ones, which both broke
        # the completeness check and truncated returned bodies.
        headers = {"User-Agent": USER_AGENT, "Accept-Encoding": "identity"}
        if data:
            headers["Range"] = f"bytes={len(data)}-"
        try:
            response = requests.get(url, headers=headers, timeout=60, stream=True)
            response.raise_for_status()
            if data and response.status_code == 200:
                data.clear()
            compressed = response.headers.get("Content-Encoding", "") not in ("", "identity")
            content_range = response.headers.get("Content-Range", "")
            if compressed:
                total_length = None
            elif "/" in content_range:
                total_length = int(content_range.rsplit("/", 1)[1])
            elif response.headers.get("Content-Length"):
                total_length = len(data) + int(response.headers["Content-Length"])
            for chunk in response.iter_content(chunk_size=65_536):
                if chunk:
                    data.extend(chunk)
            if total_length is None or len(data) >= total_length:
                return bytes(data[:total_length]) if total_length else bytes(data)
        except requests.RequestException as error:
            last_error = error
            response = getattr(error, "response", None)
            if response is not None and response.status_code == 429:
                retry_after = response.headers.get("Retry-After", "")
                wait = int(retry_after) if retry_after.isdigit() else 30
                time.sleep(min(wait, 120))
                continue
        time.sleep(min(attempt + 1, 3))
    raise RuntimeError(f"download remained incomplete after retries: {url}") from last_error


def fetch_json(url: str) -> dict:
    # Chunked API responses carry no Content-Length, so a truncated body looks
    # complete to fetch_bytes; validate by parsing and retry on damage. The
    # truncation is Wikimedia edge throttling, so back off for real between
    # attempts instead of hammering.
    last_error: Exception | None = None
    for attempt in range(7):
        try:
            return json.loads(fetch_bytes(url).decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as error:
            last_error = error
            time.sleep(min(5 * 2 ** attempt, 120))
    raise RuntimeError(f"API response stayed truncated after retries: {url}") from last_error


def clean_gutenberg(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    start = re.search(r"\*\*\* START OF (?:THE|THIS) PROJECT GUTENBERG EBOOK .*?\*\*\*", text, re.I | re.S)
    if start:
        text = text[start.end():]
    end = re.search(r"\*\*\* END OF (?:THE|THIS) PROJECT GUTENBERG EBOOK .*?\*\*\*", text, re.I | re.S)
    if end:
        text = text[:end.start()]
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def extract_aozora(raw: bytes) -> str:
    decoded = UnicodeDammit(raw).unicode_markup
    if not decoded:
        raise ValueError("Could not decode Aozora HTML")
    soup = BeautifulSoup(decoded, "html.parser")
    main = soup.select_one(".main_text")
    if main is None:
        raise ValueError("Aozora page has no .main_text element")
    for node in main.select("rt, rp, script, style"):
        node.decompose()
    return re.sub(r"\n{3,}", "\n\n", main.get_text("\n", strip=True)).strip()


def wikisource_title(url: str) -> str:
    path = urllib.parse.unquote(urllib.parse.urlsplit(url).path)
    return path.split("/wiki/", 1)[1].replace("_", " ")


def wikisource_api(language: str, params: dict[str, str]) -> dict:
    # Politeness per Wikimedia etiquette: pause between calls and send maxlag
    # so overloaded replicas can ask us to wait instead of cutting us off.
    query = urllib.parse.urlencode(
        {**params, "maxlag": "5", "format": "json", "formatversion": "2"}
    )
    url = f"https://{language}.wikisource.org/w/api.php?{query}"
    for attempt in range(6):
        time.sleep(0.5)
        payload = fetch_json(url)
        if payload.get("error", {}).get("code") != "maxlag":
            return payload
        time.sleep(10 * (attempt + 1))
    raise RuntimeError(f"Wikisource stayed lagged after retries: {url}")


def wikisource_extract(language: str, title: str) -> str:
    # action=parse renders ProofreadPage transclusions (<pages index=...>),
    # which TextExtracts returns as empty; most Wikisource novels use them.
    payload = wikisource_api(language, {"action": "parse", "page": title, "prop": "text"})
    html = payload.get("parse", {}).get("text", "")
    if not html:
        return ""
    soup = BeautifulSoup(html, "html.parser")
    for node in soup.select("table, style, script, sup.reference, .mw-editsection, .noprint, .printfooter"):
        node.decompose()
    text = soup.get_text("\n", strip=True)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def wikisource_subpages(language: str, title: str) -> list[str]:
    pages: list[str] = []
    continuation = ""
    while True:
        params = {
            "action": "query",
            "list": "allpages",
            "apprefix": f"{title}/",
            "apnamespace": "0",
            "aplimit": "max",
        }
        if continuation:
            params["apcontinue"] = continuation
        payload = wikisource_api(language, params)
        pages.extend(page["title"] for page in payload.get("query", {}).get("allpages", []))
        continuation = payload.get("continue", {}).get("apcontinue", "")
        if not continuation:
            break
    excluded = ("/Версия", "/Примечания", "/ДО")
    return [page for page in pages if not any(marker in page for marker in excluded)]


def extract_wikisource(language: str, url: str) -> str:
    title = wikisource_title(url)
    subpages = wikisource_subpages(language, title)
    titles = subpages or [title]
    extracts = [wikisource_extract(language, page).strip() for page in titles]
    return "\n\n".join(extract for extract in extracts if extract)


def script_ratio(text: str, language: str) -> float:
    if language == "zh":
        expected = re.findall(r"[\u3400-\u9fff]", text)
        denominator = re.findall(r"[^\W\d_]", text)
    elif language == "ja":
        expected = re.findall(r"[\u3040-\u30ff\u3400-\u9fff]", text)
        denominator = re.findall(r"[^\W\d_]", text)
    elif language == "ru":
        expected = re.findall(r"[А-Яа-яЁё]", text)
        denominator = re.findall(r"[^\W\d_]", text)
    else:
        # Ā-ſ (Latin Extended-A) covers Polish ł/ż/ś/ą and similar diacritics.
        expected = re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿÄÖÜäöüßĀ-ſ]", text)
        denominator = re.findall(r"[^\W\d_]", text)
    return len(expected) / max(len(denominator), 1)


def validate_text(text: str, language: str, source_id: str) -> None:
    compact_length = len(re.sub(r"\s+", "", text))
    # CJK characters are denser than Latin words: 2,000 hanzi/kana is already a
    # substantial essay, while 5,000 Latin characters is only ~900 words.
    minimum_length = 2_000 if language in {"zh", "ja"} else 5_000
    if compact_length < minimum_length:
        raise ValueError(f"{source_id}: text too short ({compact_length} non-space characters)")
    minimum = 0.50 if language in {"zh", "ja"} else 0.75
    ratio = script_ratio(text, language)
    if ratio < minimum:
        raise ValueError(f"{source_id}: unexpected script ratio {ratio:.3f} for {language}")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_manifest(rows: list[dict[str, str]], manifest: Path = MANIFEST) -> None:
    existing = read_csv(manifest) if manifest.exists() else []
    merged = {
        (row["corpus"], row["name"], row["original_language"], row["source_id"]): row
        for row in existing + rows
    }
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(merged.values())


def fetch_row(row: dict[str, str]) -> dict[str, str]:
    source_format = row["source_format"]
    if source_format == "gutenberg_text":
        match = re.search(r"/ebooks/(\d+)", row["source_url"])
        if not match:
            raise ValueError(f"Invalid Gutenberg URL: {row['source_url']}")
        book_id = match.group(1)
        download_url = f"https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt"
        text = clean_gutenberg(fetch_bytes(download_url).decode("utf-8", errors="replace"))
    elif source_format == "aozora_html":
        text = extract_aozora(fetch_bytes(row["source_url"]))
    elif source_format == "wikisource_subpages":
        text = extract_wikisource(row["original_language"], row["source_url"])
    elif source_format == "approved_remote_text":
        download_url = row["source_url"]
        parsed = urllib.parse.urlsplit(download_url)
        if parsed.netloc == "github.com" and "/blob/" in parsed.path:
            repository_path, file_path = parsed.path.split("/blob/", 1)
            download_url = f"https://raw.githubusercontent.com{repository_path}/{file_path}"
        decoded = UnicodeDammit(fetch_bytes(download_url)).unicode_markup
        if not decoded:
            raise ValueError(f"Could not decode approved remote text: {row['source_id']}")
        text = decoded.strip()
    elif source_format == "http_text":
        text = fetch_bytes(row["source_url"]).decode("utf-8", errors="replace").strip()
    elif source_format == "local_text":
        # Rights-cleared, in-copyright works the owner supplies herself: the file
        # must already sit in raw_inputs/ under the standard name. Never fetched.
        local_path = REGISTRY_DIR / "raw_inputs" / f"{slug(row['name'])}_{slug(row['source_id'])}.txt"
        if not local_path.exists():
            raise FileNotFoundError(
                f"rights-cleared text missing: put the file at {local_path} and rerun"
            )
        text = local_path.read_text(encoding="utf-8")
    else:
        raise ValueError(f"Unsupported source_format: {source_format}")

    validate_text(text, row["original_language"], row["source_id"])
    filename = f"{slug(row['name'])}_{slug(row['source_id'])}.txt"
    relative_path = Path("raw_inputs") / filename
    output_path = REGISTRY_DIR / relative_path
    output_path.write_text(text.rstrip("\n") + "\n", encoding="utf-8")
    return row_manifest_metadata(row) | {"local_text_path": str(relative_path)}


def row_manifest_metadata(row: dict[str, str]) -> dict[str, str]:
    # local_text rows are privately cleared for research/indexing, not for
    # public passage display, unless the catalog row explicitly overrides.
    local = row.get("source_format") == "local_text"
    approved_remote = row.get("source_format") == "approved_remote_text"
    metadata = {field: row.get(field, "") for field in MANIFEST_FIELDS[:-1]}
    metadata.update({
        "independent_source_id": source_identity(row),
        "domain": row.get("domain", "literature"),
        "register": row.get("register", "literary_prose"),
        "source_type": row.get("source_type", "work"),
        "delivered_language": row["original_language"],
        "license_status": row.get("license_status") or (
            "rights_cleared_research" if approved_remote else "rights_cleared_private" if local else "public_domain"
        ),
        "display_allowed": row.get("display_allowed") or ("false" if local or approved_remote else "true"),
        "canonical_url": row.get("canonical_url", row["source_url"]),
    })
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch curated original-language literary sources.")
    parser.add_argument("--catalog", type=Path, default=CATALOG)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--language", action="append")
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Keep an existing raw input and continue; useful after a Colab reconnect.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_csv(args.catalog)
    languages = set(args.language or [])
    if languages:
        rows = [row for row in rows if row["original_language"] in languages]
    manifest_rows = []
    failures: list[tuple[str, str]] = []
    for row in rows:
        print(f"fetch {row['original_language']} | {row['name']} | {row['title']}", flush=True)
        existing_path = REGISTRY_DIR / "raw_inputs" / f"{slug(row['name'])}_{slug(row['source_id'])}.txt"
        if args.skip_existing and existing_path.exists() and row["source_format"] != "local_text":
            print(f"skip existing {existing_path.name}", flush=True)
            manifest_rows.append(
                row_manifest_metadata(row)
                | {"local_text_path": str(existing_path.relative_to(REGISTRY_DIR))}
            )
            continue
        try:
            manifest_rows.append(fetch_row(row))
        except Exception as error:  # noqa: BLE001 - keep the batch alive, fail loudly at the end
            failures.append((row["source_id"], f"{type(error).__name__}: {error}"))
            print(f"FAILED {row['source_id']}: {error}", flush=True)
    write_manifest(manifest_rows, args.manifest)
    print(f"Wrote {len(manifest_rows)} original-language sources and updated {args.manifest}")
    if failures:
        print(f"\n{len(failures)} source(s) failed; successes were kept and rerunning with --skip-existing resumes here:")
        for source_id, message in failures:
            print(f"  {source_id}: {message}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
