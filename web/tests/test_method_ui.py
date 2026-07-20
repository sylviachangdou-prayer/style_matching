from __future__ import annotations

import json
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
STATIC = ROOT / "web" / "static"
SCHEMA = ROOT / "docs" / "architecture" / "schema.yaml"


def generated_flow() -> dict:
    source = (STATIC / "method-flow.js").read_text(encoding="utf-8")
    prefix = "// Generated from docs/architecture/schema.yaml. Do not edit by hand.\nwindow.STYLEMATCH_METHOD_FLOW = "
    assert source.startswith(prefix)
    assert source.endswith(";\n")
    return json.loads(source[len(prefix):-2])


def test_web_flow_matches_architecture_schema() -> None:
    schema = yaml.safe_load(SCHEMA.read_text(encoding="utf-8"))
    flow = generated_flow()
    assert [node["id"] for node in flow["nodes"]] == [node["id"] for node in schema["nodes"]]
    assert [
        (edge["from"], edge["to"], edge["label"])
        for edge in flow["edges"]
    ] == [
        (edge["from"], edge["to"], edge["label"])
        for edge in schema["edges"]
    ]


def test_index_exposes_primary_navigation_and_visible_passage() -> None:
    page = (STATIC / "index.html").read_text(encoding="utf-8")
    assert 'class="site-nav"' in page
    assert 'href="https://github.com/sylviachangdou-prayer"' in page
    assert "Pure style." in page and "Beyond language." in page
    assert "Cross-lingual, multi-style retrieval" in page
    assert "A multilingual cabinet of voices" not in page
    assert "Paste a passage you wrote — we compare" not in page
    assert 'href="method.html"' in page
    assert 'href="authors.html"' in page
    assert '<dialog' not in page
    assert '<details' not in page
    assert 'class="passage-match"' in page
    assert 'class="art-ribbon"' in page
    assert 'class="loading-theatre"' in page
    assert "scroll-unfurl" in (STATIC / "site-shell.css").read_text(encoding="utf-8")


def test_method_page_and_background_asset_are_packaged() -> None:
    page = (STATIC / "method.html").read_text(encoding="utf-8")
    assert 'data-method-static' in page
    assert '<script src="method-flow.js"></script>' in page
    assert '<script src="method-static.js"></script>' in page
    assert 'method-modal.js' not in page
    assert '<link rel="stylesheet" href="method-flow.css">' in page
    assert "Read the full method" not in page
    background = STATIC / "head.webp"
    assert background.stat().st_size > 100_000
    header = background.read_bytes()[:16]
    assert header[:4] == b"RIFF" and header[8:12] == b"WEBP"


def test_author_library_is_generated_from_complete_registry() -> None:
    page = (STATIC / "authors.html").read_text(encoding="utf-8")
    assert 'class="site-nav"' in page
    assert 'id="author-search"' in page
    assert '<script src="authors-data.js"></script>' in page
    registry = (ROOT / "data" / "source_registry" / "all_people.csv").read_text(encoding="utf-8")
    authors = (STATIC / "authors-data.js").read_text(encoding="utf-8")
    assert "Oscar Wilde" in registry and "jewel-cut" in registry
    assert "window.STYLEMATCH_AUTHORS" in authors
    assert "J. K. Rowling" in authors and "morally legible" in authors
    payload = json.loads(authors.split(" = ", 1)[1][:-2])
    assert len(payload) == 270
    assert all(author["profile"] and author["style_traits"] for author in payload)
    assert next(author for author in payload if author["name"] == "Hannah Arendt")["original_languages"] == ["de", "en"]
