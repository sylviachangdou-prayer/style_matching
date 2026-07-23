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
    assert 'id="home-title"' not in page
    assert "Pure style." not in page and "Beyond language." not in page
    assert "Cross-lingual, multi-style retrieval" in page
    assert '<span aria-hidden="true">✦</span> Cross-lingual' not in page
    assert page.count('class="voice-sticker"') == 9
    assert "portraits/jian-zhen.jpg" not in page
    assert "portraits/simone-de-beauvoir.jpg" not in page
    assert "portraits/eileen-chang.jpg" in page
    assert "retrieval —" not in page
    assert "academic authorship research." not in page
    assert "Original-language passages only" not in page
    assert "A multilingual cabinet of voices" not in page
    assert "Paste a passage you wrote — we compare" not in page
    assert 'href="method.html"' in page
    assert 'href="authors.html"' in page
    assert 'href="fyi.html"' in page
    assert '<dialog' not in page
    assert '<details' not in page
    assert 'class="passage-match"' in page
    assert 'class="art-ribbon"' in page
    assert 'class="loading-theatre"' in page
    transition = page.split('class="loading-theatre"', 1)[1].split("</section>", 1)[0].lower()
    assert "<img" not in transition
    assert "hand" not in transition and "person" not in transition and "human" not in transition
    assert transition.count("transition-strip") == 8
    assert 'id="transition-passage"' in transition
    assert "manuscript-sheet" not in transition and "vertical-brushes" not in transition
    css = (STATIC / "site-shell.css").read_text(encoding="utf-8")
    assert "transition-gallery" in css and "typed-manuscript" in css
    assert "manuscript-turn" not in css and "paper-burn" not in css
    assert 'url("lantingxu.jpg")' in css
    assert 'url("art/wanderer.jpg")' in css
    assert 'url("art/impression-sunrise.jpg")' in css
    assert 'url("art/beethoven-score.jpg")' in css
    assert (STATIC / "lantingxu.jpg").stat().st_size > 1_000_000
    assert "TRANSITION_MIN_MS = 10000" in page
    assert "Beowulf" in page and "First Rhapsody on the Red Cliffs" in page
    assert "Second Rhapsody on the Red Cliffs" in page
    assert page.count("author:") == 10
    assert "Your literary constellation" not in page
    assert "Style leads the ranking" not in page
    assert "Demo data." not in page and "DEMO DATA" not in page
    assert "and not yet calibrated" not in page
    assert "formatTransitionPassage" in page and "subtitle-scroll" in css
    assert "TRANSITION_EXCERPT_RATIO = 2 / 3" in page
    assert "typed-glyph" not in page and "gallery-breathe" not in css
    assert ".transition-lanting," in css and "background-size: auto 155%" in css
    assert "(original language)" not in page
    assert "See translated text" in page and "/api/translate" in page
    result_template = page.split("function matchRow", 1)[1].split("function galleryFor", 1)[0]
    assert "cross-language</span>" not in result_template
    assert 'admission_tier || "exploratory"' not in result_template
    assert "scroll-stage" not in page


def test_method_page_and_background_asset_are_packaged() -> None:
    page = (STATIC / "method.html").read_text(encoding="utf-8")
    assert 'data-method-static' in page
    assert '<script src="method-flow.js"></script>' in page
    assert '<script src="method-static.js"></script>' in page
    assert 'method-modal.js' not in page
    assert '<link rel="stylesheet" href="method-flow.css">' in page
    assert "Read the full method" not in page
    assert "Fine-tuned multilingual authorship representation" in page
    assert page.count('class="model-comparison') == 1
    assert "method-brief" in page
    assert "method-performance-figure" in page
    assert "performance-table" not in page
    assert "Selected design:" in page
    assert "learned multi-view reranker" in page
    assert "MRR .785" in page and "representation .782" in page
    assert "The academic mechanism behind the match" not in page
    assert "classification F1" in page
    assert "Direct original vs translation-mediated retrieval" not in page
    assert "commit 15978e1" not in page
    assert "Two safeguards come from a measured failure" not in page
    assert "One non-crossing ranking path" not in page
    performance = page.split("<h2>6 · Performance record</h2>", 1)[1].split("</table>", 1)[0]
    assert 'class="num"' not in performance
    method_ui = (STATIC / "method-static.js").read_text(encoding="utf-8")
    assert "Can style travel across languages?" in method_ui
    assert 'answer.split("; ")' in method_ui
    assert 'document.createElement("br")' in method_ui
    assert "The upper route is the complete author-ranking path" not in method_ui
    assert 'element("div", "primary-flow")' in method_ui
    flow_css = (STATIC / "method-flow.css").read_text(encoding="utf-8")
    assert 'url("head.webp")' not in flow_css
    assert "font: 700 .72rem/1.5 var(--mono)" in flow_css
    background = STATIC / "head.webp"
    assert background.stat().st_size > 100_000
    header = background.read_bytes()[:16]
    assert header[:4] == b"RIFF" and header[8:12] == b"WEBP"
    css = (STATIC / "site-shell.css").read_text(encoding="utf-8")
    assert ".page-library.stylematch-atmosphere::after" in css


def test_author_library_is_generated_from_complete_registry() -> None:
    page = (STATIC / "authors.html").read_text(encoding="utf-8")
    assert 'class="site-nav"' in page
    assert 'id="author-search"' in page
    assert '<script src="authors-data.js"></script>' in page
    registry = (ROOT / "data" / "source_registry" / "all_people.csv").read_text(encoding="utf-8")
    authors = (STATIC / "authors-data.js").read_text(encoding="utf-8")
    assert "Oscar Wilde" in registry and "jewel-cut" in registry
    assert "window.STYLEMATCH_AUTHORS" in authors
    assert page.index('class="author-traits"') < page.index('class="author-profile"')
    assert 'profile.split(";")' in page
    assert "J. K. Rowling" in authors and "morally legible" in authors
    payload = json.loads(authors.split(" = ", 1)[1][:-2])
    assert len(payload) == 270
    assert all(author["profile"] and author["style_traits"] for author in payload)
    assert next(author for author in payload if author["name"] == "Hannah Arendt")["original_languages"] == ["de", "en"]


def test_fyi_page_explains_scores_without_calling_them_probabilities() -> None:
    page = (STATIC / "fyi.html").read_text(encoding="utf-8")
    assert 'class="site-nav"' in page
    assert page.count('href="fyi.html"') == 1
    assert "Why does a passage copied from the library not score 1.00?" in page
    assert "does not compare the passage with itself" in page
    assert "not a probability of authorship" in page
    assert "Style matching and source identification are different tasks" in page
