from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from scripts import discover_listed_gutenberg_authors as discover


ROOT = Path(__file__).resolve().parents[1]


def book(
    identifier: int,
    title: str,
    *,
    language: str = "en",
    translators: list | None = None,
    authors: list | None = None,
) -> dict:
    return {
        "id": identifier,
        "title": title,
        "copyright": False,
        "languages": [language],
        "translators": translators or [],
        "authors": authors or [{"name": "Bennett, Arnold"}],
        "formats": {"text/plain; charset=utf-8": f"https://example.org/{identifier}.txt"},
        "download_count": identifier,
    }


def test_targeted_verification_enforces_source_admission(monkeypatch) -> None:
    payload = {
        "results": [
            book(1, "Independent novel"),
            book(2, "Independent novel, Volume II"),
            book(3, "A translated work", translators=[{"name": "Translator"}]),
            book(4, "A collaboration", authors=[
                {"name": "Bennett, Arnold"}, {"name": "Other, Author"}
            ]),
            book(5, "Another novel"),
        ],
        "next": None,
    }
    monkeypatch.setattr(discover, "fetch_json", lambda _: payload)
    works = discover.verified_works("Arnold Bennett", "Bennett Arnold", "en", 2)
    assert [work["id"] for work in works] == [5, 1]


def test_target_list_is_bounded_multilingual_and_reviewed_additions_are_registered() -> None:
    targets = pd.read_csv(
        ROOT / "data/source_registry/gutenberg_target_authors_2026_07.csv"
    )
    registry = pd.read_csv(ROOT / "data/source_registry/all_people.csv")
    assert 50 <= len(targets) <= 100
    assert targets["original_language"].nunique() >= 5
    assert not targets.duplicated(["name", "original_language"]).any()
    assert targets["min_independent_sources"].eq(3).all()
    additions = pd.read_csv(
        ROOT / "data/source_registry/gutenberg_indexed_metadata_2026_07.csv"
    )
    assert len(additions) == additions["name"].nunique() == 88
    registered = additions.merge(
        registry[["name", "original_language"]],
        on=["name", "original_language"],
    )
    assert len(registered) == 88
    assert additions["profile"].str.count(";").eq(2).all()
    assert additions["style_traits"].str.split(",").map(len).between(3, 5).all()
    assert not additions["profile"].duplicated().any()
    assert not additions["style_traits"].duplicated().any()


def test_part8_keeps_targeted_expansion_and_hubness_gate() -> None:
    notebook = json.loads(
        (ROOT / "ecore_pt8.ipynb").read_text(encoding="utf-8")
    )
    source = "\n".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"]
    )
    assert "discover_listed_gutenberg_authors.py" in source
    assert "discover_gutenberg_authors.py" not in source
    assert "evaluate_hubness_correction.py" in source
    assert "attach_hubness_correction.py" in source
    assert "source_heldout_splits.parquet" in source
    assert "style_embedding_recall.py" in source
    assert "merge_author_registry_metadata.py" in source
    assert "indexed_missing_registry_metadata" in source
    assert "compare_index_retrieval.py" in source
    assert "old_vs_new_index_metrics.json" in source
