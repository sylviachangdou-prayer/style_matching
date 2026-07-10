from __future__ import annotations

from argparse import Namespace

import pandas as pd

from scripts.build_chunk_parquet_from_sources import coverage
from scripts.finetune_multilingual_style import make_pairs
from scripts.make_source_heldout_splits import split_author
from scripts.multilingual_style_index import balanced_profile_sample, profile_metadata


def _rows() -> pd.DataFrame:
    rows = []
    for corpus in ["literary", "rhetorical"]:
        for source_number in range(1, 5):
            for chunk_number in range(3):
                rows.append({
                    "chunk_id": f"{corpus}_{source_number}_{chunk_number}",
                    "corpus": corpus,
                    "language": "en",
                    "author_or_speaker": "Same Author",
                    "source_id": f"source_{source_number}",
                    "title": f"{corpus} work {source_number}",
                    "text": f"{corpus} source {source_number} chunk {chunk_number}",
                })
    return pd.DataFrame(rows)


def test_balanced_profile_sampling_merges_corpora_and_caps_sources() -> None:
    sampled = balanced_profile_sample(_rows(), per_source_cap=2, profile_cap=10, seed=7)
    assert len(sampled) == 10
    assert sampled.groupby(["language", "author_or_speaker"]).ngroups == 1
    assert set(sampled["corpus"]) == {"literary", "rhetorical"}
    assert (sampled.groupby(["corpus", "source_id"]).size() <= 2).all()


def test_index_metadata_has_one_profile_and_internal_corpus_provenance() -> None:
    profiles = profile_metadata(_rows())
    assert len(profiles) == 1
    assert profiles[0]["author_or_speaker"] == "Same Author"
    assert profiles[0]["n_sources"] == 8
    assert profiles[0]["source_corpora"] == ["literary", "rhetorical"]


def test_pair_sampling_is_bounded_and_cross_source() -> None:
    pairs = make_pairs(_rows(), pairs_per_author=11, seed=7)
    assert len(pairs) == 11
    assert all(left != right for left, right in pairs)


def test_coverage_uses_unified_author_language_profile() -> None:
    report = coverage(_rows().to_dict("records"), Namespace(
        corpus="both", language=None, min_sources=3, min_chunks=3,
    ))
    assert report["n_authors"] == 1
    assert report["n_author_language_profiles"] == 1
    assert report["people"][0]["source_count"] == 8
    assert report["people"][0]["source_corpora"] == ["literary", "rhetorical"]


def test_source_heldout_never_reuses_a_source_across_splits() -> None:
    split, report = split_author(_rows(), seed=9, args=Namespace(
        train_cap=300, dev_cap=50, test_cap=50, min_train=1, min_dev=1, min_test=1,
    ))
    assert report["eligible"]
    assert split is not None
    source_split = split.assign(source_key=split["corpus"] + "::" + split["source_id"])
    assert source_split.groupby("source_key")["split"].nunique().max() == 1
