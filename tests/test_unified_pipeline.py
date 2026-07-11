from __future__ import annotations

from argparse import Namespace
import json

import numpy as np
import pandas as pd

from scripts.build_chunk_parquet_from_sources import coverage
from scripts.audit_release_readiness import coverage_matrix
from scripts.audit_source_metadata import audit as audit_source_metadata
from scripts.finetune_multilingual_style import make_hard_negative_examples, make_pairs, training_coverage
from scripts.make_group_heldout_splits import build_split
from scripts.make_source_heldout_splits import split_author
from scripts.multilingual_style_index import StyleIndex, balanced_profile_sample, profile_metadata
from scripts.retrieval_metrics import paired_bootstrap_mrr, ranking_metrics
from scripts.style_embedding_recall import mask_cross_language_candidates
from scripts.style_robust_baseline import compression_distance_scores


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


def test_query_result_is_json_serializable_when_parquet_returns_array() -> None:
    class FakeModel:
        def encode(self, *args, **kwargs):
            return np.asarray([[1.0, 0.0]], dtype="float32")

    index = StyleIndex.__new__(StyleIndex)
    index.metadata = {"style_weight_within": 0.7}
    index.profiles = pd.DataFrame([{
        "profile_id": 0,
        "language": "en",
        "author_or_speaker": "Test Author",
        "source_corpora": np.asarray(["literary", "rhetorical"]),
        "n_sources": 2,
    }])
    index.passages = pd.DataFrame([{
        "profile_id": 0,
        "title": "Test Work",
        "source_id": "test_source",
        "text": "Test passage.",
    }])
    index.centroids = np.asarray([[1.0, 0.0]], dtype="float32")
    index.topic_centroids = None
    index.topic_model = None
    index.decade_centroids = np.asarray([[1.0, 0.0]], dtype="float32")
    index.decades = pd.DataFrame([{
        "decade_id": 0,
        "language": "en",
        "corpus": "literary",
        "decade": "1920s",
        "n_authors": 5,
        "n_sources": 2,
        "n_chunks": 6,
    }])
    index.model = FakeModel()

    result = index.query("Test text.", "en", "within", 1)

    assert result["results"]["en"][0]["source_corpora"] == ["literary", "rhetorical"]
    assert result["decade_match"]["decade"] == "1920s"
    json.dumps(result)


def test_real_mini_index_artifact_loads_without_demo(monkeypatch, tmp_path) -> None:
    class FakeModel:
        def encode(self, *args, **kwargs):
            return np.asarray([[1.0, 0.0]], dtype="float32")

    metadata = {
        "model_name": "fake",
        "backend": None,
        "topic_model_name": None,
        "n_decade_profiles": 0,
        "profile_strategy": "source_prototype_topk_mean",
        "prototype_top_k": 1,
        "style_weight_within": 1.0,
        "score_version": "test_v1",
        "artifact_version": "test_artifact",
    }
    (tmp_path / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    pd.DataFrame([{
        "profile_id": 0, "language": "en", "author_or_speaker": "A", "source_corpora": ["literary"],
        "n_sources": 1, "profile": "Author A", "style_traits": "spare, lucid, exact", "photo_url": "",
    }]).to_parquet(tmp_path / "profiles.parquet", index=False)
    pd.DataFrame([{
        "profile_id": 0, "corpus": "literary", "source_id": "s1", "title": "Work", "text": "Passage",
        "centroid_similarity": 1.0,
    }]).to_parquet(tmp_path / "representative_passages.parquet", index=False)
    pd.DataFrame([{
        "prototype_id": 0, "profile_id": 0, "language": "en", "author_or_speaker": "A",
        "corpus": "literary", "source_id": "s1", "n_chunks": 1,
    }]).to_parquet(tmp_path / "source_prototypes.parquet", index=False)
    np.save(tmp_path / "centroids.npy", np.asarray([[1.0, 0.0]], dtype="float32"))
    np.save(tmp_path / "source_prototype_centroids.npy", np.asarray([[1.0, 0.0]], dtype="float32"))
    monkeypatch.setattr("scripts.multilingual_style_index.load_model", lambda *args, **kwargs: FakeModel())

    result = StyleIndex(tmp_path, device="cpu").query("text", "en", "within", 1)

    assert result["artifact_version"] == "test_artifact"
    assert result["profile_strategy"] == "source_prototype_topk_mean"
    assert result["decade_status"] == "unavailable_not_validated"
    assert result["results"]["en"][0]["profile"] == "Author A"


def test_pair_sampling_is_bounded_and_cross_source() -> None:
    pairs = make_pairs(_rows(), pairs_per_author=11, seed=7)
    assert len(pairs) == 11
    assert all(left != right for left, right in pairs)


def test_hard_negative_examples_use_a_different_author() -> None:
    frame = pd.concat([
        _rows().assign(
            author_or_speaker="Author A", topic="shared", decade="1920s",
            text=lambda value: "A " + value["text"],
        ),
        _rows().assign(
            author_or_speaker="Author B", topic="shared", decade="1920s",
            text=lambda value: "B " + value["text"],
        ),
    ], ignore_index=True)
    examples, relaxed = make_hard_negative_examples(frame, pairs_per_author=3, seed=4)
    assert relaxed == 0
    assert len(examples["en"]) == 6
    assert all(len(example) == 3 and example[0] != example[2] for example in examples["en"])


def test_training_coverage_distinguishes_registry_from_eligible_profiles(tmp_path) -> None:
    registry = tmp_path / "registry.csv"
    registry.write_text(
        "name,original_language\nSame Author,en\nMissing Author,fr\n",
        encoding="utf-8",
    )
    report = training_coverage(_rows(), registry)
    assert report["n_registry_author_language_profiles"] == 2
    assert report["n_profiles_with_chunks"] == 1
    assert report["n_profiles_eligible_for_finetuning"] == 1
    assert report["n_profiles_with_at_least_3_sources"] == 1
    assert report["registry_profiles_without_chunks"] == [
        {"language": "fr", "author_or_speaker": "Missing Author"}
    ]


def test_coverage_matrix_assigns_formal_exploratory_and_catalog_tiers(tmp_path) -> None:
    registry = tmp_path / "registry.csv"
    registry.write_text(
        "name,original_language,corpus\nFormal,en,literary\nExplore,en,literary\nMissing,fr,literary\n",
        encoding="utf-8",
    )
    chunks = pd.DataFrame([
        {"language": "en", "author_or_speaker": "Formal", "corpus": "literary", "source_id": "a"},
        {"language": "en", "author_or_speaker": "Explore", "corpus": "literary", "source_id": "b"},
    ])
    heldout = {"authors": [
        {"language": "en", "author": "Formal", "eligible": True, "reason": "ok"},
        {"language": "en", "author": "Explore", "eligible": False, "reason": "fewer_than_3_sources"},
    ]}
    matrix = coverage_matrix(registry, chunks, heldout).set_index("author_or_speaker")
    assert matrix.loc["Formal", "admission_tier"] == "formal"
    assert matrix.loc["Explore", "admission_tier"] == "exploratory"
    assert matrix.loc["Missing", "admission_tier"] == "catalog_only"


def test_global_group_split_never_reuses_group_across_splits() -> None:
    rows = []
    for author in ("A", "B"):
        for topic in [f"topic{i}" for i in range(12)]:
            for chunk in range(3):
                rows.append({
                    "language": "en",
                    "author_or_speaker": author,
                    "topic": topic,
                    "text": f"{author} {topic} {chunk}",
                })
    output, report = build_split(
        pd.DataFrame(rows), "topic", 7, {"train": 3, "dev": 3, "test": 3}
    )
    assert not output.empty
    assert not report["global_group_leakage"]
    assert output.groupby("topic")["split"].nunique().max() == 1


def test_source_metadata_audit_rejects_unlicensed_display(tmp_path) -> None:
    path = tmp_path / "sources.csv"
    fields = [
        "corpus", "author_or_speaker", "title", "source_id", "independent_source_id", "language", "year", "topic",
        "domain", "register", "source_type", "delivered_language", "license_status",
        "display_allowed", "canonical_url", "raw_text_path",
    ]
    path.write_text(
        ",".join(fields) + "\n" +
        "literary,A,Work,s1,1920_work,en,1920,city,literature,prose,work,en,unknown,true,https://x,raw/a.txt\n",
        encoding="utf-8",
    )
    report = audit_source_metadata([path])
    assert not report["complete"]
    assert report["invalid_display_sources"][0]["source_id"] == "s1"


def test_source_metadata_audit_rejects_missing_files(tmp_path) -> None:
    missing = tmp_path / "missing.csv"
    report = audit_source_metadata([missing])
    assert not report["complete"]
    assert report["n_sources"] == 0
    assert report["missing_files"] == [str(missing)]


def test_ranking_metrics_and_bootstrap_use_query_as_unit() -> None:
    labels = np.asarray([0, 1, 2, 0])
    baseline = np.asarray([[.6, .3, .1], [.4, .5, .1], [.4, .3, .3], [.4, .5, .1]])
    candidate = np.asarray([[.8, .1, .1], [.1, .8, .1], [.1, .1, .8], [.8, .1, .1]])
    assert ranking_metrics(candidate, labels)["mrr"] == 1.0
    assert paired_bootstrap_mrr(baseline, candidate, labels, runs=100, seed=3)["mrr_delta"] > 0


def test_within_language_scores_mask_other_language_profiles() -> None:
    scores = np.asarray([[0.4, 0.9], [0.8, 0.3]])
    masked = mask_cross_language_candidates(
        scores,
        pd.Series(["en", "fr"]),
        np.asarray(["en::Author", "fr::Auteur"]),
    )
    assert masked[0, 1] < -1e8
    assert masked[1, 0] < -1e8


def test_compression_distance_view_returns_profile_scores() -> None:
    train = pd.DataFrame({"text": ["alpha alpha alpha", "omega omega omega"]})
    query = pd.DataFrame({"text": ["alpha alpha"]})
    scores = compression_distance_scores(train, query, np.asarray([0, 1]), 2)
    assert scores.shape == (1, 2)
    assert np.isfinite(scores).all()


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


def test_source_heldout_deduplicates_versions_of_the_same_work() -> None:
    rows = _rows().iloc[:12].copy()
    rows["corpus"] = "literary"
    rows["source_id"] = ["edition_a"] * 3 + ["edition_b"] * 3 + ["work_2"] * 3 + ["work_3"] * 3
    rows["independent_source_id"] = ["work_1"] * 6 + ["work_2"] * 3 + ["work_3"] * 3
    split, report = split_author(rows, seed=11, args=Namespace(
        train_cap=300, dev_cap=50, test_cap=50, min_train=1, min_dev=1, min_test=1,
    ))
    assert report["n_sources"] == 3
    assert split is not None
    work_split = split.groupby("independent_source_id")["split"].nunique()
    assert work_split.max() == 1
