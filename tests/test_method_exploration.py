from __future__ import annotations

import json
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.evaluate_matched_difficulty import chance_adjust, evaluate_sample
from scripts.evaluate_evidence_gated_reranker import grouped_folds
from scripts.evaluate_multiview_fusion import (
    fit_fusion,
    make_view_features,
    predict_fusion,
    profile_bootstrap_intervals,
)
from scripts.evaluate_robust_rank_fusion import simplex_weights
from scripts.discover_gutenberg_authors import display_name, estimated_birth_year
from scripts.fetch_gutendex import author_match, existing_work_counts, independent_title_key
from scripts.evaluate_multiview_open_set import confidence_features
from scripts.finetune_multilingual_style import pcm_mask_examples
from scripts.score_artifact_utils import (
    aggregate_scores_by_source,
    independent_source_keys,
    normalized_score_features,
)
from scripts.style_syntax_baseline import parse_syntax


def test_source_aggregation_uses_independent_sources() -> None:
    frame = pd.DataFrame(
        {
            "chunk_id": ["c1", "c2", "c3"],
            "author_or_speaker": ["A", "A", "B"],
            "language": ["en", "en", "en"],
            "corpus": ["literary"] * 3,
            "split": ["test"] * 3,
            "source_id": ["web1", "web2", "web3"],
            "independent_source_id": ["work1", "work1", "work2"],
        }
    )
    labels = np.asarray([0, 0, 1])
    scores = {"view": np.asarray([[0.8, 0.2], [0.6, 0.4], [0.1, 0.9]])}
    source_frame, source_labels, source_scores = aggregate_scores_by_source(frame, labels, scores)
    assert len(source_frame) == 2
    assert source_labels.tolist() == [0, 1]
    np.testing.assert_allclose(source_scores["view"][0], [0.7, 0.3])
    assert independent_source_keys(frame).nunique() == 2


def test_profile_bootstrap_reports_exact_point_estimate() -> None:
    scores = np.eye(4)
    labels = np.arange(4)
    intervals = profile_bootstrap_intervals(scores, labels, runs=20, seed=3)
    assert intervals["mrr"]["estimate"] == 1.0
    assert intervals["recall_at_3"]["ci_low"] == 1.0


def test_multilingual_gutendex_matching_and_work_identity() -> None:
    book = {"authors": [{"name": "Gautier, Théophile"}]}
    assert author_match(book, "Theophile Gautier")
    assert independent_title_key("Example, Volume II") == independent_title_key("Example, Volume I")
    assert display_name("Gautier, Théophile") == "Théophile Gautier"
    assert estimated_birth_year({"birth_year": 1811, "death_year": 1872}) == 1811
    assert estimated_birth_year({"birth_year": None, "death_year": 1600}) == 1530
    assert existing_work_counts([
        {"author_or_speaker": "A", "independent_source_id": "work-1"},
        {"author_or_speaker": "A", "independent_source_id": "work-1"},
        {"author_or_speaker": "A", "source_id": "work-2"},
    ]) == {"A": {"work-1", "work-2"}}


def test_matched_candidate_metrics_include_theoretical_chance() -> None:
    scores = np.eye(4)
    labels = np.arange(4)
    result = evaluate_sample(scores, labels, np.arange(4))
    assert result["observed.mrr"] == 1.0
    expected = chance_adjust({
        "recall_at_1": 1.0,
        "recall_at_3": 1.0,
        "recall_at_5": 1.0,
        "mrr": 1.0,
    }, 4)
    assert result["chance.mrr"] == expected["chance"]["mrr"]
    assert result["chance_adjusted.mrr"] == 1.0


def test_normalization_never_compares_languages() -> None:
    profiles = np.asarray(["en::A", "en::B", "de::C", "de::D"])
    scores = np.asarray([[2.0, 1.0, 100.0, 99.0], [100.0, 99.0, 4.0, 1.0]])
    z, percentile = normalized_score_features(scores, np.asarray(["en", "de"]), profiles)
    assert z[0, 0] > z[0, 1]
    assert z[0, 2] == -8.0
    assert percentile[1, 2] == 1.0
    assert percentile[1, 0] == 0.0


def test_fusion_trains_only_same_language_candidate_pairs() -> None:
    profiles = np.asarray(["en::A", "en::B", "de::C", "de::D"])
    languages = np.asarray(["en", "en", "de", "de"])
    labels = np.asarray([0, 1, 2, 3])
    view1 = np.asarray(
        [[0.9, 0.1, -1e9, -1e9], [0.2, 0.8, -1e9, -1e9], [-1e9, -1e9, 0.8, 0.2], [-1e9, -1e9, 0.1, 0.9]]
    )
    view2 = view1 * 0.5
    feature_names, features = make_view_features(
        {"one": view1, "two": view2}, languages, profiles
    )
    rows = np.arange(4)
    model = fit_fusion(rows, feature_names, features, languages, profiles, labels, 1.0, 0.25, 1)
    fused = predict_fusion(model, rows, feature_names, features, languages, profiles)
    assert fused.argmax(axis=1).tolist() == labels.tolist()
    assert np.all(fused[0, 2:] == -1e9)


def test_robust_fusion_weights_are_convex_and_base_anchored() -> None:
    candidates = simplex_weights(["base", "classical", "pretrained"], "base", 0.1, 0.5)
    assert candidates
    assert all(np.isclose(sum(row.values()), 1.0) for row in candidates)
    assert all(row["base"] >= 0.5 for row in candidates)
    assert {"base": 1.0, "classical": 0.0, "pretrained": 0.0} in candidates


def test_neural_reranker_folds_keep_profiles_together() -> None:
    labels = np.asarray([0, 0, 1, 1, 2, 2, 3, 3])
    folds = grouped_folds(labels, folds=3, seed=4)
    assert sorted(np.concatenate(folds).tolist()) == list(range(len(labels)))
    for label in np.unique(labels):
        assert sum(bool(np.isin(label, labels[fold]).any()) for fold in folds) == 1


def test_open_set_features_include_cross_view_agreement() -> None:
    matrices = {
        "one": np.asarray([[0.9, 0.1, 0.0], [0.1, 0.8, 0.2]]),
        "two": np.asarray([[0.8, 0.2, 0.0], [0.7, 0.2, 0.1]]),
    }
    features, names, predictions = confidence_features(matrices, np.arange(2), np.arange(3))
    assert features.shape == (2, 11)
    assert names[-1] == "cross_view.top1_agreement"
    assert features[0, -1] == 1.0
    assert features[1, -1] == 0.5
    assert predictions.shape == (2, 2)


class FakeTokenizer:
    mask_token_id = 99

    def __call__(self, texts, **kwargs):
        return {
            "input_ids": [[0] + [int(token) for token in text.split()] + [1] for text in texts],
            "special_tokens_mask": [[1] + [0] * len(text.split()) + [1] for text in texts],
        }

    def convert_ids_to_tokens(self, token_id):
        return "<mask>" if token_id == 99 else str(token_id)

    def convert_tokens_to_string(self, tokens):
        return " ".join(tokens)


def test_pcm_protects_frequent_tokens_and_masks_content() -> None:
    masked = pcm_mask_examples(
        {"en": [("2 3", "2 4")]},
        FakeTokenizer(),
        protected_token_ids={2},
        mask_prob=0.999,
        max_length=10,
        seed=1,
    )
    assert masked["en"][0] == ("2 <mask>", "2 <mask>")


def test_syntax_parser_checkpoints_without_real_stanza(tmp_path: Path, monkeypatch) -> None:
    class Word:
        def __init__(self, word_id, upos, deprel, head):
            self.id = word_id
            self.upos = upos
            self.deprel = deprel
            self.head = head

    document = types.SimpleNamespace(
        sentences=[
            types.SimpleNamespace(
                words=[Word(1, "PRON", "nsubj", 2), Word(2, "VERB", "root", 0)]
            )
        ]
    )

    class Pipeline:
        def __init__(self, **kwargs):
            pass

        def __call__(self, text):
            return document

    fake_stanza = types.SimpleNamespace(Pipeline=Pipeline, download=lambda *args, **kwargs: None)
    monkeypatch.setitem(sys.modules, "stanza", fake_stanza)
    frame = pd.DataFrame(
        {
            "chunk_id": ["one", "two"],
            "language": ["en", "en"],
            "text": ["We write.", "They write."],
        }
    )
    cache_path = tmp_path / "syntax.parquet"
    parsed = parse_syntax(frame, cache_path, "cpu", False, 1)
    assert cache_path.exists()
    assert parsed["chunk_id"].tolist() == ["one", "two"]
    assert parsed["syntax_stream"].str.contains("D_PRON_nsubj_VERB_R").all()


def test_source_level_experiment_clis(tmp_path: Path) -> None:
    profiles = np.asarray([f"en::A{index:02d}" for index in range(12)])
    rows = []
    labels = []
    chunk_ids = []
    splits = []
    for split in ("dev", "test"):
        for label, profile in enumerate(profiles):
            author = profile.split("::", 1)[1]
            for chunk in range(2):
                chunk_id = f"{split}-{label}-{chunk}"
                rows.append(
                    {
                        "chunk_id": chunk_id,
                        "author_or_speaker": author,
                        "language": "en",
                        "corpus": "literary" if label % 2 else "rhetorical",
                        "split": split,
                        "source_id": f"{split}-{label}",
                        "independent_source_id": f"{split}-{label}",
                        "text": "synthetic diagnostic text",
                    }
                )
                labels.append(label)
                chunk_ids.append(chunk_id)
                splits.append(split)
    frame = pd.DataFrame(rows)
    input_path = tmp_path / "heldout.parquet"
    frame.to_parquet(input_path, index=False)
    labels_array = np.asarray(labels)
    view1 = np.full((len(frame), len(profiles)), 0.1, dtype="float32")
    view2 = np.full_like(view1, 0.2)
    view3 = np.full_like(view1, 0.15)
    view1[np.arange(len(frame)), labels_array] = 0.9
    view2[np.arange(len(frame)), labels_array] = 0.8
    view3[np.arange(len(frame)), labels_array] = 0.85
    artifact = tmp_path / "scores.npz"
    np.savez_compressed(
        artifact,
        chunk_ids=np.asarray(chunk_ids),
        splits=np.asarray(splits),
        query_languages=np.asarray(["en"] * len(frame)),
        query_corpora=frame["corpus"].astype(str).to_numpy(),
        profiles=profiles,
        y_true=labels_array,
        view1=view1,
        view2=view2,
        view3=view3,
    )
    root = Path(__file__).resolve().parents[1]
    difficulty_output = tmp_path / "difficulty.json"
    subprocess.run(
        [
            sys.executable,
            "scripts/evaluate_matched_difficulty.py",
            "--input",
            str(input_path),
            "--scores",
            f"one={artifact}:view1",
            "--output",
            str(difficulty_output),
            "--candidate-sizes",
            "5,all",
            "--repeats",
            "3",
        ],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(difficulty_output.read_text())["models"]["one"]["en"][
        "candidate_sizes"
    ]["12"]["observed"]["mrr"]["mean"] == 1.0
    fusion_dir = tmp_path / "fusion"
    subprocess.run(
        [
            sys.executable,
            "scripts/evaluate_multiview_fusion.py",
            "--input",
            str(input_path),
            "--scores",
            f"one={artifact}:view1",
            "--scores",
            f"two={artifact}:view2",
            "--output-dir",
            str(fusion_dir),
            "--bootstrap-runs",
            "20",
        ],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    assert (fusion_dir / "multiview_fusion_metrics.json").exists()
    neural_dir = tmp_path / "neural"
    subprocess.run(
        [
            sys.executable,
            "scripts/evaluate_evidence_gated_reranker.py",
            "--input",
            str(input_path),
            "--scores",
            f"anchor={artifact}:view1",
            "--scores",
            f"support={artifact}:view2",
            "--scores",
            f"prototype={artifact}:view3",
            "--anchor",
            "anchor",
            "--output-dir",
            str(neural_dir),
            "--folds",
            "3",
            "--epochs",
            "8",
            "--patience",
            "3",
            "--bootstrap-runs",
            "20",
            "--device",
            "cpu",
        ],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    neural_report = json.loads(
        (neural_dir / "evidence_gated_reranker_metrics.json").read_text()
    )
    assert neural_report["anchor"] == "anchor"
    assert neural_report["protocol"]["reinforcement_learning_used"] is False
    assert (neural_dir / "evidence_gated_reranker.pt").exists()
    robust_dir = tmp_path / "robust"
    subprocess.run(
        [
            sys.executable,
            "scripts/evaluate_robust_rank_fusion.py",
            "--input",
            str(input_path),
            "--scores",
            f"base={artifact}:view1",
            "--scores",
            f"support={artifact}:view2",
            "--base",
            "base",
            "--output-dir",
            str(robust_dir),
            "--minimum-group-sources",
            "2",
            "--bootstrap-runs",
            "20",
        ],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    report = json.loads((robust_dir / "robust_reranker_metrics.json").read_text())
    assert report["decision"] == "base"
    assert (robust_dir / "robust_reranker_candidates.csv").exists()
    open_dir = tmp_path / "open"
    subprocess.run(
        [
            sys.executable,
            "scripts/evaluate_multiview_open_set.py",
            "--input",
            str(input_path),
            "--scores",
            f"one={artifact}:view1",
            "--scores",
            f"two={artifact}:view2",
            "--language",
            "en",
            "--output-dir",
            str(open_dir),
        ],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads((open_dir / "open_set_metrics.json").read_text())["n_candidate_profiles"] == 10
