import numpy as np

from scripts.search_postwhitening_calibrators import (
    blend_scores,
    candidate_calibration,
    source_balanced_maui,
    structural_scores,
)


def test_candidate_calibration_downweights_a_high_baseline_candidate():
    fit_scores = np.array([
        [0.9, 0.80], [0.8, 0.81], [0.5, 0.90],
        [0.6, 0.85], [0.7, 0.82], [0.65, 0.88],
    ])
    labels = np.array([0, 0, 1, 1, 0, 1])
    languages = np.array(["en"] * 6)
    corpora = np.array(["literary"] * 6)
    query_scores = np.array([[0.75, 0.805]])
    percentiles, robust_z = candidate_calibration(
        fit_scores, labels, languages, corpora,
        query_scores, np.array(["en"]), np.array(["literary"]),
        np.array(["en", "en"]),
    )
    assert percentiles[0, 0] > percentiles[0, 1]
    assert robust_z[0, 0] > robust_z[0, 1]


def test_structural_penalty_exempts_reverse_supported_anchor():
    baseline = np.array([[0.9, 0.8, 0.7]])
    reverse = np.array([[0.99, 0.5, 0.5]])
    popularity = np.array([0.99, 0.95, 0.1])
    result = structural_scores(
        baseline, reverse, np.array(["en"]), np.array(["en"] * 3), popularity,
        anchor_threshold=0.98, popularity_threshold=0.9,
        penalty=0.1, bonus=0.0, anchor_rank=3, shortlist=3,
    )
    assert result[0, 0] == baseline[0, 0]
    assert result[0, 1] < baseline[0, 1]


def test_blend_preserves_language_mask_at_alpha_one():
    first = np.array([[1.0, -np.inf]])
    second = np.array([[2.0, -np.inf]])
    result = blend_scores(
        first, second, 1.0, np.array(["en"]), np.array(["en", "fr"])
    )
    assert result[0, 0] == 2.0
    assert np.isneginf(result[0, 1])


def test_source_balanced_maui_is_lower_for_even_false_exposure():
    labels = np.array([0, 1, 2, 3, 0, 1, 2, 3])
    languages = np.array(["en"] * len(labels))
    profile_languages = np.array(["en"] * 4)
    groups = np.array([f"s{index}" for index in range(len(labels))])
    concentrated = np.zeros((len(labels), 4))
    even = np.zeros_like(concentrated)
    for row, label in enumerate(labels):
        concentrated[row] = np.array([4.0, 3.0, 2.0, 1.0])
        even[row] = np.roll(np.array([4.0, 3.0, 2.0, 1.0]), row % 4)
        concentrated[row, label] = 5.0
        even[row, label] = 5.0
    first = source_balanced_maui(
        concentrated, labels, languages, profile_languages, groups, top_k=2
    )
    second = source_balanced_maui(
        even, labels, languages, profile_languages, groups, top_k=2
    )
    assert second["source_balanced_maui_at_3"] < first["source_balanced_maui_at_3"]
