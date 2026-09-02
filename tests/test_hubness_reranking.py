from __future__ import annotations

import numpy as np

from scripts.evaluate_hubness_reranking import (
    empirical_percentiles,
    exposure,
    mask_languages,
)


def test_empirical_null_downweights_broad_candidate() -> None:
    fit = np.asarray([
        [0.80, 0.30],
        [0.79, 0.31],
        [0.78, 0.32],
        [0.77, 0.33],
        [0.76, 0.34],
        [0.75, 0.35],
    ])
    labels = np.asarray([1, 1, 1, 0, 0, 0])
    languages = np.asarray(["en"] * 6)
    corpora = np.asarray(["literary"] * 6)
    calibrated = empirical_percentiles(
        fit, labels, languages, corpora,
        np.asarray([[0.79, 0.70]]), np.asarray(["en"]),
        np.asarray(["literary"]), np.asarray(["en", "en"]),
    )
    assert calibrated[0, 1] > calibrated[0, 0]


def test_language_mask_and_exposure() -> None:
    scores = mask_languages(
        np.asarray([[0.9, 0.8], [0.7, 0.6]]),
        np.asarray(["en", "fr"]), np.asarray(["en", "fr"]),
    )
    assert np.isneginf(scores[0, 1])
    summary, table = exposure(scores, np.asarray([0, 1]), np.asarray(["en::A", "fr::B"]))
    assert summary["maximum_false_top3_share"] == 0.0
    assert table["false_top3_count"].sum() == 0
