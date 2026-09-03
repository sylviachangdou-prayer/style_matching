from __future__ import annotations

import numpy as np

from scripts.evaluate_similarity_backends import (
    adaptive_snorm,
    csls,
    language_scores,
    l2,
    remove_components,
)


def fixture() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(20260902)
    centers = l2(rng.normal(size=(4, 12)))
    labels = np.repeat(np.arange(4), 8)
    train = l2(centers[labels] + 0.08 * rng.normal(size=(len(labels), 12)))
    query = l2(centers + 0.08 * rng.normal(size=(4, 12)))
    return train, labels, query


def test_every_backend_returns_finite_candidate_scores() -> None:
    train, labels, query = fixture()
    specifications = [
        ("cosine", None),
        ("centered_cosine", None),
        ("all_but_top", 1),
        ("whitened_cosine", 0.3),
        ("l1", None),
        ("spearman", None),
        ("csls", 5),
        ("adaptive_snorm", 5),
        ("plda", (4, 0.3)),
        ("plda_snorm", (4, 0.3)),
        ("author_balanced_whitened_cosine", 0.1),
        ("whitened_csls", (0.1, 5)),
        ("author_balanced_whitened_csls", (0.1, 5)),
        ("cosine_whitened_blend", (0.1, 0.5)),
    ]
    for method, parameter in specifications:
        scores = language_scores(train, labels, query, method, parameter)
        assert scores.shape == (4, 4)
        assert np.isfinite(scores).all(), method


def test_density_normalizers_reduce_broad_candidate_advantage() -> None:
    fit = np.asarray([
        [0.90, 0.25], [0.88, 0.30], [0.86, 0.35],
        [0.84, 0.40], [0.82, 0.45], [0.80, 0.50],
    ])
    labels = np.asarray([1, 1, 1, 0, 0, 0])
    query = np.asarray([[0.85, 0.75]])
    assert csls(query, fit, labels, 3)[0, 1] > csls(query, fit, labels, 3)[0, 0]
    assert adaptive_snorm(query, fit, labels, 3)[0, 1] > adaptive_snorm(query, fit, labels, 3)[0, 0]


def test_component_removal_eliminates_selected_direction() -> None:
    values = np.asarray([[2.0, 1.0], [3.0, -1.0]])
    transformed = remove_components(values, np.zeros(2), np.asarray([[1.0, 0.0]]))
    assert np.allclose(transformed[:, 0], 0.0)
