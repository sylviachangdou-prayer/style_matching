from __future__ import annotations

import numpy as np
from sklearn.metrics import brier_score_loss, roc_auc_score, roc_curve


def ranks_from_scores(scores: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    order = np.argsort(scores, axis=1)[:, ::-1]
    return np.asarray([
        int(np.flatnonzero(row == true)[0]) + 1 for row, true in zip(order, y_true)
    ])


def ranking_metrics(scores: np.ndarray, y_true: np.ndarray) -> dict[str, float]:
    ranks = ranks_from_scores(scores, y_true)
    return {
        "recall_at_1": float(np.mean(ranks <= 1)),
        "recall_at_3": float(np.mean(ranks <= 3)),
        "recall_at_5": float(np.mean(ranks <= 5)),
        "recall_at_20": float(np.mean(ranks <= 20)),
        "mrr": float(np.mean(1.0 / ranks)),
    }


def expected_calibration_error(labels: np.ndarray, probabilities: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = max(len(labels), 1)
    error = 0.0
    for left, right in zip(edges[:-1], edges[1:]):
        mask = (probabilities >= left) & (probabilities < right if right < 1 else probabilities <= right)
        if mask.any():
            error += mask.sum() / total * abs(float(labels[mask].mean()) - float(probabilities[mask].mean()))
    return float(error)


def equal_error_rate(labels: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    fpr, tpr, thresholds = roc_curve(labels, scores)
    fnr = 1.0 - tpr
    index = int(np.argmin(np.abs(fpr - fnr)))
    return float((fpr[index] + fnr[index]) / 2.0), float(thresholds[index])


def open_set_metrics(labels: np.ndarray, scores: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    eer, threshold = equal_error_rate(labels, scores)
    return {
        "auroc": float(roc_auc_score(labels, scores)),
        "equal_error_rate": eer,
        "equal_error_threshold": threshold,
        "brier": float(brier_score_loss(labels, probabilities)),
        "ece": expected_calibration_error(labels, probabilities),
    }


def misattribution_unfairness(scores: np.ndarray, y_true: np.ndarray, k: int = 3) -> dict[str, object]:
    topk = np.argsort(scores, axis=1)[:, ::-1][:, :k]
    false_counts = np.zeros(scores.shape[1], dtype=int)
    opportunities = np.zeros(scores.shape[1], dtype=int)
    for candidates, true in zip(topk, y_true):
        for author in range(scores.shape[1]):
            if author != true:
                opportunities[author] += 1
        for candidate in candidates:
            if candidate != true:
                false_counts[candidate] += 1
    rates = false_counts / np.maximum(opportunities, 1)
    return {
        "k": k,
        "mean_false_topk_rate": float(rates.mean()),
        "max_false_topk_rate": float(rates.max()),
        "per_candidate_rate": rates.tolist(),
    }


def paired_bootstrap_mrr(
    baseline_scores: np.ndarray,
    candidate_scores: np.ndarray,
    y_true: np.ndarray,
    runs: int = 2000,
    seed: int = 20260711,
) -> dict[str, float]:
    baseline = 1.0 / ranks_from_scores(baseline_scores, y_true)
    candidate = 1.0 / ranks_from_scores(candidate_scores, y_true)
    delta = candidate - baseline
    rng = np.random.default_rng(seed)
    means = np.asarray([
        delta[rng.integers(0, len(delta), len(delta))].mean() for _ in range(runs)
    ])
    return {
        "mrr_delta": float(delta.mean()),
        "ci_low": float(np.quantile(means, 0.025)),
        "ci_high": float(np.quantile(means, 0.975)),
    }
