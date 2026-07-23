from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.compare_retrieval_models import top1_calibration
from scripts.retrieval_metrics import ranks_from_scores, ranking_metrics
from scripts.score_artifact_utils import (
    aggregate_scores_by_source,
    aligned_metadata,
    candidate_mask,
    independent_source_keys,
    load_aligned_scores,
    normalized_score_features,
)


@dataclass(frozen=True)
class RankerConfig:
    name: str
    hidden_size: int
    dropout: float
    pairwise_weight: float
    anchor_penalty: float


CONFIGS = (
    RankerConfig("global_listwise", 0, 0.0, 0.25, 0.05),
    RankerConfig("evidence_gated_listwise", 16, 0.25, 0.25, 0.05),
    RankerConfig("evidence_gated_hard_negative", 24, 0.30, 0.75, 0.08),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Select a base-anchored neural listwise reranker by profile-grouped dev "
            "cross-validation, then evaluate it once on locked test sources."
        )
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--scores", action="append", required=True)
    parser.add_argument("--anchor", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--hard-negatives", type=int, default=8)
    parser.add_argument("--bootstrap-runs", type=int, default=5000)
    parser.add_argument("--minimum-subgroup-sources", type=int, default=5)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--seed", type=int, default=20260724)
    return parser.parse_args()


def macro_metrics(scores: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    rows = [
        ranking_metrics(scores[labels == label], labels[labels == label])
        for label in np.unique(labels)
    ]
    return {
        metric: float(np.mean([row[metric] for row in rows]))
        for metric in rows[0]
    }


def profile_bootstrap_intervals(
    scores: np.ndarray,
    labels: np.ndarray,
    runs: int,
    seed: int,
) -> dict[str, dict[str, float]]:
    profiles = np.unique(labels)
    per_profile = [
        ranking_metrics(scores[labels == profile], labels[labels == profile])
        for profile in profiles
    ]
    metrics = tuple(per_profile[0])
    values = np.asarray([[row[metric] for metric in metrics] for row in per_profile])
    rng = np.random.default_rng(seed)
    sampled = rng.integers(0, len(profiles), size=(runs, len(profiles)))
    draws = values[sampled].mean(axis=1)
    return {
        metric: {
            "estimate": float(values[:, index].mean()),
            "ci_low": float(np.quantile(draws[:, index], 0.025)),
            "ci_high": float(np.quantile(draws[:, index], 0.975)),
        }
        for index, metric in enumerate(metrics)
    }


def paired_profile_bootstrap(
    baseline: np.ndarray,
    candidate: np.ndarray,
    labels: np.ndarray,
    runs: int,
    seed: int,
) -> dict[str, float]:
    base_rr = 1.0 / ranks_from_scores(baseline, labels)
    candidate_rr = 1.0 / ranks_from_scores(candidate, labels)
    profile_delta = np.asarray([
        float((candidate_rr[labels == label] - base_rr[labels == label]).mean())
        for label in np.unique(labels)
    ])
    rng = np.random.default_rng(seed)
    sampled = rng.integers(0, len(profile_delta), size=(runs, len(profile_delta)))
    draws = profile_delta[sampled].mean(axis=1)
    return {
        "mrr_delta": float(profile_delta.mean()),
        "ci_low": float(np.quantile(draws, 0.025)),
        "ci_high": float(np.quantile(draws, 0.975)),
    }


def aggregate_source_moments(
    frame: pd.DataFrame,
    labels: np.ndarray,
    matrices: dict[str, np.ndarray],
) -> tuple[pd.DataFrame, np.ndarray, dict[str, np.ndarray], dict[str, np.ndarray]]:
    source_frame, source_labels, means = aggregate_scores_by_source(frame, labels, matrices)
    work = frame.copy().reset_index(drop=True)
    work["source_key"] = independent_source_keys(work)
    groups = list(work.groupby("source_key", sort=True, observed=True).indices.items())
    stability = {
        name: np.vstack([
            matrix[np.asarray(positions, dtype=int)].std(axis=0)
            if len(positions) > 1 else np.zeros(matrix.shape[1], dtype="float64")
            for _, positions in groups
        ])
        for name, matrix in matrices.items()
    }
    return source_frame, source_labels, means, stability


def make_features(
    means: dict[str, np.ndarray],
    stability: dict[str, np.ndarray],
    languages: np.ndarray,
    profiles: np.ndarray,
) -> tuple[np.ndarray, list[str], list[int]]:
    arrays: list[np.ndarray] = []
    names: list[str] = []
    view_z_positions: list[int] = []
    percentiles: list[np.ndarray] = []
    for view, scores in means.items():
        z_score, percentile = normalized_score_features(scores, languages, profiles)
        stability_z, _ = normalized_score_features(stability[view], languages, profiles)
        view_z_positions.append(len(arrays))
        arrays.extend([z_score, percentile, stability_z])
        names.extend([f"{view}.z", f"{view}.percentile", f"{view}.boundary_stability_z"])
        percentiles.append(percentile)
    percentile_stack = np.stack(percentiles, axis=-1)
    arrays.extend([
        percentile_stack.mean(axis=-1),
        percentile_stack.std(axis=-1),
        percentile_stack.min(axis=-1),
        percentile_stack.max(axis=-1),
    ])
    names.extend([
        "views.percentile_mean",
        "views.percentile_disagreement",
        "views.percentile_min",
        "views.percentile_max",
    ])
    return np.stack(arrays, axis=-1), names, view_z_positions


def candidate_masks(languages: np.ndarray, profiles: np.ndarray) -> np.ndarray:
    return np.vstack([candidate_mask(profiles, language) for language in languages])


def grouped_folds(labels: np.ndarray, folds: int, seed: int) -> list[np.ndarray]:
    profiles = np.unique(labels)
    if len(profiles) < 2:
        raise ValueError("At least two dev profiles are required for grouped selection")
    folds = min(max(2, folds), len(profiles))
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(profiles)
    buckets = [shuffled[index::folds] for index in range(folds)]
    return [np.flatnonzero(np.isin(labels, bucket)) for bucket in buckets]


class EvidenceGatedRanker(nn.Module):
    def __init__(
        self,
        n_features: int,
        n_views: int,
        view_z_positions: list[int],
        anchor_view: int,
        hidden_size: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.view_z_positions = view_z_positions
        self.anchor_view = anchor_view
        self.global_gate_logits = nn.Parameter(torch.zeros(n_views))
        self.blend_logit = nn.Parameter(torch.tensor(-1.4))
        if hidden_size:
            self.gate = nn.Sequential(
                nn.Linear(n_features, hidden_size),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size, n_views),
            )
            self.interaction = nn.Sequential(
                nn.Linear(n_features, hidden_size),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size, 1),
            )
        else:
            self.gate = None
            self.interaction = None

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        view_scores = features[..., self.view_z_positions]
        if self.gate is None:
            gate_logits = self.global_gate_logits.view(1, 1, -1).expand_as(view_scores)
            interaction = torch.zeros_like(view_scores[..., 0])
        else:
            gate_logits = self.gate(features) + self.global_gate_logits
            interaction = 0.20 * torch.tanh(self.interaction(features).squeeze(-1))
        gates = torch.softmax(gate_logits, dim=-1)
        mixture = (gates * view_scores).sum(dim=-1)
        anchor = view_scores[..., self.anchor_view]
        blend = 0.50 * torch.sigmoid(self.blend_logit)
        score = anchor + blend * (mixture - anchor) + interaction
        return score, gates


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ranking_loss(
    logits: torch.Tensor,
    masks: torch.Tensor,
    labels: torch.Tensor,
    anchor: torch.Tensor,
    pairwise_weight: float,
    anchor_penalty: float,
    hard_negatives: int,
) -> torch.Tensor:
    masked_logits = logits.masked_fill(~masks, -1e9)
    listwise = F.cross_entropy(masked_logits, labels)
    rows = torch.arange(len(labels), device=labels.device)
    positive = masked_logits[rows, labels]
    anchor_candidates = anchor.masked_fill(~masks, -1e9).clone()
    anchor_candidates[rows, labels] = -1e9
    k = min(hard_negatives, max(1, int(masks.sum(dim=1).min().item()) - 1))
    negative_indices = torch.topk(anchor_candidates, k=k, dim=1).indices
    negatives = masked_logits.gather(1, negative_indices)
    pairwise = F.softplus(negatives - positive[:, None]).mean()
    valid_delta = (logits - anchor)[masks]
    return listwise + pairwise_weight * pairwise + anchor_penalty * valid_delta.square().mean()


def predict(
    model: EvidenceGatedRanker,
    features: torch.Tensor,
    masks: torch.Tensor,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    with torch.no_grad():
        scores, gates = model(features)
        scores = scores.masked_fill(~masks, -1e9)
    return scores.cpu().numpy(), gates.cpu().numpy()


def fit_model(
    features: torch.Tensor,
    masks: torch.Tensor,
    labels: torch.Tensor,
    train_rows: np.ndarray,
    validation_rows: np.ndarray | None,
    config: RankerConfig,
    view_z_positions: list[int],
    anchor_view: int,
    epochs: int,
    patience: int,
    learning_rate: float,
    hard_negatives: int,
    seed: int,
) -> tuple[EvidenceGatedRanker, int, float]:
    set_seed(seed)
    model = EvidenceGatedRanker(
        features.shape[-1],
        len(view_z_positions),
        view_z_positions,
        anchor_view,
        config.hidden_size,
        config.dropout,
    ).to(features.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.02)
    train_index = torch.as_tensor(train_rows, dtype=torch.long, device=features.device)
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 1
    best_mrr = -math.inf
    stale = 0
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits, _ = model(features[train_index])
        anchor = features[train_index, :, view_z_positions[anchor_view]]
        loss = ranking_loss(
            logits,
            masks[train_index],
            labels[train_index],
            anchor,
            config.pairwise_weight,
            config.anchor_penalty,
            hard_negatives,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if validation_rows is None:
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            continue
        validation_index = torch.as_tensor(
            validation_rows, dtype=torch.long, device=features.device
        )
        scores, _ = predict(model, features[validation_index], masks[validation_index])
        score = macro_metrics(scores, labels[validation_index].cpu().numpy())["mrr"]
        if score > best_mrr + 1e-8:
            best_mrr = score
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    model.load_state_dict(best_state)
    return model, best_epoch, best_mrr


def select_config(
    features: torch.Tensor,
    masks: torch.Tensor,
    labels: torch.Tensor,
    folds: list[np.ndarray],
    view_z_positions: list[int],
    anchor_view: int,
    args: argparse.Namespace,
) -> tuple[RankerConfig, dict[str, object], np.ndarray, list[int]]:
    all_rows = np.arange(len(labels))
    reports: dict[str, object] = {}
    oof_by_config: dict[str, np.ndarray] = {}
    epochs_by_config: dict[str, list[int]] = {}
    for config_index, config in enumerate(CONFIGS):
        oof = np.full((len(labels), masks.shape[1]), -1e9, dtype="float64")
        fold_metrics = []
        best_epochs = []
        for fold_index, validation_rows in enumerate(folds):
            train_rows = np.setdiff1d(all_rows, validation_rows)
            model, best_epoch, _ = fit_model(
                features,
                masks,
                labels,
                train_rows,
                validation_rows,
                config,
                view_z_positions,
                anchor_view,
                args.epochs,
                args.patience,
                args.learning_rate,
                args.hard_negatives,
                args.seed + config_index * 100 + fold_index,
            )
            index = torch.as_tensor(validation_rows, dtype=torch.long, device=features.device)
            scores, _ = predict(model, features[index], masks[index])
            oof[validation_rows] = scores
            fold_metrics.append(macro_metrics(scores, labels[index].cpu().numpy())["mrr"])
            best_epochs.append(best_epoch)
        oof_metrics = macro_metrics(oof, labels.cpu().numpy())
        reports[config.name] = {
            "config": asdict(config),
            "fold_mrr": fold_metrics,
            "mean_fold_mrr": float(np.mean(fold_metrics)),
            "standard_deviation_fold_mrr": float(np.std(fold_metrics)),
            "oof_metrics": oof_metrics,
            "best_epochs": best_epochs,
        }
        oof_by_config[config.name] = oof
        epochs_by_config[config.name] = best_epochs
    selected = max(
        CONFIGS,
        key=lambda config: (
            reports[config.name]["oof_metrics"]["mrr"],
            -reports[config.name]["standard_deviation_fold_mrr"],
            -config.hidden_size,
        ),
    )
    return selected, reports, oof_by_config[selected.name], epochs_by_config[selected.name]


def subgroup_report(
    baseline: np.ndarray,
    candidate: np.ndarray,
    labels: np.ndarray,
    frame: pd.DataFrame,
    minimum_sources: int,
) -> tuple[dict[str, object], bool]:
    report: dict[str, object] = {}
    not_worse = True
    for column in ("language", "corpus"):
        field = {}
        values = frame[column].astype(str).to_numpy()
        for value in sorted(set(values)):
            mask = values == value
            if int(mask.sum()) < minimum_sources:
                continue
            base_mrr = macro_metrics(baseline[mask], labels[mask])["mrr"]
            candidate_mrr = macro_metrics(candidate[mask], labels[mask])["mrr"]
            delta = candidate_mrr - base_mrr
            field[value] = {
                "n_sources": int(mask.sum()),
                "anchor_mrr": base_mrr,
                "reranker_mrr": candidate_mrr,
                "delta": delta,
            }
            not_worse &= delta >= 0
        report[column] = field
    return report, not_worse


def main() -> None:
    args = parse_args()
    if any("topic" in spec.split("=", 1)[0].lower() for spec in args.scores):
        raise ValueError("Topic/content views are excluded from the Style Match rank")
    matrices, reference = load_aligned_scores(args.scores)
    if args.anchor not in matrices:
        raise ValueError(f"Anchor view not found: {args.anchor}")
    if len(matrices) < 3:
        raise ValueError("At least three independently motivated style views are required")

    frame = aligned_metadata(args.input, reference)
    source_frame, labels, means, stability = aggregate_source_moments(
        frame, reference["y_true"].astype(int), matrices
    )
    profiles = reference["profiles"].astype(str)
    languages = source_frame["language"].astype(str).to_numpy()
    splits = source_frame["split"].astype(str).to_numpy()
    dev_rows = np.flatnonzero(splits == "dev")
    test_rows = np.flatnonzero(splits == "test")
    if not len(dev_rows) or not len(test_rows):
        raise ValueError("Separate dev and locked test sources are required")

    feature_array, feature_names, view_z_positions = make_features(
        means, stability, languages, profiles
    )
    masks_array = candidate_masks(languages, profiles)
    if int(masks_array.sum(axis=1).min()) < 2:
        raise ValueError("Every evaluated language needs at least two candidate profiles")
    device_name = (
        "cuda" if args.device == "auto" and torch.cuda.is_available() else
        "cpu" if args.device == "auto" else args.device
    )
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(device_name)
    features = torch.as_tensor(feature_array, dtype=torch.float32, device=device)
    masks = torch.as_tensor(masks_array, dtype=torch.bool, device=device)
    labels_tensor = torch.as_tensor(labels, dtype=torch.long, device=device)
    anchor_view = list(means).index(args.anchor)

    dev_features = features[dev_rows]
    dev_masks = masks[dev_rows]
    dev_labels = labels_tensor[dev_rows]
    folds = grouped_folds(labels[dev_rows], args.folds, args.seed)
    selected, selection_report, oof_dev, selected_epochs = select_config(
        dev_features,
        dev_masks,
        dev_labels,
        folds,
        view_z_positions,
        anchor_view,
        args,
    )
    final_epochs = max(1, int(round(float(np.median(selected_epochs)))))
    final_model, _, _ = fit_model(
        dev_features,
        dev_masks,
        dev_labels,
        np.arange(len(dev_rows)),
        None,
        selected,
        view_z_positions,
        anchor_view,
        final_epochs,
        args.patience,
        args.learning_rate,
        args.hard_negatives,
        args.seed + 1000,
    )
    test_index = torch.as_tensor(test_rows, dtype=torch.long, device=device)
    test_scores, test_gates = predict(final_model, features[test_index], masks[test_index])
    test_labels = labels[test_rows]
    anchor_dev = means[args.anchor][dev_rows]
    anchor_test = means[args.anchor][test_rows]

    test_metrics = {
        name: macro_metrics(scores[test_rows], test_labels)
        for name, scores in means.items()
    }
    test_metrics["evidence_gated_reranker"] = macro_metrics(test_scores, test_labels)
    intervals = {
        name: profile_bootstrap_intervals(
            scores[test_rows], test_labels, args.bootstrap_runs, args.seed + index
        )
        for index, (name, scores) in enumerate(means.items())
    }
    intervals["evidence_gated_reranker"] = profile_bootstrap_intervals(
        test_scores, test_labels, args.bootstrap_runs, args.seed + len(means)
    )
    bootstrap = paired_profile_bootstrap(
        anchor_test, test_scores, test_labels, args.bootstrap_runs, args.seed + 2000
    )
    calibration = {
        args.anchor: top1_calibration(anchor_dev, labels[dev_rows], anchor_test, test_labels),
        "evidence_gated_reranker": top1_calibration(
            oof_dev, labels[dev_rows], test_scores, test_labels
        ),
    }
    subgroups, subgroup_not_worse = subgroup_report(
        anchor_test,
        test_scores,
        test_labels,
        source_frame.iloc[test_rows].reset_index(drop=True),
        args.minimum_subgroup_sources,
    )
    adopted = (
        bootstrap["ci_low"] > 0
        and calibration["evidence_gated_reranker"]["ece"]
        <= calibration[args.anchor]["ece"] + 0.01
        and calibration["evidence_gated_reranker"]["top1_precision_at_50pct_coverage"]
        >= calibration[args.anchor]["top1_precision_at_50pct_coverage"]
        and subgroup_not_worse
    )

    valid_test = masks_array[test_rows]
    gate_average = {
        name: float(test_gates[..., index][valid_test].mean())
        for index, name in enumerate(means)
    }
    dev_metrics = {
        name: macro_metrics(scores[dev_rows], labels[dev_rows])
        for name, scores in means.items()
    }
    report = {
        "protocol": {
            "evaluation_unit": "independent source",
            "selection": "author-language-profile-grouped cross-validation on dev",
            "locked_evaluation_split": "test",
            "objective": "anchor-residual listwise softmax plus anchor-hard-negative pairwise loss",
            "boundary_signal": "candidate-wise standard deviation across source chunks",
            "score_normalization": "within-language z-score and candidate percentile",
            "topic_views_allowed": False,
            "reinforcement_learning_used": False,
            "reinforcement_learning_reason": "No defensible online reward or interaction data; policy gradients would optimize dev noise.",
            "device": device_name,
        },
        "anchor": args.anchor,
        "candidate_views": list(means),
        "feature_names": feature_names,
        "model_selection": selection_report,
        "selected_configuration": asdict(selected),
        "final_epochs": final_epochs,
        "dev_metrics": dev_metrics,
        "dev_oof_metrics": macro_metrics(oof_dev, labels[dev_rows]),
        "test_metrics": test_metrics,
        "test_intervals": intervals,
        "paired_profile_bootstrap": bootstrap,
        "calibration": calibration,
        "subgroups": subgroups,
        "major_subgroups_not_worse": subgroup_not_worse,
        "mean_test_gate_weight": gate_average,
        "decision": "evidence_gated_reranker" if adopted else args.anchor,
        "reranker_adopted": adopted,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.output_dir / "evidence_gated_reranker_metrics.json"
    score_path = args.output_dir / "evidence_gated_reranker_source_scores.npz"
    model_path = args.output_dir / "evidence_gated_reranker.pt"
    metrics_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    combined_scores = np.full_like(means[args.anchor], -1e9, dtype="float64")
    combined_scores[dev_rows] = oof_dev
    combined_scores[test_rows] = test_scores
    np.savez_compressed(
        score_path,
        source_ids=source_frame["source_key"].astype(str).to_numpy(),
        splits=splits,
        query_languages=languages,
        query_corpora=source_frame["corpus"].astype(str).to_numpy(),
        profiles=profiles,
        y_true=labels,
        anchor_scores=means[args.anchor].astype("float32"),
        evidence_gated_scores=combined_scores.astype("float32"),
    )
    torch.save(
        {
            "state_dict": final_model.cpu().state_dict(),
            "feature_names": feature_names,
            "view_names": list(means),
            "view_z_positions": view_z_positions,
            "anchor_view": anchor_view,
            "configuration": asdict(selected),
        },
        model_path,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
