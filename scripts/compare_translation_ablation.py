from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.retrieval_metrics import paired_bootstrap_mrr, ranking_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare direct original-text and translated-query scores.")
    parser.add_argument("--direct", required=True, help="NPZ_PATH:ARRAY_KEY")
    parser.add_argument("--translated", required=True, help="NPZ_PATH:ARRAY_KEY")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-runs", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260711)
    return parser.parse_args()


def load(spec: str) -> tuple[dict[str, np.ndarray], np.ndarray]:
    path, key = spec.rsplit(":", 1)
    # Same object-dtype string arrays as the recall score files; trusted artifacts.
    payload = np.load(path, allow_pickle=True)
    reference = {field: payload[field] for field in ("chunk_ids", "profiles", "y_true", "splits")}
    return reference, payload[key]


def main() -> None:
    args = parse_args()
    direct_ref, direct = load(args.direct)
    translated_ref, translated = load(args.translated)
    for field in direct_ref:
        if not np.array_equal(direct_ref[field], translated_ref[field]):
            raise ValueError(f"Translation ablation inputs are not aligned on {field}")
    test = direct_ref["splits"].astype(str) == "test"
    labels = direct_ref["y_true"].astype(int)[test]
    report = {
        "direct_original_text": ranking_metrics(direct[test], labels),
        "translation_mediated": ranking_metrics(translated[test], labels),
        "translation_minus_direct": paired_bootstrap_mrr(
            direct[test], translated[test], labels, args.bootstrap_runs, args.seed
        ),
        "headline_condition": "direct_original_text",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
