from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply fixed StyleMatch private-beta release gates.")
    parser.add_argument("--retrieval-metrics", type=Path, required=True)
    parser.add_argument("--open-set-metrics", action="append", type=Path, required=True)
    parser.add_argument("--latency", type=Path, required=True)
    parser.add_argument("--heldout-report", type=Path, required=True)
    parser.add_argument("--index-metadata", type=Path, required=True)
    parser.add_argument("--source-metadata-audit", type=Path, required=True)
    parser.add_argument("--language-id-report", type=Path, required=True)
    parser.add_argument("--readiness-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args()


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def macro_metric(retrieval: dict, key: str, legacy_key: str) -> float:
    groups = retrieval.get("by_language", {})
    values = [
        float(group.get(key, group.get(legacy_key, 0.0)))
        for group in groups.values()
        if int(group.get("n_candidates", 0)) >= 10
    ]
    if values:
        return sum(values) / len(values)
    if groups:
        return 0.0
    return float(retrieval.get(key, retrieval.get(legacy_key, 0.0)))


def main() -> None:
    args = parse_args()
    retrieval = load(args.retrieval_metrics)
    open_sets = [load(path) for path in args.open_set_metrics]
    latency = load(args.latency)
    heldout = load(args.heldout_report)
    index = load(args.index_metadata)
    source_metadata = load(args.source_metadata_audit)
    language_id = load(args.language_id_report)
    readiness = load(args.readiness_report)
    macro_mrr = macro_metric(retrieval, "mrr", "mrr")
    macro_recall3 = macro_metric(retrieval, "recall_at_3", "top3_accuracy")
    open_set_by_language = {
        str(report.get("language") or path.parent.name): {
            "auroc": float(report["open_set"]["auroc"]),
            "ece": float(report["open_set"]["ece"]),
            "top1_precision_at_50pct_coverage": float(
                report["selective_top1_precision_at_50pct_coverage"]
            ),
        }
        for path, report in zip(args.open_set_metrics, open_sets)
    }
    min_auroc = min(value["auroc"] for value in open_set_by_language.values())
    max_ece = max(value["ece"] for value in open_set_by_language.values())
    min_selective = min(
        value["top1_precision_at_50pct_coverage"] for value in open_set_by_language.values()
    )
    checks = {
        "macro_mrr_at_least_0_30": macro_mrr >= 0.30,
        "macro_recall_at_3_at_least_0_35": macro_recall3 >= 0.35,
        "open_set_auroc_at_least_0_80_every_language": min_auroc >= 0.80,
        "ece_at_most_0_08_every_language": max_ece <= 0.08,
        "top1_precision_at_50pct_coverage_at_least_0_60_every_language": min_selective >= 0.60,
        "cpu_p95_at_most_4000ms": float(latency["p95_ms"]) <= 4000.0,
        "no_source_leakage": not bool(heldout.get("source_leakage", True)),
        "deployment_matches_model_selection": bool(index.get("deployment_matches_selection", False)),
        "source_metadata_complete": bool(source_metadata.get("complete", False)),
        "language_identification_passes": bool(language_id.get("passes", False)),
        "baseline_readiness_accepted": bool(readiness.get("accepted", False)),
    }
    report = {
        "observed": {
            "macro_mrr": macro_mrr,
            "macro_recall_at_3": macro_recall3,
            "open_set_by_language": open_set_by_language,
            "minimum_open_set_auroc": min_auroc,
            "maximum_ece": max_ece,
            "minimum_top1_precision_at_50pct_coverage": min_selective,
            "cpu_p95_ms": float(latency["p95_ms"]),
            "selection_decision": index.get("selection_decision"),
            "deployed_model_label": index.get("model_label"),
        },
        "checks": checks,
        "private_beta_ready": all(checks.values()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if args.strict and not report["private_beta_ready"]:
        raise SystemExit("Private-beta release gates failed")


if __name__ == "__main__":
    main()
