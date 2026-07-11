from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit corpus tiers and freeze a reproducible baseline manifest.")
    parser.add_argument("--registry", type=Path, default=Path("data/source_registry/all_people.csv"))
    parser.add_argument("--chunks", type=Path, required=True)
    parser.add_argument("--heldout-report", type=Path, required=True)
    parser.add_argument("--training-config", type=Path, required=True)
    parser.add_argument("--index-metadata", type=Path, required=True)
    parser.add_argument("--embedding-metrics", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--version", default="baseline_v1")
    parser.add_argument("--expected-languages", default="en,zh,ja,fr,de,ru")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--extra-artifact", action="append", type=Path, default=[])
    return parser.parse_args()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def coverage_matrix(registry_path: Path, chunks: pd.DataFrame, heldout_report: dict) -> pd.DataFrame:
    with registry_path.open(newline="", encoding="utf-8") as handle:
        registry = list(csv.DictReader(handle))
    registry_profiles: dict[tuple[str, str], set[str]] = {}
    for row in registry:
        registry_profiles.setdefault((row["original_language"], row["name"]), set()).add(row["corpus"])

    chunks = chunks.copy()
    identity = (
        chunks["independent_source_id"]
        if "independent_source_id" in chunks
        else chunks["source_id"]
    )
    chunks["source_key"] = chunks["corpus"].astype(str) + "::" + identity.fillna("").astype(str)
    observed = {
        (str(key[0]), str(key[1])): {
            "n_sources": int(group["source_key"].nunique()),
            "n_chunks": int(len(group)),
            "observed_corpora": ",".join(sorted(group["corpus"].astype(str).unique())),
        }
        for key, group in chunks.groupby(["language", "author_or_speaker"], sort=True)
    }
    heldout = {
        (str(row["language"]), str(row["author"])): row
        for row in heldout_report.get("authors", [])
    }
    rows = []
    for language, author in sorted(registry_profiles):
        counts = observed.get((language, author), {"n_sources": 0, "n_chunks": 0, "observed_corpora": ""})
        report = heldout.get((language, author), {})
        eligible = bool(report.get("eligible", False))
        if eligible:
            tier = "formal"
        elif counts["n_sources"]:
            tier = "exploratory"
        else:
            tier = "catalog_only"
        rows.append({
            "language": language,
            "author_or_speaker": author,
            "registry_corpora": ",".join(sorted(registry_profiles[(language, author)])),
            **counts,
            "heldout_eligible": eligible,
            "heldout_reason": report.get("reason", "no_chunks" if not counts["n_sources"] else "not_evaluated"),
            "admission_tier": tier,
        })
    return pd.DataFrame(rows)


def git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def main() -> None:
    args = parse_args()
    heldout = read_json(args.heldout_report)
    training = read_json(args.training_config)
    index = read_json(args.index_metadata)
    metrics = read_json(args.embedding_metrics)
    chunks = pd.read_parquet(args.chunks)
    matrix = coverage_matrix(args.registry, chunks, heldout)

    observed_profiles = int(chunks[["language", "author_or_speaker"]].drop_duplicates().shape[0])
    expected_languages = {value.strip() for value in args.expected_languages.split(",") if value.strip()}
    index_languages = set(index.get("languages", []))
    checks = {
        "independent_source_identity_present": "independent_source_id" in chunks.columns,
        "expected_languages_present": expected_languages.issubset(index_languages),
        "all_eligible_profiles_contributed_pairs": bool(training.get("all_eligible_profiles_contributed_pairs")),
        "index_profiles_match_chunks": int(index.get("n_profiles", -1)) == observed_profiles,
        "heldout_profile_count_matches": int(heldout.get("n_author_language_profiles", -1))
        == int(matrix["heldout_eligible"].sum()),
        "within_metrics_present": all(key in metrics for key in ("top1_accuracy", "top3_accuracy", "mrr")),
    }
    summary = {
        "version": args.version,
        "registry_profiles": int(len(matrix)),
        "profiles_with_chunks": observed_profiles,
        "formal_profiles": int(matrix["admission_tier"].eq("formal").sum()),
        "exploratory_profiles": int(matrix["admission_tier"].eq("exploratory").sum()),
        "catalog_only_profiles": int(matrix["admission_tier"].eq("catalog_only").sum()),
        "source_coverage_buckets": {
            name: {
                "count": int(mask.sum()),
                "profiles": matrix.loc[mask, ["language", "author_or_speaker"]].to_dict("records"),
            }
            for name, mask in {
                "no_text": matrix["n_sources"].eq(0),
                "one_source": matrix["n_sources"].eq(1),
                "two_sources": matrix["n_sources"].eq(2),
                "three_or_more_sources": matrix["n_sources"].ge(3),
            }.items()
        },
        "languages": sorted(index_languages),
        "checks": checks,
        "accepted": all(checks.values()),
    }
    artifacts = [
        args.registry,
        args.chunks,
        args.heldout_report,
        args.training_config,
        args.index_metadata,
        args.embedding_metrics,
    ] + [path for path in args.extra_artifact if path.exists()]
    manifest = {
        "version": args.version,
        "git_commit": git_commit(),
        "artifacts": [
            {"path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size}
            for path in artifacts
        ],
        "summary": summary,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    matrix.to_csv(args.output_dir / "coverage_matrix.csv", index=False)
    (args.output_dir / "readiness_report.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (args.output_dir / "artifact_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if args.strict and not summary["accepted"]:
        failed = [name for name, passed in checks.items() if not passed]
        raise SystemExit(f"Baseline acceptance failed: {failed}")


if __name__ == "__main__":
    main()
