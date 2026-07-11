from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from web.api.language import SUPPORTED_LANGUAGES, detect_language


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark supported-language detection.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-accuracy", type=float, default=0.95)
    parser.add_argument("--min-cases-per-language", type=int, default=20)
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.input)
    if not {"language", "text"}.issubset(df.columns):
        raise ValueError("Language benchmark requires language,text columns")
    df["predicted_language"] = df["text"].astype(str).map(detect_language)
    df["correct"] = df["predicted_language"].eq(df["language"])
    accuracy = float(df["correct"].mean())
    counts = df["language"].value_counts().to_dict()
    coverage_passes = all(
        counts.get(language, 0) >= args.min_cases_per_language for language in SUPPORTED_LANGUAGES
    )
    report = {
        "n_cases": int(len(df)),
        "accuracy": accuracy,
        "minimum_accuracy": args.min_accuracy,
        "minimum_cases_per_language": args.min_cases_per_language,
        "cases_by_language": {str(key): int(value) for key, value in counts.items()},
        "passes": accuracy >= args.min_accuracy and coverage_passes,
        "by_language": df.groupby("language")["correct"].mean().astype(float).to_dict(),
        "errors": df.loc[~df["correct"], ["language", "predicted_language"]].to_dict("records"),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if args.strict and not report["passes"]:
        raise SystemExit("Language identification gate failed; replace the heuristic with a local LID model")


if __name__ == "__main__":
    main()
