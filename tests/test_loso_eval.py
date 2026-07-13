from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

from scripts.evaluate_loso_retrieval import compute_loso_metrics, load_aligned_cache


ROOT = Path(__file__).resolve().parents[1]


def synthetic_corpus(seed: int = 7):
    rng = np.random.default_rng(seed)
    rows = []
    vectors = []
    for language in ("en", "zh"):
        for author_id in range(4):
            author_direction = rng.normal(size=32)
            author_direction /= np.linalg.norm(author_direction)
            # authors 0-2 have 3 sources; author 3 has only 1 (excluded from folds)
            n_sources = 3 if author_id < 3 else 1
            for source_id in range(n_sources):
                for chunk in range(6):
                    noise = rng.normal(scale=0.05, size=32)
                    vector = author_direction + noise
                    vectors.append(vector / np.linalg.norm(vector))
                    rows.append({
                        "language": language,
                        "author_or_speaker": f"{language}_author{author_id}",
                        "corpus": "literary",
                        "source_id": f"{language}{author_id}s{source_id}",
                        "independent_source_id": f"{language}{author_id}s{source_id}",
                        "text": f"chunk {language} {author_id} {source_id} {chunk}",
                    })
    return pd.DataFrame(rows), np.vstack(vectors).astype("float32")


class LosoEvaluationTests(unittest.TestCase):
    def test_separable_authors_are_recovered_and_folds_counted(self) -> None:
        df, embeddings = synthetic_corpus()
        metrics = compute_loso_metrics(df, embeddings, per_source_cap=10, query_cap=10)
        # 3 eligible authors per language, 3 folds each
        self.assertEqual(metrics["n_profiles_evaluated"], 6)
        self.assertEqual(metrics["n_folds"], 18)
        self.assertGreaterEqual(metrics["chunk_level"]["top1_accuracy"], 0.99)
        self.assertGreaterEqual(metrics["source_level"]["mrr"], 0.99)
        self.assertEqual(set(metrics["by_language"]), {"en", "zh"})

    def test_single_source_authors_stay_out_of_folds_but_in_candidates(self) -> None:
        df, embeddings = synthetic_corpus()
        metrics = compute_loso_metrics(df, embeddings)
        # 4 authors per language exist as candidates; only 3 are evaluated
        self.assertEqual(metrics["n_profiles_evaluated"], 6)

    def test_two_source_author_gets_two_folds(self) -> None:
        df, embeddings = synthetic_corpus()
        mask = ~(
            df["author_or_speaker"].eq("en_author0")
            & df["independent_source_id"].eq("en0s2")
        )
        metrics = compute_loso_metrics(
            df[mask].reset_index(drop=True), embeddings[mask.to_numpy()]
        )
        self.assertEqual(metrics["n_folds"], 17)

    def test_cache_alignment_filters_uncached_rows_and_reports_coverage(self) -> None:
        df, embeddings = synthetic_corpus()
        cached_rows = np.arange(len(df) - 6)
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory) / "cache.npz"
            np.savez_compressed(
                cache,
                model_name=np.asarray("test-model"),
                chunk_ids=np.asarray(cached_rows, dtype=str),
                embeddings=embeddings[cached_rows],
            )
            keyed = df.copy()
            keyed["chunk_id"] = keyed.index.astype(str)
            aligned, matrix, coverage = load_aligned_cache(keyed, cache, "test-model")
        self.assertEqual(len(aligned), len(cached_rows))
        self.assertEqual(matrix.shape[0], len(cached_rows))
        self.assertLess(coverage["input_chunk_coverage"], 1.0)

    def test_cli_runs_as_a_direct_script_with_cache_only(self) -> None:
        df, embeddings = synthetic_corpus()
        df["chunk_id"] = [f"chunk_{index}" for index in range(len(df))]
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            parquet = directory / "chunks.parquet"
            cache = directory / "cache.npz"
            output = directory / "metrics.json"
            df.to_parquet(parquet, index=False)
            np.savez_compressed(
                cache,
                model_name=np.asarray("test-model"),
                chunk_ids=np.asarray(df["chunk_id"].astype(str).tolist()),
                embeddings=embeddings,
            )
            result = subprocess.run(
                [
                    sys.executable,
                    "scripts/evaluate_loso_retrieval.py",
                    "--input", str(parquet),
                    "--model-name", "test-model",
                    "--embedding-cache", str(cache),
                    "--output", str(output),
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            report = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(report["n_folds"], 18)
            self.assertEqual(report["cache_coverage"]["input_chunk_coverage"], 1.0)


if __name__ == "__main__":
    unittest.main()
