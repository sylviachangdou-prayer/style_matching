from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from scripts.build_chunk_parquet_from_sources import chunk_cjk, chunk_words, normalize_source_text
from scripts.fetch_multilingual_sources import clean_gutenberg, row_manifest_metadata, script_ratio
from scripts import import_source_manifest


class ChunkingTests(unittest.TestCase):
    def test_word_chunking_preserves_bounds(self) -> None:
        chunks = chunk_words(" ".join(f"word{i}" for i in range(310)), 75, 150)
        self.assertEqual([len(chunk.split()) for chunk in chunks], [150, 150])

    def test_cjk_chunking_preserves_bounds_and_sentence_endings(self) -> None:
        text = "这是一个用于测试分块边界的句子。" * 80
        chunks = chunk_cjk(text, 250, 500)
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(250 <= len(chunk) <= 500 for chunk in chunks))
        self.assertTrue(all(chunk.endswith("。") for chunk in chunks))


class SourceValidationTests(unittest.TestCase):
    def test_html_wrappers_are_removed_from_source_chunks(self) -> None:
        self.assertEqual(normalize_source_text("甲<br>乙&nbsp;丙"), "甲\n乙 丙")

    def test_gutenberg_boilerplate_is_removed(self) -> None:
        text = "header\n*** START OF THE PROJECT GUTENBERG EBOOK TEST ***\nbody\n*** END OF THE PROJECT GUTENBERG EBOOK TEST ***\nfooter"
        self.assertEqual(clean_gutenberg(text), "body")

    def test_script_ratios_distinguish_original_scripts(self) -> None:
        self.assertGreater(script_ratio("这是中文原文。", "zh"), 0.9)
        self.assertGreater(script_ratio("Это русский текст.", "ru"), 0.9)
        self.assertLess(script_ratio("This is English.", "ru"), 0.1)

    def test_approved_remote_text_is_indexable_but_not_displayable(self) -> None:
        metadata = row_manifest_metadata(
            {
                "corpus": "literary",
                "name": "J. K. Rowling",
                "original_language": "en",
                "year": "1997",
                "title": "Harry Potter and the Sorcerer's Stone",
                "source_id": "approved_rowling_sorcerers_stone",
                "source_url": "https://github.com/example/repo/blob/main/book.txt",
                "source_format": "approved_remote_text",
            }
        )
        self.assertEqual(metadata["license_status"], "rights_cleared_research")
        self.assertEqual(metadata["display_allowed"], "false")

    def test_manifest_outside_registry_resolves_registry_raw_input(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            registry = root / "data" / "source_registry"
            raw_input = registry / "raw_inputs" / "book.txt"
            raw_input.parent.mkdir(parents=True)
            raw_input.write_text("text", encoding="utf-8")
            with mock.patch.object(import_source_manifest, "REGISTRY_DIR", registry):
                resolved, _ = import_source_manifest.resolve_local_path(
                    "raw_inputs/book.txt", root / "artifacts" / "expansion"
                )
            self.assertEqual(resolved, raw_input)

    def test_manifest_can_skip_a_stale_missing_source(self) -> None:
        row = {
            "corpus": "literary",
            "name": "Missing Writer",
            "original_language": "en",
            "title": "Missing Work",
            "source_id": "missing_work",
            "local_text_path": "raw_inputs/missing.txt",
        }
        with TemporaryDirectory() as directory, mock.patch.object(
            import_source_manifest, "registry_keys", return_value=set()
        ):
            imported = import_source_manifest.import_rows(
                [row], Path(directory), dry_run=True, skip_missing=True
            )
        self.assertEqual(imported, {})


if __name__ == "__main__":
    unittest.main()
