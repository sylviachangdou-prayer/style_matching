from __future__ import annotations

import unittest

from scripts.build_chunk_parquet_from_sources import chunk_cjk, chunk_words, normalize_source_text
from scripts.fetch_multilingual_sources import clean_gutenberg, script_ratio


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


if __name__ == "__main__":
    unittest.main()
