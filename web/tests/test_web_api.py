from __future__ import annotations

import os
import unittest
from unittest.mock import patch

os.environ["STYLEMATCH_INDEX_DIR"] = "/nonexistent-force-demo"

from fastapi.testclient import TestClient

from web.api.explain import passage_features
from web.api.language import detect_language
from web.api.main import app

EN_TEXT = (
    "We hold that the measure of a sentence is not its length but its aim, and that the "
    "aim of public speech is to move people toward what they already suspect is right. "
    "When we speak plainly, when we refuse ornament for its own sake, the listener hears "
    "not the speaker but the argument itself, and the argument stands or falls on its merits. "
    "A plain style is not a poor style; it is a discipline, a refusal to let decoration do "
    "the work that reasoning should do. The writer who trusts the reader writes shorter "
    "sentences, chooses common words, and lets the shape of the argument carry the feeling."
)
ZH_TEXT = (
    "我小的时候住在江南的一个小镇上，镇边有一条河，河水在夏天涨起来，淹没了半条街。"
    "大人们并不着急，只把门槛垫高，照旧做买卖，仿佛水与人早已讲好了规矩。"
    "多年以后我在北方想起这条河，才明白所谓故乡，不过是一套彼此忍让的旧约。"
    "北方的冬天干燥而漫长，屋檐下挂着冰，街上的人缩着脖子走路，谁也不肯多说一句话。"
    "我起初很不习惯，后来也学会了在沉默里过日子，把想说的话写在纸上，寄给远方的旧友。"
    "信寄出去往往石沉大海，我却并不灰心，仿佛写下来这件事本身，已经把话说完了。"
)


class LanguageDetectionTests(unittest.TestCase):
    def test_detects_major_scripts_and_english(self) -> None:
        self.assertEqual(detect_language(ZH_TEXT), "zh")
        self.assertEqual(detect_language("吾輩は猫である。名前はまだ無い。"), "ja")
        self.assertEqual(detect_language("Все счастливые семьи похожи друг на друга."), "ru")
        self.assertEqual(detect_language(EN_TEXT), "en")

    def test_returns_none_when_unreliable(self) -> None:
        self.assertIsNone(detect_language("12345 67890"))
        self.assertIsNone(detect_language("lorem ipsum dolor sit amet"))


class ExplainFeatureTests(unittest.TestCase):
    def test_features_are_finite_and_language_aware(self) -> None:
        english = passage_features(EN_TEXT, "en")
        chinese = passage_features(ZH_TEXT, "zh")
        for features in (english, chinese):
            for value in features.values():
                self.assertGreaterEqual(value, 0.0)
        self.assertGreater(english["avg_sentence_len"], 10)
        self.assertGreater(chinese["avg_sentence_len"], 10)


class MatchEndpointTests(unittest.TestCase):
    def test_method_and_author_pages_serve_blueprint_assets(self) -> None:
        with TestClient(app) as client:
            page = client.get("/method.html")
            self.assertEqual(page.status_code, 200)
            self.assertIn('data-method-static', page.text)
            self.assertIn('src="method-flow.js"', page.text)
            self.assertIn('src="method-static.js"', page.text)
            self.assertNotIn('src="method-modal.js"', page.text)

            flow = client.get("/method-flow.js")
            self.assertEqual(flow.status_code, 200)
            self.assertIn("STYLEMATCH_METHOD_FLOW", flow.text)
            self.assertIn("Topic never enters Style Match", flow.text)

            fallback = client.get("/method-flow.svg")
            self.assertEqual(fallback.status_code, 200)
            self.assertEqual(fallback.headers["content-type"], "image/svg+xml")

            portrait = client.get("/head.webp")
            self.assertEqual(portrait.status_code, 200)
            self.assertEqual(portrait.headers["content-type"], "image/webp")

            static_ui = client.get("/method-static.js")
            self.assertEqual(static_ui.status_code, 200)
            self.assertIn("primary-flow", static_ui.text)

            home = client.get("/index.html")
            self.assertEqual(home.status_code, 200)
            self.assertIn('class="site-nav"', home.text)
            self.assertIn('class="art-ribbon"', home.text)
            self.assertNotIn('<dialog', home.text)

            authors = client.get("/authors.html")
            self.assertEqual(authors.status_code, 200)
            self.assertIn('id="author-grid"', authors.text)
            fyi = client.get("/fyi.html")
            self.assertEqual(fyi.status_code, 200)
            self.assertIn("not score 1.00", fyi.text)
            author_data = client.get("/authors-data.js")
            self.assertEqual(author_data.status_code, 200)
            self.assertIn("STYLEMATCH_AUTHORS", author_data.text)

    def test_demo_mode_health_and_default_global_match(self) -> None:
        with TestClient(app) as client:
            health = client.get("/api/health").json()
            self.assertTrue(health["demo"])
            self.assertEqual(health["score_status"], "demo_fixture")
            self.assertEqual(health["score_version"], "demo_v1")
            self.assertEqual(health["artifact_version"], "demo_fixture")

            response = client.post("/api/match", json={"text": EN_TEXT})
            self.assertEqual(response.status_code, 200)
            payload = response.json()
            self.assertTrue(payload["demo"])
            self.assertEqual(payload["input_language"], "en")
            self.assertTrue(payload["language_detected"])
            self.assertEqual(payload["mode"], "all")
            self.assertEqual(payload["ranking_scope"], "global_all_languages")
            self.assertIn(payload["confidence"], {"standard", "reduced"})
            self.assertEqual(payload["style_match_status"], "demo_fixture")
            self.assertEqual(payload["affinity_status"], "provisional_uncalibrated")
            self.assertEqual(payload["decade_match"]["decade"], "1920s")
            self.assertEqual(payload["decade_status"], "demo_fixture")
            self.assertEqual(list(payload["results"]), ["all"])
            matches = payload["results"]["all"]
            self.assertEqual(len(matches), 3)
            top = matches[0]
            for field in ("style_similarity", "topic_similarity", "affinity_score",
                          "style_weight", "why", "representative_passages", "low_confidence",
                          "profile", "style_traits", "photo_url", "cross_language"):
                self.assertIn(field, top)
            self.assertEqual(top["admission_tier"], "exploratory")
            self.assertNotIn("corpus", top)
            self.assertIn("source_corpora", top)
            self.assertEqual(
                len({match["author_or_speaker"] for match in matches}),
                len(matches),
            )
            self.assertTrue(all("corpus" not in match for match in matches))
            for match in matches:
                self.assertEqual(match["cross_language"], match["target_language"] != "en")
                self.assertEqual(match["style_weight"], 0.5 if match["cross_language"] else 0.7)
            affinities = [match["affinity_score"] for match in matches]
            self.assertEqual(affinities, sorted(affinities, reverse=True))

    def test_within_mode_still_ranks_query_language_only(self) -> None:
        with TestClient(app) as client:
            payload = client.post(
                "/api/match", json={"text": EN_TEXT, "mode": "within"}
            ).json()
            self.assertEqual(payload["confidence"], "standard")
            self.assertEqual(payload["ranking_scope"], "within_language")
            matches = payload["results"]["en"]
            self.assertTrue(matches)
            self.assertTrue(all(match["target_language"] == "en" for match in matches))
            self.assertEqual(matches[0]["style_weight"], 0.7)

    def test_cross_mode_is_reduced_confidence_per_language(self) -> None:
        with TestClient(app) as client:
            payload = client.post(
                "/api/match", json={"text": ZH_TEXT, "mode": "cross"}
            ).json()
            self.assertEqual(payload["confidence"], "reduced")
            self.assertEqual(payload["ranking_scope"], "per_target_language")
            self.assertGreater(len(payload["results"]), 1)
            self.assertEqual(payload["results"]["zh"][0]["style_weight"], 0.5)

    def test_short_input_is_rejected_honestly(self) -> None:
        with TestClient(app) as client:
            response = client.post(
                "/api/match", json={"text": "Too short to judge style.", "language": "en"}
            )
            self.assertEqual(response.status_code, 400)
            self.assertIn("too short", response.json()["detail"].lower())

    def test_unsupported_language_is_rejected(self) -> None:
        with TestClient(app) as client:
            response = client.post(
                "/api/match", json={"text": EN_TEXT, "language": "ko"}
            )
            self.assertEqual(response.status_code, 400)

    def test_translation_endpoint_labels_generated_text_honestly(self) -> None:
        with patch("web.api.main._translate_on_demand", return_value="A translated passage."):
            with TestClient(app) as client:
                response = client.post("/api/translate", json={
                    "text": "一段需要翻译的文字。",
                    "source_language": "zh",
                    "target_language": "en",
                })
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["text"], "A translated passage.")
        self.assertEqual(payload["translation_type"], "ai_generated")
        self.assertIsNone(payload["translator"])
        self.assertIsNone(payload["publication_year"])


if __name__ == "__main__":
    unittest.main()
