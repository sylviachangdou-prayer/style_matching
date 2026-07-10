from __future__ import annotations

import hashlib

# Public-domain excerpts used only so the frontend can be developed before the
# Colab-built index artifacts exist. Scores produced by DemoIndex are fabricated
# fixtures; every response is flagged demo and the UI must show a banner.
_DEMO_PROFILES: list[dict] = [
    {
        "author_or_speaker": "Abraham Lincoln",
        "corpus": "rhetorical",
        "language": "en",
        "passages": [
            {
                "title": "Gettysburg Address (demo fixture)",
                "source_id": "demo-lincoln-1",
                "text": "Four score and seven years ago our fathers brought forth on this continent, a new nation, conceived in Liberty, and dedicated to the proposition that all men are created equal.",
            },
            {
                "title": "Second Inaugural (demo fixture)",
                "source_id": "demo-lincoln-2",
                "text": "With malice toward none, with charity for all, with firmness in the right as God gives us to see the right, let us strive on to finish the work we are in.",
            },
        ],
    },
    {
        "author_or_speaker": "Jane Austen",
        "corpus": "literary",
        "language": "en",
        "passages": [
            {
                "title": "Pride and Prejudice (demo fixture)",
                "source_id": "demo-austen-1",
                "text": "It is a truth universally acknowledged, that a single man in possession of a good fortune, must be in want of a wife.",
            },
            {
                "title": "Emma (demo fixture)",
                "source_id": "demo-austen-2",
                "text": "Emma Woodhouse, handsome, clever, and rich, with a comfortable home and happy disposition, seemed to unite some of the best blessings of existence.",
            },
        ],
    },
    {
        "author_or_speaker": "Frederick Douglass",
        "corpus": "rhetorical",
        "language": "en",
        "passages": [
            {
                "title": "What to the Slave is the Fourth of July? (demo fixture)",
                "source_id": "demo-douglass-1",
                "text": "What have I, or those I represent, to do with your national independence? Are the great principles of political freedom and of natural justice, embodied in that Declaration of Independence, extended to us?",
            },
        ],
    },
    {
        "author_or_speaker": "鲁迅",
        "corpus": "literary",
        "language": "zh",
        "passages": [
            {
                "title": "《呐喊》自序（演示样例）",
                "source_id": "demo-luxun-1",
                "text": "我在年青时候也曾经做过许多梦，后来大半忘却了，但自己也并不以为可惜。所谓回忆者，虽说可以使人欢欣，有时也不免使人寂寞。",
            },
        ],
    },
    {
        "author_or_speaker": "吴趼人",
        "corpus": "literary",
        "language": "zh",
        "passages": [
            {
                "title": "《二十年目睹之怪现状》（演示样例）",
                "source_id": "demo-wujianren-1",
                "text": "上海地方，为通商大埠，中外杂处，人烟稠密，轮舶往来，百货輻辏。",
            },
        ],
    },
    {
        "author_or_speaker": "Victor Hugo",
        "corpus": "literary",
        "language": "fr",
        "passages": [
            {
                "title": "Les Misérables (demo fixture)",
                "source_id": "demo-hugo-1",
                "text": "Il y a un spectacle plus grand que la mer, c'est le ciel; il y a un spectacle plus grand que le ciel, c'est l'intérieur de l'âme.",
            },
        ],
    },
    {
        "author_or_speaker": "Franz Kafka",
        "corpus": "literary",
        "language": "de",
        "passages": [
            {
                "title": "Die Verwandlung (demo fixture)",
                "source_id": "demo-kafka-1",
                "text": "Als Gregor Samsa eines Morgens aus unruhigen Träumen erwachte, fand er sich in seinem Bett zu einem ungeheueren Ungeziefer verwandelt.",
            },
        ],
    },
    {
        "author_or_speaker": "Лев Толстой",
        "corpus": "literary",
        "language": "ru",
        "passages": [
            {
                "title": "Анна Каренина (demo fixture)",
                "source_id": "demo-tolstoy-1",
                "text": "Все счастливые семьи похожи друг на друга, каждая несчастливая семья несчастлива по-своему.",
            },
        ],
    },
    {
        "author_or_speaker": "夏目漱石",
        "corpus": "literary",
        "language": "ja",
        "passages": [
            {
                "title": "吾輩は猫である（デモ）",
                "source_id": "demo-soseki-1",
                "text": "吾輩は猫である。名前はまだ無い。どこで生れたかとんと見当がつかぬ。",
            },
        ],
    },
]


def _pseudo_score(salt: str, author: str, text: str, low: float, high: float) -> float:
    digest = hashlib.md5(f"{salt}|{author}|{len(text)}|{text[:64]}".encode()).hexdigest()
    fraction = int(digest[:8], 16) / 0xFFFFFFFF
    return low + fraction * (high - low)


class DemoIndex:
    """Same response shape as scripts.multilingual_style_index.StyleIndex.query,
    with fabricated deterministic scores. Used only when the real index
    artifacts are absent so frontend work can proceed during training."""

    is_demo = True

    def __init__(self) -> None:
        self.metadata = {
            "model_name": "demo-fixture",
            "topic_model_name": "demo-fixture",
            "style_weight_within": 0.7,
            "style_weight_cross": 0.5,
            "n_profiles": len(_DEMO_PROFILES),
            "languages": sorted({profile["language"] for profile in _DEMO_PROFILES}),
            "score_status": "demo_fixture",
        }

    def query(self, text: str, language: str, mode: str, top_k: int) -> dict:
        style_weight = self.metadata[
            "style_weight_within" if mode == "within" else "style_weight_cross"
        ]
        if mode == "within":
            groups = {language: [p for p in _DEMO_PROFILES if p["language"] == language]}
            confidence = "standard"
            scope = "within_language"
        else:
            groups = {}
            for profile in _DEMO_PROFILES:
                groups.setdefault(profile["language"], []).append(profile)
            confidence = "reduced"
            scope = "per_target_language"

        results: dict[str, list[dict]] = {}
        for target_language, profiles in groups.items():
            matches = []
            for profile in profiles:
                author = profile["author_or_speaker"]
                style = _pseudo_score("style", author, text, 0.15, 0.60)
                topic = _pseudo_score("topic", author, text, 0.20, 0.70)
                matches.append({
                    "author_or_speaker": author,
                    "source_corpora": [profile["corpus"]],
                    "target_language": target_language,
                    "style_similarity": style,
                    "topic_similarity": topic,
                    "affinity_score": style_weight * style + (1.0 - style_weight) * topic,
                    "style_weight": float(style_weight),
                    "calibrated": False,
                    "representative_passages": profile["passages"],
                })
            matches.sort(key=lambda match: match["affinity_score"], reverse=True)
            results[target_language] = matches[:top_k]
        return {
            "mode": mode,
            "input_language": language,
            "confidence": confidence,
            "ranking_scope": scope,
            "score_status": "demo_fixture",
            "results": results,
        }
