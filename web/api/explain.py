from __future__ import annotations

import re
import statistics

CJK_LANGUAGES = {"zh", "ja"}

_SENTENCE_RE = re.compile(r"[.!?。！？…]+")
_WORD_RE = re.compile(r"[^\W\d_]+", re.UNICODE)


def passage_features(text: str, language: str) -> dict[str, float]:
    """Classical interpretable features of one passage.

    Sentence length is measured in words for space-delimited languages and in
    characters for Chinese/Japanese, so values are only comparable within one
    language — which is all the explanation layer compares.
    """
    sentences = [part.strip() for part in _SENTENCE_RE.split(text) if part.strip()]
    if not sentences:
        sentences = [text]
    if language in CJK_LANGUAGES:
        tokens_per_sentence = [len([c for c in sentence if c.isalpha()]) for sentence in sentences]
        tokens = [char for char in text if char.isalpha()]
    else:
        tokens_per_sentence = [len(_WORD_RE.findall(sentence)) for sentence in sentences]
        tokens = [word.lower() for word in _WORD_RE.findall(text)]
    n_sentences = len(sentences)
    n_tokens = max(len(tokens), 1)
    return {
        "avg_sentence_len": sum(tokens_per_sentence) / n_sentences,
        "sentence_len_std": statistics.pstdev(tokens_per_sentence) if n_sentences > 1 else 0.0,
        "type_token_ratio": len(set(tokens)) / n_tokens,
        "comma_rate": sum(text.count(mark) for mark in ",，、") / n_sentences,
        "exclamation_rate": sum(text.count(mark) for mark in "!！") / n_sentences,
        "question_rate": sum(text.count(mark) for mark in "?？") / n_sentences,
        "pause_mark_rate": sum(text.count(mark) for mark in ";；:：—–") / n_sentences,
    }


_TEMPLATES: dict[str, tuple[str, str]] = {
    "avg_sentence_len": (
        "Long sentences (yours average {user:.0f} units per sentence; this author {author:.0f}, library {cohort:.0f})",
        "Short, clipped sentences (yours average {user:.0f} units; this author {author:.0f}, library {cohort:.0f})",
    ),
    "sentence_len_std": (
        "Strongly varied sentence rhythm (spread {user:.0f} vs library {cohort:.0f})",
        "Even, steady sentence rhythm (spread {user:.0f} vs library {cohort:.0f})",
    ),
    "type_token_ratio": (
        "Wide vocabulary range ({user:.2f} distinct-word ratio vs library {cohort:.2f})",
        "Deliberately repetitive wording ({user:.2f} distinct-word ratio vs library {cohort:.2f})",
    ),
    "comma_rate": (
        "Comma-rich, heavily subordinated clauses ({user:.1f} per sentence vs library {cohort:.1f})",
        "Sparse comma use ({user:.1f} per sentence vs library {cohort:.1f})",
    ),
    "exclamation_rate": (
        "Frequent exclamations ({user:.2f} per sentence vs library {cohort:.2f})",
        "Avoids exclamation ({user:.2f} per sentence vs library {cohort:.2f})",
    ),
    "question_rate": (
        "Frequent rhetorical questions ({user:.2f} per sentence vs library {cohort:.2f})",
        "Rarely asks questions ({user:.2f} per sentence vs library {cohort:.2f})",
    ),
    "pause_mark_rate": (
        "Fond of semicolons, colons and dashes ({user:.2f} per sentence vs library {cohort:.2f})",
        "Plain punctuation, few pause marks ({user:.2f} per sentence vs library {cohort:.2f})",
    ),
}


def shared_style_sentences(
    user: dict[str, float],
    author: dict[str, float],
    cohort_mean: dict[str, float],
    max_sentences: int = 4,
) -> list[str]:
    """Features where the user deviates from the candidate cohort mean in the
    same direction as the matched author, rendered as plain sentences."""
    scored: list[tuple[float, str]] = []
    for feature, (high_template, low_template) in _TEMPLATES.items():
        cohort = cohort_mean.get(feature, 0.0)
        user_delta = user[feature] - cohort
        author_delta = author[feature] - cohort
        if user_delta == 0 or author_delta == 0 or (user_delta > 0) != (author_delta > 0):
            continue
        scale = abs(cohort) + 1e-9
        strength = min(abs(user_delta), abs(author_delta)) / scale
        if strength < 0.15:
            continue
        template = high_template if user_delta > 0 else low_template
        scored.append((
            strength,
            template.format(user=user[feature], author=author[feature], cohort=cohort),
        ))
    scored.sort(key=lambda item: item[0], reverse=True)
    sentences = [sentence for _, sentence in scored[:max_sentences]]
    if not sentences:
        sentences = [
            "No single classical feature stands out; the match rests on the style embedding as a whole."
        ]
    return sentences


def cohort_feature_mean(feature_dicts: list[dict[str, float]]) -> dict[str, float]:
    if not feature_dicts:
        return {}
    keys = feature_dicts[0].keys()
    return {
        key: sum(features[key] for features in feature_dicts) / len(feature_dicts)
        for key in keys
    }
