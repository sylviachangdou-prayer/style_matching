from __future__ import annotations

import re

SUPPORTED_LANGUAGES = {"de", "en", "es", "fr", "it", "ja", "pl", "ru", "zh"}

# Distinctive high-frequency function words per Latin-script language. Detection
# only has to be good enough to preselect a language the user can still override.
_STOPWORDS: dict[str, set[str]] = {
    "en": {"the", "and", "of", "is", "was", "that", "with", "for", "have", "not", "this", "you"},
    "de": {"und", "der", "die", "das", "nicht", "ein", "ist", "mit", "den", "von", "sich", "auch"},
    "fr": {"le", "les", "des", "une", "est", "qui", "dans", "pas", "pour", "avec", "sur", "cette"},
    "es": {"el", "los", "las", "es", "un", "una", "por", "para", "como", "pero", "muy", "cuando"},
    "it": {"il", "che", "di", "non", "per", "sono", "della", "come", "anche", "più", "questo", "gli"},
    "pl": {"się", "nie", "jest", "że", "jak", "ale", "tak", "przez", "tego", "jego", "jednak", "być"},
}

_WORD_RE = re.compile(r"[^\W\d_]+", re.UNICODE)


def _in_range(char: str, start: int, end: int) -> bool:
    return start <= ord(char) <= end


def detect_language(text: str) -> str | None:
    """Best-effort language guess restricted to SUPPORTED_LANGUAGES.

    Returns None when the guess would be unreliable; the caller must then ask
    the user to pick a language instead of silently defaulting.
    """
    letters = [char for char in text if char.isalpha()]
    if not letters:
        return None
    total = len(letters)
    kana = sum(1 for char in letters if _in_range(char, 0x3040, 0x30FF))
    han = sum(1 for char in letters if _in_range(char, 0x4E00, 0x9FFF))
    cyrillic = sum(1 for char in letters if _in_range(char, 0x0400, 0x04FF))
    if kana / total > 0.05:
        return "ja"
    if han / total > 0.30:
        return "zh"
    if cyrillic / total > 0.30:
        return "ru"

    words = [word.lower() for word in _WORD_RE.findall(text)]
    if not words:
        return None
    scores = {
        language: sum(1 for word in words if word in stopwords)
        for language, stopwords in _STOPWORDS.items()
    }
    best = max(scores, key=lambda language: scores[language])
    best_score = scores[best]
    runner_up = max(score for language, score in scores.items() if language != best)
    if best_score < 2 or best_score == runner_up:
        return None
    return best
