from __future__ import annotations

import os
import json
import logging
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import yaml
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from web.api.demo_index import DemoIndex
from web.api.explain import cohort_feature_mean, passage_features, shared_style_sentences
from web.api.language import SUPPORTED_LANGUAGES, detect_language

CJK_LANGUAGES = {"zh", "ja"}
DEFAULT_INDEX_DIR = ROOT / "artifacts" / "multilingual_style_index_v1"
CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "weights.yaml"
LOGGER = logging.getLogger("stylematch.metrics")

_DEFAULT_CONFIG = {
    "affinity": {"low_confidence_threshold": 0.25},
    "input": {"min_words": 80, "min_cjk_chars": 160, "max_chars": 8000},
    "retrieval": {"top_k": 3, "max_top_k": 5},
}


def load_config() -> dict:
    if CONFIG_PATH.exists():
        loaded = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8")) or {}
        return {section: {**values, **loaded.get(section, {})} for section, values in _DEFAULT_CONFIG.items()}
    return _DEFAULT_CONFIG


def load_index() -> tuple[object, bool]:
    """Load the real StyleIndex once at process start; fall back to the demo
    fixture when the Colab-built artifacts are not present yet."""
    index_dir = Path(os.environ.get("STYLEMATCH_INDEX_DIR", DEFAULT_INDEX_DIR))
    hub_repo = os.environ.get("STYLEMATCH_HUB_REPO")
    if not (index_dir / "metadata.json").exists() and hub_repo:
        from huggingface_hub import snapshot_download

        snapshot_download(
            repo_id=hub_repo,
            repo_type=os.environ.get("STYLEMATCH_HUB_REPO_TYPE", "dataset"),
            local_dir=index_dir,
        )
    if (index_dir / "metadata.json").exists():
        from scripts.multilingual_style_index import StyleIndex

        return StyleIndex(index_dir, device=os.environ.get("STYLEMATCH_DEVICE", "auto")), False
    return DemoIndex(), True


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.config = load_config()
    app.state.index, app.state.demo = load_index()
    yield


app = FastAPI(title="StyleMatch API", lifespan=lifespan)

_cors_origins = os.environ.get("STYLEMATCH_CORS_ORIGINS", "*")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[origin.strip() for origin in _cors_origins.split(",")],
    allow_methods=["*"],
    allow_headers=["*"],
)


class MatchRequest(BaseModel):
    text: str = Field(min_length=1)
    language: str | None = None
    mode: Literal["all", "within", "cross"] = "all"
    top_k: int | None = Field(default=None, ge=1)


def _validate_input(text: str, language: str, config: dict) -> None:
    limits = config["input"]
    if len(text) > limits["max_chars"]:
        raise HTTPException(status_code=400, detail=f"Text exceeds {limits['max_chars']} characters.")
    if language in CJK_LANGUAGES:
        n_chars = sum(1 for char in text if char.isalpha())
        if n_chars < limits["min_cjk_chars"]:
            raise HTTPException(
                status_code=400,
                detail=f"Passage too short for a defensible match: need at least {limits['min_cjk_chars']} characters.",
            )
    else:
        n_words = len(text.split())
        if n_words < limits["min_words"]:
            raise HTTPException(
                status_code=400,
                detail=f"Passage too short for a defensible match: need at least {limits['min_words']} words.",
            )


def _attach_explanations(result: dict, text: str, language: str) -> None:
    """Add interpretable 'why' sentences per match, comparing the user passage
    with each match's representative passages against the candidate cohort."""
    all_matches = [match for matches in result["results"].values() for match in matches]
    author_features: dict[int, dict[str, float]] = {}
    for position, match in enumerate(all_matches):
        joined = " ".join(passage["text"] for passage in match["representative_passages"])
        feature_language = match["target_language"]
        author_features[position] = passage_features(joined, feature_language)
    cohort = cohort_feature_mean(list(author_features.values()))
    user = passage_features(text, language)
    for position, match in enumerate(all_matches):
        match["why"] = shared_style_sentences(user, author_features[position], cohort)


@app.get("/api/health")
def health() -> dict:
    metadata = app.state.index.metadata
    return {
        "status": "ok",
        "demo": app.state.demo,
        "model_name": metadata.get("model_name"),
        "topic_model_name": metadata.get("topic_model_name"),
        "n_profiles": metadata.get("n_profiles"),
        "n_decade_profiles": metadata.get("n_decade_profiles", 0),
        "languages": metadata.get("languages"),
        "score_status": metadata.get("score_status"),
        "score_version": metadata.get("score_version"),
        "artifact_version": metadata.get("artifact_version"),
        "selection_decision": metadata.get("selection_decision"),
        "deployment_matches_selection": metadata.get("deployment_matches_selection"),
    }


@app.post("/api/match")
def match(request: MatchRequest) -> dict:
    config = app.state.config
    text = " ".join(request.text.split())
    language = request.language or detect_language(text)
    if language is None:
        raise HTTPException(
            status_code=400,
            detail="Could not detect the language reliably; please paste a longer passage in its original language.",
        )
    if language not in SUPPORTED_LANGUAGES:
        raise HTTPException(status_code=400, detail=f"Unsupported language: {language}")
    _validate_input(text, language, config)

    top_k = min(request.top_k or config["retrieval"]["top_k"], config["retrieval"]["max_top_k"])
    started = time.perf_counter()
    result = app.state.index.query(text, language, request.mode, top_k)
    elapsed_ms = (time.perf_counter() - started) * 1000

    _attach_explanations(result, text, language)
    threshold = config["affinity"]["low_confidence_threshold"]
    for matches in result["results"].values():
        for match_row in matches:
            rejection = result.get("rejection", {}).get(match_row.get("target_language"), {})
            match_row["low_confidence"] = (
                match_row["affinity_score"] < threshold
                or match_row.get("admission_tier", "exploratory") != "formal"
                or rejection.get("accept") is False
            )

    result["style_match_status"] = "calibrated" if result.get("calibrated") else result.get(
        "score_status", "uncalibrated"
    )
    result["affinity_status"] = "provisional_uncalibrated"

    result.update({
        "language_detected": request.language is None,
        "demo": app.state.demo,
        "low_confidence_threshold": threshold,
        "elapsed_ms": round(elapsed_ms, 1),
    })
    LOGGER.info(json.dumps({
        "event": "match",
        "input_language": language,
        "mode": request.mode,
        "elapsed_ms": round(elapsed_ms, 1),
        "low_confidence": all(
            match_row.get("low_confidence", False)
            for matches in result["results"].values()
            for match_row in matches
        ),
        "returned_authors": [
            match_row["author_or_speaker"]
            for matches in result["results"].values()
            for match_row in matches
        ],
        "decade": (result.get("decade_match") or {}).get("decade"),
    }, ensure_ascii=False))
    return result


_static_dir = Path(__file__).resolve().parents[1] / "static"
if _static_dir.exists():
    app.mount("/", StaticFiles(directory=_static_dir, html=True), name="static")
