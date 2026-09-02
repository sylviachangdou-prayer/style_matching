# CLAUDE.md — StyleMatch

StyleMatch is a public-facing, open-set style-profile retrieval system: given a passage it
returns the closest author profiles with separate style and topic scores and evidence. It is
not a closed-set classifier and must allow "no strong match."

## Hard requirements

- Use only original-language primary texts. Exclude translations, adaptations, subtitles,
  summaries, paraphrases, and generated imitation.
- Keep style and topic physically separate through encoding, indexing, scoring, and display.
- Never call an uncalibrated cosine a probability or percentage.
- Never force a match below the calibrated rejection threshold.
- Never use random-split accuracy as headline evidence; split complete sources, not chunks.
- Label cross-language retrieval as reduced-confidence; do not imply parity with
  within-language retrieval.
- Do not add scientific text, per-author language models, online translation fan-out, or
  video to v1.

## Product output

Ranked top three when confidence permits: StyleSim determines rank; composite Affinity and
TopicSim remain separate context; 2–4 plain-language stylometric features; up to three
original-language passages only when display is explicitly allowed; calibration and
reduced-confidence status. Affinity weights (product rules, not ranking weights):
`0.7*Style + 0.3*Topic` within language, `0.5/0.5` across. Tune display weights and rejection
thresholds on held-out development data.

## Model and retrieval

Open-set author-profile ranking plus verification. Deployed (July 2026 selection, recorded
in `artifacts/model_comparison_v1.json` and index metadata): the fine-tuned multilingual
authorship-representation encoder `artifacts/multilingual_author_style_v1` with index
`artifacts/multilingual_style_index_challenger_v1`. `StyleDistance/mstyledistance` is the
reproducible baseline of record — never silently replaced, never silently restored. Topic:
`intfloat/multilingual-e5-base`, separate channel.

Offline: detect language, strip boilerplate/quotes, chunk 75–150 words register-consistently;
build source/topic/time-heldout splits; encode original-language chunks; build normalized
author centroids plus source prototypes (strategy chosen only on identical held-out
comparisons); cache embeddings by `chunk_id`. Corpus expansion re-encodes only new chunks
with the frozen encoder — no re-training. After the profile set changes, refit open-set
calibration (`evaluate_open_set.py` per language, then
`multilingual_style_index.py calibrate`; it refuses cross-encoder calibration).

Online: validate language and length; encode the query once per channel; retrieve by exact
matrix multiplication; never encode profiles or download models in a request handler.
Cross-language mode compares original texts directly (no query translation) and returns
per-target-language rankings until ordered-pair calibration exists. Rerank only a small
top-K with interpretable features (character n-grams, function words, punctuation, rhythm,
lexical diversity, hedging) — complements, never replaces, the encoder.

## Evaluation gates

Production requires: source-heldout Recall@3/5/20 and MRR overall and by language/corpus;
cross-topic and cross-domain by language; open-set AUROC, EER, thresholds, calibration,
selective risk; ordered language-pair results without macro-average hiding; a
translation-mediated ablation outside the headline; warm p50/p95 latency (GPU p50 ≤ 500 ms,
p95 ≤ 1.5 s; CPU p95 ≤ 4 s; ≥30 timed requests; model load separate). Source-heldout is
necessary but insufficient (author–topic correlation persists); character models are
leakage ceilings; raw multilingual cosines are not comparable across pairs.

## Corpus

The registry is the author universe; source rows determine coverage. Keep author, language,
title, date, topic/domain, corpus, source URL/ID, `independent_source_id` (shared by
mirrors/editions), and original-text status. Fewer than three independent sources →
exploratory only, excluded from headline evaluation, flagged. Fewer than two → excluded
from any fine-tuning. Only licence-approved sources may expose passages; rights-cleared
in-copyright texts enter via `source_format: local_text` (owner-supplied files,
`rights_cleared_private`, passages never displayed). Decade output needs verified years,
author-heldout validation, five authors and twenty sources per language/register/class.

## Implementation priorities

Run the canonical multilingual notebook from source collection; preserve artifact hashes,
coverage tiers, model comparison, open-set calibration, ordered-pair evaluation, CPU/GPU
latency, and release gates. Do not publish when `private_beta_ready` is false. Do not build
speculative v2 components (explicit disentanglement, interpretable embedding objectives,
per-author LMs, LLM reranking); LLMs may later explain or rerank but never solely score.
