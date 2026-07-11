# CLAUDE.md — StyleMatch

StyleMatch is a public-facing, open-set style-profile retrieval system. Given a passage, it
returns the closest author profiles, separate style and topic scores, and evidence for the
match. It is not a closed-set authorship classifier and must allow “no strong match.”

## Hard requirements

- Use only original-language primary texts. Exclude translations, adaptations, subtitles,
  summaries, paraphrases, and generated imitation.
- Keep style and topic physically separate through encoding, indexing, scoring, and display.
- Never call an uncalibrated cosine a probability or percentage.
- Never force a match below the calibrated rejection threshold.
- Never use random-split accuracy as headline evidence. Split complete sources, not chunks.
- Label cross-language retrieval as reduced-confidence. Do not imply parity with
  within-language retrieval.
- Do not add scientific text, per-author language models, online translation fan-out, or video
  to v1.

## Product output

Return a ranked top three when confidence permits. For each result show:

- composite Affinity score;
- separate StyleSim and TopicSim;
- 2–4 distinguishing stylometric features in plain language;
- up to three representative passages in the author’s original language, only when display is
  explicitly allowed;
- calibration status and reduced-confidence status where applicable.

Use `0.7 * StyleSim + 0.3 * TopicSim` within language. Use `0.5 * StyleSim + 0.5 *
TopicSim` across languages only as the current product rule, not as an empirically established
optimum. Tune weights and rejection thresholds on held-out development data.

## Model and retrieval strategy

Treat the task as open-set author-profile ranking plus verification.

### Offline

1. Detect language, remove boilerplate/markup/quoted contamination, and create
   register-consistent chunks of roughly 75–150 words.
2. Build source-heldout, topic/domain-heldout, and, where possible, time-heldout splits.
3. Encode original-language chunks with `StyleDistance/mstyledistance`. This is the current
   multilingual backbone. Compare the 2025 multilingual authorship-representation model as the
   main challenger for within-language and cross-domain recall; do not silently replace the
   reproducible baseline.
4. If adapting the backbone, use same-author, different-source positive pairs and in-batch
   negatives. Authors with fewer than two independent sources cannot enter fine-tuning.
5. Build both a normalized author centroid and source/work prototypes. Select single-centroid,
   prototype, or learned fusion only from identical held-out comparisons. Literary and rhetorical
   texts may be pooled while corpus provenance remains stored.
6. Encode topic separately with `intfloat/multilingual-e5-base`.
7. Cache chunk embeddings and representative original-language passages. At the current scale,
   use normalized NumPy matrix multiplication; add ANN/FAISS only after measured need.

### Online

1. Validate language and minimum length, then encode the query once per channel.
2. Retrieve profiles by exact matrix multiplication. Do not encode profiles or download models
   inside the request handler.
3. Within-language mode ranks profiles in the query language.
4. Cross-language mode compares original query text directly with original-language profiles.
   Do not translate the query by default. Return separate rankings by target language until
   ordered language-pair calibration makes a global ranking defensible.
5. Rerank only a small top-K with character n-grams, compression distance, function words,
   sentence-length distribution, punctuation, lexical diversity, hedging, and related
   interpretable features. These features complement rather than replace the encoder.

## Evaluation gates

The system is not production-ready until it reports:

- source-heldout Recall@3/5/20 and MRR overall and by language/corpus;
- cross-topic and cross-domain performance by language;
- open-set AUROC, equal-error rate, rejection thresholds, calibration, and selective risk;
- cross-language results by ordered language pair, without hiding failures in a macro average;
- a direct-original versus translation-mediated ablation, kept outside the headline score;
- warm p50/p95 latency on deployment hardware after at least 30 timed requests.

Source-heldout evaluation is necessary but insufficient: author and topic can remain correlated
across sources. Character models are topic-leakage ceilings, not production style models.
Translation is an experimental condition because translationese changes syntax, punctuation,
register, and language-specific distinctions. Raw multilingual cosine scores are not assumed
comparable across language pairs.

Latency targets: warm GPU p50 ≤ 500 ms and p95 ≤ 1.5 s for 75–500 word-equivalent inputs; CPU
fallback p95 ≤ 4 s. Measure model load separately. Use ONNX/quantization only when benchmarks
show an improvement on the deployment target.

## Corpus requirements

The registry is the author universe; source rows determine usable coverage. Keep author,
language, title, date, topic/domain, corpus, source URL/ID, independent-source ID, and
original-text status. Different mirrors or editions of one work must share an
`independent_source_id`. Every author
with a source may enter exploratory retrieval, but profiles with fewer than three independent
sources must be flagged as not source-heldout-ready and excluded from headline evaluation.
Only licence-approved sources may expose representative passages. Decade results require verified
source years, author-heldout validation, five authors and twenty sources per language/register/class.

The current multilingual expansion contains 45 primary literary texts: three authors × three
works in Chinese, Japanese, French, German, and Russian. It validates the pipeline, not broad
cross-language generalization. Modern rhetorical expansion requires separate verification of
speech authorship and delivered language.

## Implementation priorities

Must do: run the canonical multilingual notebook from source collection; preserve artifact hashes,
coverage tiers, model comparison, open-set calibration, ordered-pair evaluation, CPU/GPU latency,
and release-gate outputs. Do not publish when `private_beta_ready` is false.

Do not implement speculative v2 components. Explicit disentanglement models, interpretable
embedding objectives, per-author language models, and LLM reranking remain research directions;
LLMs may later explain or rerank but may never be the sole scorer.
