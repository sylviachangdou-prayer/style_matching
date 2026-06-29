# CLAUDE.md — StyleMatch

A web app for the general public that takes a passage of text and returns the author/speaker
whose writing style + thematic register it most resembles, with a transparent, defensible
breakdown of *why*. Fun on the surface, academically rigorous underneath.

The core promise: every match is a *real* style match, not a topic-leakage artifact. We would
rather show an honest "low confidence / not much like anyone" than a fake-precise label.

---

## 1. Product goals (read first, these constrain everything)

1. **Mass-appeal, playful UX** — "Which famous voice do you write like?" Result is a ranked
   top-3 with a headline match, a similarity score, and a human-readable explanation.
2. **Frontier-accurate underneath** — contrastive authorship embeddings (LUAR / StyleDistance
   family) for recall, classical stylometry + verification for rerank, strict cross-topic
   evaluation so the numbers are real.
3. **Two corpora at launch** (scientific/report corpus is explicitly CUT):
   - `rhetorical` — speeches & first-hand spoken/written rhetoric from major figures
     (presidents, popes, Nobel lectures, famous TED scripts). PRIMARY corpus: cleanest data,
     strongest style signal, highest public resonance, mostly public domain.
   - `literary` — prose by manually approved authors, stored only in the author's original language.
4. **Two matching modes**, surfaced honestly in the UI:
   - `within-language` (default, the headline experience): real style matching. We *do* claim
     "your writing style resembles X."
   - `cross-language` (exploratory): we claim "your themes & tone — and, more weakly, your
     style — most resemble X." Style signal is real but weaker across languages; the UI must
     show reduced confidence, never pretend parity with within-language.

NON-GOALS for v1: scientific-text corpus, per-author ALM/perplexity scoring, video. Keep these
out of v1 scope; leave clean extension points.

---

## 2. The output: a composite Affinity Score that is always decomposable

Never collapse everything into one opaque number. Compute a weighted composite but ALWAYS
expose the sub-scores and the explanation underneath.

```
AffinityScore = w_style * StyleSim + w_topic * TopicSim
```

- **within-language:** `w_style = 0.7, w_topic = 0.3` (style leads — this is the real promise)
- **cross-language:** `w_style = 0.5, w_topic = 0.5` AND attach a `confidence` flag = "reduced"
  because cross-lingual style signal is known to decay. Do not hide this.

UI must render, for the top match and on-demand for runners-up:
- the composite Affinity Score,
- the StyleSim and TopicSim sub-scores separately,
- the interpretable "why" features (see §5),
- 1–3 passages of the matched author **in their original language**, chosen as the most
  representative of the matched style (highest per-chunk style similarity).

If top-1 AffinityScore is below a calibrated threshold, show an honest low-confidence state
("Your writing doesn't strongly resemble anyone in our library yet") rather than forcing a label.

---

## 3. Architecture: two-stage recall → rerank (this is the spine)

Framed as open-world authorship verification, NOT closed-set classification. Underlying logic is
pairwise "same-author probability"; the UI presents a friendly top-3 ranking on top of it.

### Stage 0 — Preprocessing
- language ID (per chunk),
- strip boilerplate / quotes / code blocks / markup,
- chunk into short, register-consistent passages (target ~75–150 words/chunk),
- discard chunks too short to carry style signal.

### Stage 1 — Recall (style embeddings)
- Backbone: **LUAR** (HF pretrained authorship-representation weights) for within-language.
  Evaluate **StyleDistance / mStyleDistance** as an alternative / for cross-language (9 langs).
  Do NOT use a generic semantic embedding as the style backbone — it leaks topic.
- Build each author a **profile centroid** from many multi-topic chunks.
- Encode user input, nearest-neighbor recall **top-K = 20** author candidates by style.

### Stage 2 — Rerank (ensemble, conservative)
Fuse, per candidate:
- LUAR / style-embedding cosine similarity,
- char n-gram (tetragram) TF-IDF similarity,
- compression / PPM distance,
- a handful of interpretable stylometric features (see §5),
- TopicSim from a **separate semantic channel** (see §4).

Output calibrated top-3 with composite + sub-scores. Keep the ensemble weights in a config file
so they're tunable without code changes.

**v2 hooks (build the seam, don't implement):** per-author ALM perplexity reranker; LLM as
reranker/explanation-generator (never as sole scorer — LLMs get pulled off by semantics).

---

## 4. Two channels: STYLE vs TOPIC must stay physically separate

This separation is the whole reason the product is honest. Do not let one model produce both.

- **Style channel:** LUAR / StyleDistance embeddings → `StyleSim`.
- **Topic channel:** a multilingual *semantic* embedding (e.g. multilingual-e5 or LaBSE) →
  `TopicSim`. Captures subject matter & tonal register, language-agnostic.

Within-language: both channels run in-language.
Cross-language: topic channel is naturally multilingual; style channel uses mStyleDistance and is
flagged reduced-confidence. Always store and display the two numbers independently before combining.

---

## 5. Interpretable explanation layer (the soul of the mass-market UX)

A black-box "you're like Churchill" convinces no one. Every match ships with concrete, readable
reasons computed from classical features, e.g.:
- average sentence length / sentence-length variance,
- passive-voice rate,
- abstract-noun / nominalization density,
- function-word profile (the Burrows'-Delta family signal),
- hedging density, punctuation habits, type-token ratio.

Render 2–4 of the most *distinguishing* features as plain sentences:
"Like Obama: long sentences, frequent first-person plural, low hedging."
Pick the features where the user deviates most from the global mean toward the matched author.

---

## 6. Evaluation: cross-topic is mandatory and non-negotiable

Topic leakage is the #1 way this kind of system lies. A model that looks 90% accurate is often
secretly matching *subject*, not *style* — and mass users will catch it ("I wrote about food and
it said I write like a food writer").

- **Splits:** cross-topic, cross-domain, and cross-time. Never random split.
- **Metrics:** top-1 accuracy, MRR, calibration (reliability curve), and robustness across the
  splits above. Report all, not a single accuracy number.
- Maintain a held-out cross-topic test set per corpus from day one.
- Cross-language eval needs genuinely multilingual+multi-domain data (cf. MAC-style benchmarks);
  do not let topic masquerade as author across languages.

A drop in accuracy under cross-topic splitting is EXPECTED and CORRECT. Honest lower numbers are
the moat. Do not tune against random-split accuracy.

---

## 7. Corpora: purity over coverage

Do NOT chase "every author in the world." Each author needs MULTIPLE multi-topic samples or they
can't be evaluated cross-topic.

- `rhetorical`: start 30–50 figures with abundant first-hand originals (presidents, popes, Nobel
  lectures, famous TED scripts). Mostly public domain.
- `literary`: start 20–30 manually approved authors.
- Store raw text + metadata (author, original language, date, topic-tag, source, original-text flag) so cross-topic
  and cross-time splits are constructible.
- Per-language buckets. A user writing in language L matches authors who wrote in L (within-lang).

Source hygiene: ingest only original-language source text. Do not ingest translations, adapted
transcripts, subtitles, paraphrases, summaries, or LLM-generated imitation text.

---

## 8. Cross-language mode (do build it, but label it honestly)

- Style: mStyleDistance (multilingual style embeddings, ~9 langs). Real but weaker across langs.
- Topic: multilingual-e5 / LaBSE.
- UI copy switches from "your style resembles X" to "your themes & tone — and more weakly your
  style — most resemble X," with a visible reduced-confidence indicator and the 0.5/0.5 weights.
- Matched passages still shown in the author's ORIGINAL language.

---

## 9. GPU budget (≈10h, single Colab/M5-class GPU)

NOT for from-scratch training. Spend it on, in priority order:
1. Encode all corpus chunks and build per-author profile centroids.
2. Light **contrastive domain-adaptation** of the style backbone on our two corpora (adapt LUAR/
   StyleDistance to speech-register and literary-register) — the highest-value few hours.
3. Fit & calibrate the rerank ensemble + decision thresholds on the cross-topic dev set.

Translation models / per-author ALMs are the LOWEST-value use of this budget. Skip in v1.

---

## 10. Suggested stack & layout

- Python backend (FastAPI). HF `transformers` + `sentence-transformers`. PyTorch.
- Vector recall: FAISS (or in-memory cosine at this scale).
- Frontend: lightweight (React or plain). Mobile-friendly. Show score + sub-scores + why +
  original-language passages.
- Config-driven ensemble weights & thresholds (YAML).

```
/data/{rhetorical,literary}/{raw,chunks,meta}/
/src/preprocess/      # langID, boilerplate strip, chunking
/src/embed/           # luar, styledistance, semantic (topic) channels
/src/recall/          # centroids, FAISS knn
/src/rerank/          # char-ngram, compression, feature ensemble, calibration
/src/explain/         # interpretable stylometric features -> sentences
/src/score/           # composite Affinity (weights, confidence flag)
/src/eval/            # cross-topic/-domain/-time splits, metrics
/api/                 # FastAPI
/web/                 # frontend
/config/weights.yaml
```

---

## 11. Build order (validate experience before scaling)

1. **Rhetorical, within-language (English), recall-only.** LUAR pretrained → centroids → top-3 +
   show matched passages. Get the *feel* right first.
2. Add interpretable explanation layer (§5) — biggest perceived-quality jump for users.
3. Add rerank ensemble (§3 Stage 2) + topic channel (§4) + composite Affinity (§2).
4. Add cross-topic evaluation harness (§6); tune thresholds; add honest low-confidence state.
5. Add literary corpus. Then cross-language mode (§8) as exploratory.

Prefer running a real version early over speccing further — many tuning decisions only make sense
once you can see actual matches.

---

## 12. Hard rules

- Style channel and topic channel stay separate; never derive both from one generic embedding.
- Always expose sub-scores + reasons; never ship a lone opaque number.
- Evaluate cross-topic only; never report random-split accuracy as the headline.
- Cross-language results are reduced-confidence and must be labeled as such.
- Original-language corpora only; do not train on translations or adapted transcripts.
- LLMs (if used later) are rerankers/explainers, never the sole scorer.
- Honest "no strong match" beats a fake-precise label.
