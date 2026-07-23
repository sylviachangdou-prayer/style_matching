# StyleMatch research and build record

This repository contains the reproducible corpus, evaluation, and model-build code for
StyleMatch, an open-set style-profile ranking system: given a passage, it ranks author
profiles with separate style and topic scores and permits "no strong match", rather than
pretending a closed-set classifier is a universal authorship test.

Registry snapshot (after the 2026-07b expansion): 274 author-language records — 179 literary
and 95 rhetorical. The registry is the author universe; usable coverage is computed from
actual source rows, never from registry membership.

## Adopted research decisions

- **Original language only.** The user passage is compared in its original language with
  original-language profiles. Translation is permitted only as a separately labelled
  ablation: translationese changes syntax, punctuation, honorifics, and register, so a
  translated passage never updates a profile or enters the direct style score.
- **Two physically separate channels.** Style: a contrastively fine-tuned multilingual
  authorship-representation encoder (`artifacts/multilingual_author_style_v1`, selected
  July 2026); topic/tone: `intfloat/multilingual-e5-base`, used only for the topic
  component. `StyleDistance/mstyledistance` remains the reproducible baseline of record.
- **Provisional weights** (product defaults, not empirical optima): 0.7 style + 0.3 topic
  within language; 0.5/0.5 across languages with a visible reduced-confidence label.
- **Open-set ranking plus verification.** Profiles are normalized author centroids plus
  source/work prototypes; retrieval is exact matrix multiplication over cached embeddings;
  rejection uses per-language calibrated thresholds.
- **No fusion without evidence.** Candidate methods are evaluated alone on identical
  source/topic/domain-heldout splits; a combined score is adopted only if it beats the best
  single method on held-out ranking, calibration, and selective risk.

## Model selection outcome (July 2026)

`scripts/compare_retrieval_models.py` compared six candidates on identical source-heldout
dev/test splits: pretrained/fine-tuned mStyleDistance, its source-prototype variant,
pretrained/fine-tuned multilingual authorship-representation challenger, and classical
style-feature fusion. Numeric report: `artifacts/model_comparison_v1.json` (Colab/Drive);
the decision is mirrored into both index `metadata.json` files.

- **Selected: the fine-tuned challenger** (`challenger_finetuned`) — best dev MRR among
  single candidates.
- **Learned fusion rejected** (`fusion_adopted: false`): it failed the adoption gates
  (paired-bootstrap MRR interval above zero, no calibration/selective-precision regression,
  subgroup non-inferiority). Kept on record as a negative result.
- **Deployed index: `artifacts/multilingual_style_index_challenger_v1`** (167 profiles,
  2,084 source prototypes, 39,756 chunks, 9 languages), `deployment_matches_selection: true`.
- The earlier mStyleDistance index (`multilingual_style_index_v1`) is retired from serving;
  it recorded `deployment_matches_selection: false` and near-chance open-set AUROC
  (0.43–0.55 for de/en/fr/ru) — a record of why encoder and calibration must move together.

## Current performance record (as of 2026-07-18)

Every number below is committed in this repository; provenance in parentheses. Numbers that
exist only in the Colab/Drive artifacts are listed as pending, not quoted from memory.

| Measurement | Value | Provenance |
| --- | --- | --- |
| Deployed index composition | 167 profiles · 2,084 source prototypes · 39,756 chunks · 9 languages | `multilingual_style_index_challenger_v1/metadata.json` |
| Score status | `uncalibrated_cosine` (ranking evidence, not probabilities) | same metadata |
| Warm query latency, CPU (retired baseline encoder) | p50 119.2 ms · p95 124.0 ms (30 runs, within-language) | `latency_cpu.json`, commit `15978e1` |
| Warm query latency, GPU (retired baseline encoder) | p50 25.7 ms · p95 26.9 ms (30 runs, within-language) | `latency_gpu.json`, commit `15978e1` |
| Open-set AUROC, retired baseline (mStyleDistance) | de 0.433 · en 0.506 · fr 0.466 · ru 0.545 — near chance | retired `multilingual_style_index_v1/metadata.json` |

Reading: latency clears the release targets (GPU p95 ≤ 1.5 s, CPU p95 ≤ 4 s) by a wide
margin, but was measured with the smaller retired encoder — the challenger must be
re-benchmarked. The near-chance baseline open-set AUROC is kept as a negative result: it is
why the deployed challenger index refuses calibration fitted on another encoder and why its
own calibration must be refit before open-set claims.

Pending (recorded in Drive artifacts, to be copied into the repo — runbook steps 4–6):
per-candidate source-heldout MRR/Recall from `model_comparison_v1.json` (decision known:
`challenger_finetuned` had the best dev MRR; fusion rejected), challenger per-language
open-set calibration, and challenger CPU/GPU latency.

## Open-set recalibration

The challenger index shipped with an empty `open_set_calibration`; rejection thresholds
must be refit on challenger scores before any headline open-set claim. No index rebuild or
re-encoding of the corpus is needed:

```bash
# per language (GPU, Colab): fit calibrator + EER threshold on author-heldout splits
python scripts/evaluate_open_set.py \
  --input data/all/meta/all_source_heldout_splits.parquet \
  --model-name artifacts/multilingual_author_style_v1 \
  --out-dir artifacts/open_set_eval_challenger_v1/<lang> \
  --language <lang> --device cuda

# install all fitted languages into the index metadata (CPU, seconds)
python scripts/multilingual_style_index.py calibrate \
  --index-dir artifacts/multilingual_style_index_challenger_v1 \
  --open-set-calibration-dir artifacts/open_set_eval_challenger_v1
```

`calibrate` refuses calibration fitted on a different encoder than the index (the failure
mode behind the retired baseline) unless `--allow-model-mismatch` is passed. Languages with
fewer than ten heldout profiles are skipped, not failed. Notebook section 5 automates the
loop over all index languages.

## Corpus registry and sources

Registry and source rules: every source keeps author, language, title, date, topic/domain,
corpus, source URL/ID, `independent_source_id` (mirrors/editions of one work share it), and
original-text status. No translations, adaptations, subtitles, summaries, or generated
imitation. Authors with fewer than three independent sources stay in exploratory retrieval
but are flagged not source-heldout-ready and excluded from headline evaluation. Only
licence-approved sources may expose representative passages. Decade output requires
verified source years.

Expansion batches (each verified against live holdings before entry): seed English
literary/rhetorical (Gutenberg); the 45-row multilingual literary catalog (zh/ja/fr/de/ru);
`expand_multilingual_2026_07` (es/it/pl plus more of the above); and
`expand_literary_2026_07b` — 33 authors added 2026-07-17, including Nobel laureates
(Yeats, Shaw, Galsworthy, Rolland, Gide, Hauptmann, Bunin, Benavente, Deledda, Ernaux)
and canonical prose authors across en/fr/de/ru/es/it/pl/zh, verified via the Gutendex and
Wikisource APIs.

**Rights-cleared in-copyright authors** (Annie Ernaux, Jian Zhen, Qiu Miaojin,
J. K. Rowling) use `source_format: local_text`: the owner places each cleared text at
`data/source_registry/raw_inputs/<author-slug>_<source-id>.txt` (the fetcher prints the
exact expected path when a file is missing). These rows are never fetched from the network,
default to `license_status: rights_cleared_private`, and never expose representative
passages (`display_allowed: false`).

Collection, import, and chunking:

```bash
python scripts/fetch_gutendex.py --corpus both --language en --max-works 0
python scripts/fetch_multilingual_sources.py \
  --language zh --language ja --language fr --language de --language ru \
  --language es --language it --language pl --skip-existing
python scripts/import_source_manifest.py data/source_registry/source_manifest.csv --append
python scripts/build_chunk_parquet_from_sources.py --corpus both \
  --output data/all/meta/all_sources_chunks.parquet \
  --coverage-output data/all/meta/all_sources_coverage.json
```

The chunk artifact keeps source text, covers both corpora, and uses language-aware chunking
for Chinese/Japanese.

## Expanding the index without retraining

New authors and sources do **not** require re-training the challenger. The encoder is
frozen; expansion is: collect sources → rebuild chunks → rebuild the index with the same
model. The build script caches chunk embeddings by `chunk_id`, so only new chunks are
encoded; it writes one parquet, not one file per chunk.

```bash
python scripts/multilingual_style_index.py build \
  --input data/all/meta/all_sources_chunks.parquet \
  --out-dir artifacts/multilingual_style_index_challenger_v1 \
  --model-name artifacts/multilingual_author_style_v1 \
  --topic-model-name intfloat/multilingual-e5-base \
  --embedding-cache artifacts/multilingual_style_index_challenger_v1/chunk_embeddings.npz \
  --topic-embedding-cache artifacts/multilingual_style_index_challenger_v1/topic_chunk_embeddings.npz \
  --model-label challenger_finetuned \
  --model-comparison artifacts/model_comparison_v1.json \
  --artifact-version challenger_v1 --device cuda
```

Because the profile set changes, re-run the open-set recalibration afterwards. Serving
loads the rebuilt index once at process start. The full step-by-step procedure for the
2026-07b batch — including the exact filenames for the twelve rights-cleared local texts —
is `docs/expansion_runbook_2026_07.md`.

Artifacts required downstream:

```text
data/all/meta/all_sources_chunks.parquet
artifacts/multilingual_author_style_v1/            # selected encoder (frozen)
artifacts/multilingual_style_index_challenger_v1/  # deployed index
```

## Evaluation record

Adopted protocol: source-heldout splits assign complete sources to train/dev/test (chunks
never cross splits), complemented by a leave-one-source-out rotation
(`scripts/evaluate_loso_retrieval.py`) in which every cached work of every multi-source
author serves once as the query set against a profile rebuilt from the remaining works.
Source-heldout evaluation is necessary but insufficient: author and topic can stay
correlated across sources, so cross-topic/cross-domain results are reported separately.
Character models are treated as topic-leakage ceilings, not production style models. Raw
multilingual cosines are not assumed comparable across language pairs; cross-language
results are reported per ordered pair.

Explored and recorded (post-baseline method-exploration notebook, outputs under
`artifacts/method_exploration_v1/`; none silently adopted):

- matched candidate-pool and chance-adjusted language comparisons;
- independent-source profile-evidence curves at a fixed encoder;
- pinned pretrained backbone comparisons plus an English-only LUAR diagnostic;
- delexicalized character, function-word, rhythm/discourse, compression, UPOS/dependency,
  centroid, and source-prototype views;
- an elastic-net candidate reranker with paired test bootstrap, calibration gates, subgroup
  non-inferiority, and leave-one-view-out ablations — fusion remained rejected;
- probabilistic-content-masking fine-tuning ablation (authors' defaults: protect the 300
  most frequent subwords, mask at 0.4, one epoch);
- single-centroid versus source-prototype retrieval — `single_centroid` remains deployed.

### Expanded-corpus neural reranking (notebook Part 8)

Part 8 evaluates an evidence-gated neural reranker after the expanded independent-source
artifact is frozen. It reuses finished encoder and score artifacts, so reranker experiments do
not repeat embedding batches. Candidate evidence remains decomposed: fine-tuned and pretrained
author centroids, work/source prototypes, delexicalized character patterns, function words,
rhythm/discourse, compression distance, and candidate-wise score stability across source
chunks. Topic scores are prohibited.

The source split is binding: encoder and classical fitting use train only; reranker selection
uses author-language-profile-grouped cross-validation within dev; test is opened once (Cawley
& Talbot 2010). Three deliberately small candidates are compared: a global listwise mixture,
a gated listwise network, and a gated hybrid with anchor-selected hard-negative pairwise loss.
Each predicts only a bounded residual around the fine-tuned centroid score. The selected
configuration is refit on all dev sources for the median cross-validated stopping epoch.
Out-of-fold dev predictions, rather than fitted dev predictions, calibrate test confidence.
The reranker is adopted only when its paired profile-bootstrap MRR interval is wholly positive,
calibration and selective precision do not regress, and supported language/corpus groups do
not worsen (Dror et al. 2018; Guo et al. 2017; Geifman & El-Yaniv 2017). Reinforcement learning
is not used: the project has no online interaction reward, and policy-gradient optimization on
the finite dev set would amplify selection noise rather than add stylistic evidence.

Diachronic output: optional `language × decade` centroids from sources with verified years
only; never inferred from lifespan or edition dates; uncalibrated until decade-heldout
evaluation exists; unavailable rather than blocking when support is insufficient.

Release gates (all must be reported before production): source-heldout Recall@3/5/20 and
MRR overall and by language/corpus; cross-topic/cross-domain by language; open-set AUROC,
EER, rejection thresholds, calibration, selective risk; ordered language-pair results;
a direct-original versus translation-mediated ablation outside the headline score; warm
p50/p95 latency (targets: GPU p50 ≤ 500 ms, p95 ≤ 1.5 s; CPU p95 ≤ 4 s), model load
measured separately. Gate commands: `scripts/audit_source_metadata.py`,
`scripts/audit_release_readiness.py --strict`, `scripts/check_release_gates.py`
(see notebook section 4 for the exact invocations). Do not publish while
`private_beta_ready` is false; `scripts/publish_hf_artifacts.py` stays dry-run without
`--publish`.

## Fine-tuning scope (record)

There is no from-scratch language-model training, and expansion does not trigger
re-training. The one-epoch contrastive adaptation (`scripts/finetune_multilingual_style.py`)
that produced the selected encoder formed positives from different works by the same author
in the same language (blocking the book-identity shortcut), used matched hard negatives and
language-aware batches, excluded authors with fewer than two independent sources, and never
used machine translation. `--audit-only` reconciles registry profiles against the chunk
artifact before any GPU run. Profile construction caps: 50 chunks per source, 600 per
profile.

## Literature record, 1998–2026

- Kittler et al., IEEE TPAMI 1998 — theoretical framework for combining distinct classifier
  representations and the linear sum rule. [DOI](https://doi.org/10.1109/34.667881)
- Montague & Aslam, CIKM 2001 — normalization before score-based metasearch fusion; basis
  for putting heterogeneous retrieval views on a common scale.
  [DOI](https://doi.org/10.1145/502585.502657)
- Cameron, Gelbach & Miller, *Review of Economics and Statistics* 2008 — bootstrap inference
  under within-cluster dependence; basis for resampling profiles rather than dependent source
  rows. [DOI](https://doi.org/10.1162/rest.90.3.414)
- Cawley & Talbot, JMLR 2010 — model-selection overfitting and subsequent selection bias;
  basis for dev-only weight selection and a locked test.
  [Paper](https://www.jmlr.org/papers/v11/cawley10a.html)
- Guo et al., ICML 2017 — calibration and expected calibration error as a distinct evaluation
  axis. [Paper](https://proceedings.mlr.press/v70/guo17a.html)
- Geifman & El-Yaniv, NeurIPS 2017 — selective prediction and the risk–coverage trade-off;
  basis for the 50%-coverage precision gate.
  [Paper](https://proceedings.neurips.cc/paper/2017/hash/4a8423d5e91fda00bb7e46540e2b0cf1-Abstract.html)
- Dror et al., ACL 2018 — task-appropriate paired significance testing in NLP.
  [Paper](https://aclanthology.org/P18-1128/)
- Sagawa et al., ICLR 2020 — worst-group robustness under group shifts; basis for the
  language/corpus non-degradation constraint.
  [Paper](https://openreview.net/forum?id=ryxGuJrFvS)
- Sawatphol et al., TACL 2024 — topic leakage in cross-topic authorship evaluation.
- Huertas-Tato et al. 2024 — hard-negative contrastive isolation of style from content.
- Terreau et al. 2024 — interpretable stylistic features as a complementary channel.
- Wang et al. 2024 — multilingual E5; basis of the separate topic channel.
- Qiu et al., Findings ACL 2025 — mStyleDistance, the baseline backbone.
- Kim, Zhang & Jurgens, EMNLP 2025 — multilingual authorship representation (content
  masking, language-aware batching); basis of the selected challenger.
- Alshomary et al., EMNLP 2025 — multi-layer style representations.
- Alshomary et al., COLING 2025 — prototype-based latent-space interpretation.
- Alipoormolabashi et al., ACL 2025 — misattribution unfairness (MAUI) metric.
- Icard et al., COLING 2025 — embedding dispersion and cross-language score risk.
- Man et al., ACL 2026 — disentangled representations; future direction, not substituted.
- Anand et al., EACL 2026 — interpretable sense-decomposed embeddings; challenger only.

Full decision memo: `docs/frontier_multilingual_strategy_2026.md`. Model, data, and
evaluation-report templates: `docs/`.

## Verification

```bash
python -m compileall scripts
python -m pytest -q
```

GPU runs belong in Colab (`colab_stylematch_multilingual_training.ipynb` from the repo
root) and must record model name, device, corpus commit, coverage JSON, index metadata, and
warm p50/p95 benchmarks in the notebook output.
