# StyleMatch

StyleMatch is a multilingual research prototype for ranked literary-style retrieval. It
retrieves author-language profiles with similar writing patterns while keeping topic
similarity separate. This non-commercial PhD-application portfolio demonstrates NLP,
stylometry, evaluation, and research software—not forensic authorship attribution.

[Live demo](https://sylviachangdou-prayer-stylematch.hf.space) ·
[model v2](https://huggingface.co/sylviachangdou-prayer/stylematch-authorship/tree/v2) ·
[index v2](https://huggingface.co/datasets/sylviachangdou-prayer/stylematch-index/tree/v2) ·
[model card](docs/model_card.md) · [data card](docs/data_card.md)

## Research task

The task is retrieval, not closed-set identification. A query passage and each reference
profile are embedded in a shared multilingual style space; candidates are ranked by stylistic
similarity. Topic/tone is computed by another encoder and cannot change the Style Match order.
Because the deployed scores are not calibrated probabilities, a high score means “nearer in
this candidate geometry,” not “this author wrote the passage.” Formal open-set rejection
remains unavailable in v2.

The current release pairs the frozen
[`stylematch-authorship:v2`](https://huggingface.co/sylviachangdou-prayer/stylematch-authorship/tree/v2)
encoder with [`stylematch-index:v2`](https://huggingface.co/datasets/sylviachangdou-prayer/stylematch-index/tree/v2).
The model tag documents the v2 deployment but uses the same encoder weights as model v1; the
new behavior comes from expanded evidence and the ranking backend.

## Methodology

### 1. Corpus and evidence units

Profiles use original-language primary texts with work identity, register, date when known,
provenance, rights, and display permission. Mirrors, editions, or chunks from one work share
an `independent_source_id`; translations, adaptations, summaries, and generated imitations
do not update profiles.

Source-level splits keep complete works in train, dev, or test; chunks from one source never
cross a split. Headline ranking is macro-aggregated by author-language profile so prolific
authors do not dominate evaluation. Topic-, register-, time-, and author-heldout tests are
separate stress tests rather than substitutes for source-heldout evaluation. Only
rights-approved sources may supply a displayed representative passage.

### 2. Style representation

The 1024-dimensional encoder is a fine-tuned multilingual authorship representation model.
Contrastive positives come from different sources by the same author and language, blocking
direct work memorization. Hard negatives prioritize similar topic, register, and period in
the same language. Training uses language-aware batches and no machine translation.

The encoder is trained once; adding an author does not require another model run. New texts
are encoded, source prototypes and author centroids are rebuilt, and the ranking calibration
is refitted because changing the candidate pool changes its geometry.

### 3. v2 ranking backend

Raw cosine similarity produced dense “hub” profiles that appeared too often in false Top-3
results. V2 therefore performs target-language-specific shrinkage whitening before ranking.
For language \(l\), source prototypes estimate mean \(\mu_l\) and covariance \(\Sigma_l\).
The covariance is shrunk 0.30 toward an isotropic target before inverse-square-root
transformation and L2 normalization. This reduces shared high-variance directions without
retraining the encoder.

For candidate \(a\), an exposure prior \(p_a\) is the within-language percentile of its
source-balanced false Top-3 frequency: each independent source acts as a query, its own author
is excluded, and every source has equal total weight. The deployed score is

\[
s(a\mid x,l)=\cos\!\left(W_l(E(x)-\mu_l),c_a\right)-0.01p_a,
\]

where \(E(x)\) is the query embedding, \(W_l\) is the whitening transform, and \(c_a\) is the
whitened author centroid. The correction is generic; no author name is hard-coded. The 0.01
penalty was selected on dev under retrieval non-inferiority, subgroup non-degradation, and
lower-concentration requirements.

Whitening and exposure priors are refitted over the complete production source-prototype
bank. This matches the selected scoring structure but is an extrapolation beyond the narrower
locked diagnostic candidate pool, so the release remains a beta rather than claiming that
the diagnostic numbers are direct production estimates.

### 4. Separate contextual and explanatory signals

`intfloat/multilingual-e5-base` supplies topic/tone similarity. It may enter a separate
Affinity statistic but never Style Match rank. Representative passages are selected after
ranking from display-approved sources and trimmed to complete sentences. Biographies and
style traits are registry metadata, not predictions.

Cross-language queries are transformed against each target language's geometry, but ordered
language-pair calibration is incomplete; cross-language scores remain exploratory and should
be compared within a target-language group. “Closest decade” is a separate prototype task
using reliably dated sources and never changes author ranking.

## Model and backend selection

All encoder candidates below used the same locked independent-source test and candidate pool.
Intervals are profile-bootstrap 95% intervals. The learned reranker was the numerical leader,
but its MRR gain over the fine-tuned authorship representation was 0.0032 with interval
[-0.0524, 0.0599], and it violated the subgroup non-degradation gate. The simpler encoder was
therefore selected.

| Representation | MRR (95% interval) | Recall@3 |
| --- | ---: | ---: |
| Learned multi-view reranker | 0.785 [0.724, 0.843] | 0.867 |
| **Fine-tuned authorship representation — selected encoder** | **0.782 [0.721, 0.839]** | **0.858** |
| Classical style-feature fusion | 0.731 [0.666, 0.792] | 0.808 |
| Pretrained authorship representation | 0.609 [0.537, 0.678] | 0.700 |
| Fine-tuned mStyleDistance | 0.378 [0.311, 0.447] | 0.433 |
| Pretrained mStyleDistance | 0.334 [0.274, 0.398] | 0.383 |
| mStyleDistance with source prototypes | 0.331 [0.268, 0.401] | 0.350 |

Evidence: [method performance](docs/method_performance.json) and
[fusion diagnostics](docs/multiview_fusion_metrics.json).

The backend experiment then held the encoder fixed. Dev selection chose
`whitened cosine (0.30) + exposure prior (0.01)`. On the locked diagnostic test, it reduced
concentration while leaving retrieval effectively flat:

| Backend | MRR (95% interval) | Recall@3 (95% interval) | False-Top3 Gini | Source-balanced MAUI@3 |
| --- | ---: | ---: | ---: | ---: |
| Whitened cosine | 0.500 [0.466, 0.535] | 0.579 [0.539, 0.620] | 0.230 | 0.0728 |
| **+ exposure prior — deployed** | **0.501 [0.467, 0.535]** | **0.578 [0.537, 0.619]** | **0.214** | **0.0639** |

The Recall@3 change is -0.0015, inside the predeclared -0.01 tolerance; the evidence supports
a concentration reduction, not an accuracy improvement. See the
[dev search](docs/postwhitening_dev_search.csv),
[test diagnostic](docs/postwhitening_test_diagnostic.csv), and
[subgroup results](docs/postwhitening_subgroup_metrics.csv).

## Limitations

- Similarity is not authorship probability, identity verification, or plagiarism detection.
- V2 has no accepted open-set probability calibration or universal “no match” threshold.
- Style and content remain statistically entangled despite source separation and hard negatives.
- Language and register support is uneven; small subgroups yield unstable estimates.
- Cross-language ranking lacks ordered-pair calibration, and decade matching remains experimental.
- Public-domain availability overrepresents particular periods, genres, and institutions.
- Expanding the candidate pool improves coverage but makes the ranking problem harder; scores and
  ranks are version-specific and should not be compared across index releases.

## Reproduction

The notebooks are the experiment entry points:

| Notebook | Purpose |
| --- | --- |
| `multilingual_training.ipynb` | corpus audit, source-heldout split, encoder training, index build |
| `method_exploration.ipynb` | neural and classical representation comparison |
| `ranking_backends.ipynb` | whitening, metric-learning, normalization, and concentration tests |
| `hubness_reranking.ipynb` | source-balanced false-return and hubness diagnostics |
| `expand_sources.ipynb` | targeted source expansion and coverage audit |

Run the regression suite locally:

```bash
python -m pytest -q
```

Rebuild the v2 scoring artifact from a completed expanded index:

```bash
python scripts/build_postwhitening_exposure_index.py \
  --index-dir artifacts/multilingual_style_index_gutenberg_v3 \
  --output-dir artifacts/multilingual_style_index_v2 \
  --shrinkage 0.3 --penalty 0.01 \
  --model-name sylviachangdou-prayer/stylematch-authorship \
  --model-revision v2
```

Randomized experiments report seeds. Raw data are never overwritten; derived datasets,
models, and indices use separate paths. Frozen artifacts are the source of quantitative claims.

## Selected research foundations

- Reimers & Gurevych (2019), [Sentence-BERT](https://aclanthology.org/D19-1410/).
- Cawley & Talbot (2010), [model-selection overfitting](https://www.jmlr.org/papers/v11/cawley10a.html).
- Guo et al. (2017), [neural-network calibration](https://proceedings.mlr.press/v70/guo17a.html).
- Dror et al. (2018), [statistical testing for NLP](https://aclanthology.org/P18-1128/).
- Sawatphol et al. (2024), [topic leakage in authorship attribution](https://aclanthology.org/2024.tacl-1.75/).
- Huertas-Tato et al. (2024), [contrastive isolation of authorship from content](https://arxiv.org/abs/2411.18472).
- Qiu et al. (2025), [mStyleDistance](https://aclanthology.org/2025.findings-acl.869/).
- Kim, Zhang & Jurgens (2025), [multilingual authorship representation](https://aclanthology.org/2025.emnlp-main.1766/).
- Habler et al. (2026), [robust hubness diagnostics](https://arxiv.org/abs/2602.22427).

The [evidence ledger](docs/method_evidence_ledger.md) states the scope and status of every
current quantitative claim.
