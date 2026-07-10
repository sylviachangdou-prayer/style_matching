# StyleMatch research and build record

This repository contains the reproducible corpus, evaluation, and model-build code for
StyleMatch. The product is an open-set style-profile ranking system: it returns a ranked
match with separate style and topic scores, rather than pretending that a closed-set author
classifier is a universal authorship test.

Registry snapshot: 163 author-language records, representing 160 unique names; 79 literary
records and 84 rhetorical records. The registry is the author universe; source availability is
computed from actual source rows, not treated as an author class. The 45-row multilingual catalog
is only one source expansion batch.

## Current research decision

The default path compares the user passage in its original language with original-language
source profiles. It does not translate a user passage into every corpus language. Translation
is permitted only as a separately labelled ablation or topic experiment. A translated passage
must never update a style profile or be pooled into the direct style score: translationese can
change syntax, punctuation, honorifics, and register, which would make the result partly a
measurement of the translator.

The two channels are physically separate:

- Style: `StyleDistance/mstyledistance`, optionally fine-tuned on source-separated author pairs.
- Topic/tone: `intfloat/multilingual-e5-base`, used only for the topic component of the Affinity
  score.
- Within-language weighting: 0.7 style + 0.3 topic.
- Cross-language weighting: 0.5 style + 0.5 topic, with a visible reduced-confidence label.

The offline index stores one normalized centroid per language/author profile and three
representative original-language passages. Literary and rhetorical sources are pooled inside
that profile; their corpus provenance remains in metadata. Online inference loads this index once, encodes the
user passage once, and performs a matrix multiplication. The build script caches chunk
embeddings by `chunk_id`, so a reconnect or a later corpus expansion does not re-encode the
whole corpus. It writes one parquet file, not one file per chunk.

## Original-language source policy

No LLM-generated text, translation, adaptation, subtitle, summary, or imitation text is used.
The multilingual literary catalog is in
`data/source_registry/multilingual_source_catalog.csv`; it currently covers Chinese, Japanese,
French, German, and Russian originals, alongside the existing English sources. The catalog
records author, language, title, source ID, and source URL. The fetcher downloads and validates
the declared script before creating `source_manifest.csv`.
The 45 catalog rows are one multilingual expansion batch, not the total project author count.

Run source collection in Colab from the repository root:

```bash
python scripts/fetch_gutendex.py \
  --corpus both --language en \
  --max-works 0
python scripts/fetch_multilingual_sources.py \
  --language zh --language ja --language fr --language de --language ru \
  --skip-existing
python scripts/import_source_manifest.py data/source_registry/source_manifest.csv --append
python scripts/build_chunk_parquet_from_sources.py \
  --corpus both \
  --output data/all/meta/all_sources_chunks.parquet \
  --coverage-output data/all/meta/all_sources_coverage.json
```

The all-source chunk artifact is `data/all/meta/all_sources_chunks.parquet`. It retains source
text in a `text` column for evaluation, contains both `literary` and `rhetorical` rows, and uses
language-aware chunking for Chinese/Japanese. Every author with at least one source
enters the profile index. Authors with fewer than three independent sources remain available
for exploratory retrieval but are flagged as not source-heldout-ready and must not inflate the
headline source-heldout evaluation.
The coverage JSON records n_author_language_profiles, n_sources, and source counts by
language and corpus. The heldout report records the same profile/source counts plus the
source assignment for each eligible profile.

## Colab GPU workflow

Open `colab_stylematch_multilingual_training.ipynb` from the cloned `style_matching` repo in
Colab. Set the runtime to a GPU before running the notebook. The notebook:

1. mounts `/content/drive/MyDrive/style_matching`;
2. queries Gutenberg for every registry author in both corpora, then fetches the curated
   multilingual catalog with reconnect-safe `--skip-existing`;
3. imports sources and builds one combined parquet chunk artifact;
4. fine-tunes mStyleDistance with same-author/different-source positive pairs and in-batch
   negatives;
5. builds a cached style + topic profile index;
6. benchmarks warm query latency and prints a sample result.

The artifacts needed by the future web app are:

```text
data/all/meta/all_sources_chunks.parquet
artifacts/mstyledistance_stylematch_v1/
artifacts/multilingual_style_index_v1/
```

The notebook also writes artifacts/source_heldout_eval_v1/; its metrics include overall,
per-language, and per-corpus source-heldout results. The heldout split assigns complete
sources to train, dev, or test, so chunks from one source never cross those splits.

The web app should load the index once at process start. It must not download models, translate
the request, or encode every author profile inside a request handler. Benchmark model loading
separately from warm request latency; the current target is p50 <= 500 ms and p95 <= 1.5 s on a
warm GPU, with a CPU fallback target of p95 <= 4 s.

## Training and fine-tuning scope

There is no from-scratch language-model training. Colab GPU is used for:

1. corpus encoding and cached profile construction;
2. one-epoch contrastive adaptation in `scripts/finetune_multilingual_style.py`;
3. optional language-pair calibration and reranker fitting after the robust splits exist.

The fine-tuning script deliberately forms positives from different works by the same author in
the same language; literary and rhetorical works can both contribute to that author profile.
This prevents the easiest book-identity shortcut. It excludes authors without at least two
sources from the fine-tuning objective while keeping their raw sources available for indexing.
The default training path never uses machine translation. Profile construction samples at most
50 chunks per source and 600 chunks per author-language profile, while source collection itself
has no work-count cap.

## Literature record, 2023-2026

- Sawatphol, Udomcharoenchaikit, and Nutanong (TACL 2024), [Addressing Topic Leakage in
  Cross-Topic Evaluation for Authorship Verification](https://aclanthology.org/2024.tacl-1.75/).
  Basis for HITS/RAVEN-style topic-shortcut evaluation and the requirement to report robust
  cross-topic results.
- Huertas-Tato et al. (2024), [Isolating authorship from content with semantic embeddings and
  contrastive learning](https://arxiv.org/abs/2411.18472). Basis for hard-negative contrastive
  adaptation and keeping content separate from style.
- Terreau, Gourru, and Velcin (2024), [Capturing Style in Author and Document
  Representation](https://arxiv.org/abs/2407.13358). Basis for interpretable stylistic
  features as a complementary reranking channel, not a replacement for the encoder.
- Wang et al. (2024), [Multilingual E5 Text Embeddings: A Technical
  Report](https://arxiv.org/abs/2402.05672). Basis for the separate multilingual semantic/topic
  channel; the model card is [multilingual-e5-base](https://huggingface.co/intfloat/multilingual-e5-base).
- Qiu et al. (Findings ACL 2025), [mStyleDistance: Multilingual Style Embeddings and their
  Evaluation](https://aclanthology.org/2025.findings-acl.869/). Primary style backbone for
  the six-language exploratory index; model card:
  [StyleDistance/mstyledistance](https://huggingface.co/StyleDistance/mstyledistance).
- Kim, Zhang, and Jurgens (EMNLP 2025), [Leveraging Multilingual Training for Authorship
  Representation](https://aclanthology.org/2025.emnlp-main.1766/). Supports content masking,
  language-aware batching, and multilingual/domain-heldout evaluation.
- Icard et al. (COLING 2025), [Embedding Style Beyond Topics: Analyzing Dispersion Effects
  Across Different Language Models](https://aclanthology.org/2025.coling-main.236/). Basis for
  treating embedding dispersion and language-pair score calibration as measurable risks.
- Man et al. (ACL 2026), [Explainable Disentangled Representation Learning for Generalizable
  Authorship Attribution in the Era of Generative AI](https://aclanthology.org/2026.acl-long.2018/).
  Research direction for a later explicit style/content VAE and explanation layer; it is not
  silently substituted for the current reproducible baseline.

The longer decision memo, including translation-mediated ablation rules, latency contract,
and evaluation gates, is in `docs/frontier_multilingual_strategy_2026.md`.

## Web app

All web code lives in `web/` (FastAPI backend for a Hugging Face Docker Space, static
frontend for Vercel, demo fallback while the index is still training). See `web/README.md`
for local run and deployment steps. `python -m pytest web/tests -q` runs without models.

## Verification

Run local checks that do not require model downloads:

```bash
python -m compileall scripts
python -m pytest -q
```

GPU model runs belong in Colab and should record the printed model name, device, corpus commit,
coverage JSON, index metadata, and warm p50/p95 benchmark in the notebook output.
