# Expansion runbook — 2026-07b batch (no re-training)

Goal: bring the 33 new authors online with the **frozen** challenger encoder
(`artifacts/multilingual_author_style_v1`). Do **not** re-run fine-tuning; nothing in this
runbook touches model weights. Total GPU time is dominated by encoding only the *new*
chunks (the build reuses the `chunk_id` embedding cache) and the open-set refit.

## Step 0 — place the 12 rights-cleared files (local machine or Drive)

Plain UTF-8 text, one file per work, body text only (strip publisher front/back matter,
page headers, and any translator or editor apparatus). Put them in
`data/source_registry/raw_inputs/` under **exactly** these names:

```text
annie_ernaux_local_ernaux_la_place.txt          La Place (1983)
annie_ernaux_local_ernaux_une_femme.txt         Une femme (1988)
annie_ernaux_local_ernaux_les_annees.txt        Les Années (2008)
jian_zhen_local_jianzhen_shuiwen.txt            水問 (1985)
jian_zhen_local_jianzhen_nuerhong.txt           女兒紅 (1996)
jian_zhen_local_jianzhen_tianyahaijiao.txt      天涯海角 (2002)
qiu_miaojin_local_qiumiaojin_eyushouji.txt      鱷魚手記 (1994)
qiu_miaojin_local_qiumiaojin_jimodequnzhong.txt 寂寞的群眾 (1995)
qiu_miaojin_local_qiumiaojin_mengmateyishu.txt  蒙馬特遺書 (1996)
j_k_rowling_local_rowling_philosophers_stone.txt   HP 1 (1997)
j_k_rowling_local_rowling_chamber_of_secrets.txt   HP 2 (1998)
j_k_rowling_local_rowling_prisoner_of_azkaban.txt  HP 3 (1999)
```

If a name is wrong or a file missing, Step 1 prints the exact expected path in its
`FAILED local_…` line — fix and re-run; successes are kept. If a work is unavailable,
skip it: the author then simply has fewer sources and is auto-flagged exploratory.
These files are indexed but their passages are never displayed
(`license_status: rights_cleared_private`, `display_allowed: false`).

## Step 1 — fetch and import sources (Colab, CPU is fine)

From the repo root (Drive-mounted, as in notebook section 1):

```bash
# English additions (Wilde, Yeats, Shaw, …) are discovered via Gutendex:
python scripts/fetch_gutendex.py --corpus both --language en --max-works 0
# Non-English catalog rows (verified 2026-07-17) + the 12 local files:
python scripts/fetch_multilingual_sources.py \
  --language zh --language ja --language fr --language de --language ru \
  --language es --language it --language pl --language en --skip-existing
python scripts/import_source_manifest.py data/source_registry/source_manifest.csv --append
```

Note `--language en` is new here: it picks up the three Rowling `local_text` rows.

## Step 2 — rebuild chunks and heldout splits (CPU)

```bash
python scripts/build_chunk_parquet_from_sources.py --corpus both \
  --output data/all/meta/all_sources_chunks.parquet \
  --coverage-output data/all/meta/all_sources_coverage.json
python scripts/make_source_heldout_splits.py \
  --input data/all/meta/all_sources_chunks.parquet \
  --output data/all/meta/all_source_heldout_splits.parquet \
  --report data/all/meta/all_source_heldout_report.json
```

Check `all_sources_coverage.json`: the new authors should appear with their source counts;
authors with <3 independent sources are expected to be flagged not source-heldout-ready.

## Step 3 — rebuild the index with the frozen encoder (GPU, no training)

```bash
python scripts/multilingual_style_index.py build \
  --input data/all/meta/all_sources_chunks.parquet \
  --out-dir artifacts/multilingual_style_index_challenger_v1 \
  --model-name artifacts/multilingual_author_style_v1 \
  --topic-model-name intfloat/multilingual-e5-base \
  --embedding-cache artifacts/multilingual_style_index_challenger_v1/chunk_embeddings.npz \
  --topic-embedding-cache artifacts/multilingual_style_index_challenger_v1/topic_chunk_embeddings.npz \
  --batch-size 128 --per-source-cap 50 --profile-cap 600 \
  --profile-strategy single_centroid \
  --model-label challenger_finetuned \
  --model-comparison artifacts/model_comparison_v1.json \
  --artifact-version challenger_v2_expanded \
  --heldout-report data/all/meta/all_source_heldout_report.json \
  --device cuda
```

Only new `chunk_id`s are encoded; the old cache is reused. Confirm in the printed metadata:
`n_profiles` grew (167 → ~200), `deployment_matches_selection: true`, and the language list
is unchanged (9 languages).

## Step 4 — refit open-set calibration (GPU then CPU; notebook section 5 does this loop)

```bash
for lang in de en es fr it ja pl ru zh; do
  python scripts/evaluate_open_set.py \
    --input data/all/meta/all_source_heldout_splits.parquet \
    --model-name artifacts/multilingual_author_style_v1 \
    --out-dir artifacts/open_set_eval_challenger_v1/$lang \
    --language $lang --device cuda
done
python scripts/multilingual_style_index.py calibrate \
  --index-dir artifacts/multilingual_style_index_challenger_v1 \
  --open-set-calibration-dir artifacts/open_set_eval_challenger_v1
```

Languages with <10 heldout profiles are skipped (a `open_set_skipped.json` is written) —
expected for it/pl. `calibrate` hard-fails if any report was fitted on a different encoder.

## Step 5 — measure challenger latency (the committed numbers are baseline-encoder only)

```bash
python scripts/multilingual_style_index.py benchmark \
  --index-dir artifacts/multilingual_style_index_challenger_v1 \
  --text "<any ~150-word passage>" --language en --runs 30 --device cuda \
  --output artifacts/multilingual_style_index_challenger_v1/latency_gpu.json
# repeat with --device cpu → latency_cpu.json
```

## Step 6 — copy the small evidence files into the repo

So that the README/method page can cite real numbers, copy from Drive into the repo (all
are small JSON):

```text
artifacts/model_comparison_v1.json                       # heldout MRR per candidate + decision
artifacts/multilingual_style_index_challenger_v1/metadata.json   # now with open_set_calibration
artifacts/multilingual_style_index_challenger_v1/latency_{cpu,gpu}.json
artifacts/open_set_eval_challenger_v1/*/open_set_metrics.json
```

Then update the two "pending" spots with the real numbers: the *Current performance
record* table in `ReadME.md`, and the matching table in `web/static/method.html`.

## Step 7 — sanity checks before serving

```bash
python scripts/audit_source_metadata.py --corpus both \
  --output artifacts/baseline_v1/source_metadata_audit.json
python scripts/multilingual_style_index.py query \
  --index-dir artifacts/multilingual_style_index_challenger_v1 \
  --text "<a paragraph by one of the new authors>" --language fr --mode within
```

The query check should return the expected author in the top ranks for an in-corpus
passage, with `rejection.status: calibrated_open_set` for calibrated languages. The web
process loads the index once at startup, so restart it after replacing artifacts. No model
re-training happened at any step; the encoder directory is untouched.
