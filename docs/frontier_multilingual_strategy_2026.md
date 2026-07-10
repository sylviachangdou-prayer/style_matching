# StyleMatch multilingual research decision, v1

Date: 2026-07-10

## Decision

The production default compares original-language user text directly with original-language author profiles. It does not translate the query into every corpus language before style retrieval.

Translation is permitted only as a separately labelled experimental or topic-channel condition. A translation-mediated result must not be described as the user's native writing style, must not update author profiles, and must not be pooled with direct original-language scores.

The retrieval task is open-set profile ranking and verification, not closed-set authorship classification. Style and topic remain separate channels.

## Evidence base, 2023-2026

- Sawatphol et al. (TACL 2024), [Addressing Topic Leakage in Authorship Attribution](https://aclanthology.org/2024.tacl-1.75/), shows that conventional cross-topic tests can retain topic shortcuts. Source-heldout evaluation alone is therefore necessary but insufficient; topic/domain/time controls remain required.
- Patel et al. (NAACL 2025), [StyleDistance](https://aclanthology.org/2025.naacl-long.436/), trains a style encoder from controlled parallel examples designed to vary style while holding content more stable. It is the English reference model, not a multilingual production default.
- Qiu et al. (Findings ACL 2025), [mStyleDistance](https://aclanthology.org/2025.findings-acl.869/), extends this approach to multilingual and cross-lingual comparison. Its original-text condition generally outperforms translation-to-English; the paper explicitly documents style loss such as Chinese honorific distinctions collapsing in translation. This is the default cross-language style backbone for the current multilingual registry; language-pair calibration remains mandatory for languages outside the validated evaluation set.
- Kim et al. (EMNLP 2025), [Learning Multilingual Authorship Representations](https://aclanthology.org/2025.emnlp-main.1766/), uses probabilistic content masking and language-aware batching over 4.5 million authors, 36 languages, and 13 domains. Its [multilingual-style-representation model](https://huggingface.co/Blablablab/multilingual-style-representation) is the principal challenger for within-language and cross-domain recall.
- Bevendorff et al. (Findings ACL 2025), [The Million Authors Corpus](https://aclanthology.org/2025.findings-acl.1335/), supplies large-scale multilingual authorship evidence but also shows that cross-domain generalization is materially harder than unseen-language transfer. Domain-heldout reporting is mandatory.
- Lyu et al. (EMNLP 2023), [Translationese and cross-lingual transfer](https://aclanthology.org/2023.emnlp-main.438/), establishes translationese as a systematic distributional shift rather than a neutral preprocessing operation.
- Swamy et al. (EACL 2026), [iBERT](https://aclanthology.org/2026.eacl-long.65/), demonstrates that interpretable, decomposable style embeddings can remain competitive. Its current evidence is not sufficient to replace the multilingual backbone, but its decomposition objective is a reranker research direction.
- Liu et al. (*SEM 2026), [Language bias in multilingual embeddings](https://aclanthology.org/2026.starsem-conference.26/), motivates language-pair calibration: raw multilingual cosine scores are not assumed comparable across target languages.

## Retrieval architecture

### Offline

1. Keep raw texts and passages in the author's source language only.
2. Build source-heldout, topic/domain-heldout, and where possible time-heldout splits.
3. Encode corpus chunks once with `StyleDistance/mstyledistance`.
4. Normalize and average balanced samples into one profile per corpus, language, and author.
5. Store representative original-language passages nearest each profile centroid.
6. Fit calibration separately for each input-language/target-language pair when genuine positive and negative validation pairs exist.

### Online

1. Validate language and minimum input length.
2. Encode the user passage once with a preloaded model.
3. Use a single matrix multiplication against precomputed normalized profiles.
4. Within-language mode returns one calibrated ranking for the input language.
5. Cross-language mode returns separate rankings by target language until pair-specific calibration supports a defensible global ranking.
6. Rerank only a small top-K with multilingual stylometric and compression features. The topic model runs as a physically separate channel.

No per-author transformer call and no online translation fan-out is allowed in the default path.

## Latency contract

- Warm GPU target: p50 at or below 500 ms; p95 at or below 1.5 s for a 75-500 word-equivalent input.
- CPU fallback target: p95 at or below 4 s.
- Measure model load separately from request latency.
- Benchmark after one warm-up and report at least 30 timed requests.
- Use ONNX Runtime where it improves the deployment target. Sentence Transformers documents the ONNX backend and optimization path; Optimum documents CPU dynamic quantization. See [Sentence Transformers efficiency](https://sbert.net/docs/sentence_transformer/usage/efficiency.html) and [Optimum ONNX quantization](https://huggingface.co/docs/optimum-onnx/onnxruntime/usage_guides/quantization).
- At the present author count, normalized NumPy matrix multiplication is preferable to adding FAISS. Reconsider ANN only when profile count or passage-level retrieval makes exact search measurable in p95 latency.

## Evaluation gates

The model cannot be described as production-ready until all applicable gates pass:

1. Source-heldout Recall@3/5/20 and MRR by language.
2. Cross-topic and cross-domain results by language, with the raw character model reported only as a topic-leakage ceiling.
3. Open-set AUROC, equal-error rate, and calibrated rejection thresholds.
4. Reliability/error calibration and selective risk under the low-confidence option.
5. Cross-language metrics by ordered language pair; no macro average without pair-level results.
6. Translation-mediated ablation against direct original-text retrieval, never folded into the headline score.
7. Warm p50/p95 latency on the actual deployment hardware.

## Current corpus expansion

The curated expansion adds 45 independent public primary texts: three authors and three works per author in Chinese, Japanese, French, German, and Russian. Chinese, French, and German texts come from Project Gutenberg; Japanese texts come from Aozora Bunko; Russian texts come from Russian Wikisource. The catalog records only original-language editions and the fetcher rejects texts whose dominant script is inconsistent with the declared language. These 45 are an expansion batch, not the project author total. The all-source build also imports every English literary/rhetorical source returned from the full registry search and any later source batch.

This is enough to establish a multilingual pipeline and source-heldout tests. It is not enough to claim population-wide cross-language validity. Modern rhetorical corpora require a separate source-verified collection pass because official speech pages, transcript authorship, and delivered-language metadata differ from literary work metadata.
