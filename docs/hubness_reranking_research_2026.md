# Hubness-aware author reranking: research basis and test contract

## Observed problem

The expanded index increases profile coverage but reduces retrieval performance for authors shared with the earlier candidate pool. On the shared-author operational comparison, MRR falls by 0.061 (95% paired-profile CI −0.074 to −0.049). Holding the candidate set fixed makes the difference exactly zero. The loss therefore comes from additional competitors, not altered embeddings for the original profiles.

The earlier global frequency and local-density subtraction selected a zero penalty. Its smallest non-zero coefficient, 0.02, was already large enough to reduce development MRR sharply. A global exposure penalty is therefore not supported.

## Updated test

`scripts/evaluate_hubness_reranking.py` reuses the frozen score matrices and performs source-grouped cross-fitting. It compares:

- the uncorrected author centroid;
- independent-source prototype aggregation;
- reciprocal-rank fusion of centroid and source-prototype evidence;
- candidate-specific empirical-null calibration within language and corpus;
- mutual-proximity-style calibration using candidate and query neighbourhood ranks;
- calibrated multi-view fusion;
- local-density subtraction on a substantially smaller coefficient grid.

Every method reports macro MRR, Recall@1/3/5/20, profile-bootstrap intervals, paired MRR and Recall@3 differences, language results, source-balanced false-top-three Gini and HHI, maximum false exposure, and author-level exposure counts. Selection minimizes false-top-three HHI subject to non-inferiority bounds for both MRR and Recall@3.

This is an exploratory cross-fitted comparison, not a new untouched confirmatory test. A selected method must later be evaluated on newly frozen sources before deployment.

## Literature used

| Work | Relevance | Status |
|---|---|---|
| Radovanović, Nanopoulos, and Ivanović, “Hubs in Space” (2010), DOI: [10.5555/1756006.1953015](https://doi.org/10.5555/1756006.1953015) | Defines high-dimensional hubness and k-occurrence concentration. | Verified metadata |
| Schnitzer et al., “Local and Global Scaling Reduce Hubs in Space” (2012), DOI: [10.5555/2503308.2503333](https://doi.org/10.5555/2503308.2503333) | Supports distribution-aware neighbourhood rescaling rather than a universal popularity penalty. | Verified metadata |
| Huertas-Tato et al., “Isolating authorship from content with semantic embeddings and contrastive learning” (2024), [arXiv:2411.18472](https://arxiv.org/abs/2411.18472) | Motivates semantic hard negatives and explicit style–content separation. | Verified preprint |
| Zeng, Sclafani, and Rambow, “Gram2Vec: An Interpretable Document Vectorizer” (2024), [arXiv:2406.12131](https://arxiv.org/abs/2406.12131) | Supports a grammatical evidence view that is separable from neural similarity. | Verified preprint |
| Campagnano, Mallia, and Silvestri, “Unveiling DIME” (2025), DOI: [10.1145/3726302.3730318](https://doi.org/10.1145/3726302.3730318) | Motivates query-adaptive denoising and reranking of dense representations; not implemented in the first test because it requires dimension-level embedding access. | Verified metadata |
| Qiu et al., “mStyleDistance” (2025), [arXiv:2502.15168](https://arxiv.org/abs/2502.15168) | Supplies the multilingual style-embedding benchmark and style-versus-content framing. | Verified preprint |
| Miralles-González et al., “One-shot Style Transfer LLM log-probabilities for Authorship Attribution and Verification” (2025/2026), [arXiv:2510.13302](https://arxiv.org/abs/2510.13302) | Provides a future test-time verification view; excluded from the current lightweight reranker because of inference cost. | Verified preprint |
| Habler et al., “Adversarial Hubness Detector” (2026), [arXiv:2602.22427](https://arxiv.org/abs/2602.22427) | Motivates robust k-occurrence, cluster-spread, perturbation-stability, and domain-aware hub diagnostics. | Verified preprint |

Claims about these methods transferring to literary authorship retrieval remain hypotheses until tested on the source-heldout StyleMatch corpus.
