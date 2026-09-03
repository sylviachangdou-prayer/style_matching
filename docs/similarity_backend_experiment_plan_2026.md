# Similarity-backend experiment: material passport

## Objective

Determine whether recurrent false top-three authors are caused by the frozen embedding geometry, the scoring rule, or candidate-specific cohort density. The experiment does not retrain the encoder and does not authorize a production change by itself.

## Inputs

- Source-heldout split: `artifacts/source_expansion_v2/gutenberg_targeted_v1/source_heldout_splits.parquet`
- Frozen encoder: `artifacts/multilingual_author_style_v1`
- Embedding reuse: `artifacts/source_expansion_v2/gutenberg_targeted_v1/frozen_encoder_eval`
- Seed: `20260902`
- Training cap: 300 chunks per author-language profile
- Bootstrap unit: author-language profile

## Leakage boundary

- Fit centering, principal components, whitening, cohorts, covariance models, and author centroids on `train` only.
- Select each method family's hyperparameter on `dev` macro MRR only.
- Use `test` once for locked estimates and paired bootstrap intervals.
- Keep candidates language-matched. Do not compare raw cosine values across languages.
- Topic similarity is excluded from every Style Match ranking.

## Methods

1. L2-normalized cosine baseline.
2. Mean-centered cosine.
3. All-but-top removal with 1, 2, 4, or 8 train-fitted principal directions.
4. Shrinkage-whitened cosine with shrinkage 0.1, 0.3, or 0.5.
5. Negative L1 distance on frozen embeddings.
6. Spearman correlation over embedding-coordinate ranks.
7. CSLS with cohort size 5, 10, or 20.
8. Adaptive S-Norm with cohort size 20, 50, or 100.
9. Regularized two-covariance PLDA with train-fitted PCA and covariance shrinkage.
10. PLDA followed by adaptive S-Norm.

Normalized Euclidean is not a separate experiment: for unit vectors it is a monotonic transform of cosine and produces the same ranking.

## Outcomes

- Macro MRR and Recall@1/3/5/20 with profile-bootstrap 95% intervals.
- False-top3 HHI, Gini coefficient, and maximum exposure share.
- Worst-decile author-profile Recall@3.
- Language-level geometry: first-PC variance share, effective rank, and mean pairwise cosine.
- Open-set performance is explicitly unavailable here; it requires author-heldout unknown queries.

## Locked adoption rule

Select one method family by dev macro MRR before reading test. It may replace cosine only if the MRR paired-profile interval lower bound is at least −0.01, Recall@3 changes by at least −0.01, both false-top3 HHI and Gini fall, and no supported language or corpus loses more than 0.02 MRR or Recall@3. Test supplies pass/fail evidence; it never chooses among method families. If the dev-selected method fails, retain cosine. This is a non-inferiority rule, not a claim that equal exposure is intrinsically correct.

## Literature basis

- Conneau et al. (2017), *Word Translation Without Parallel Data*, introduced CSLS for neighborhood-density correction: <https://arxiv.org/abs/1710.04087>.
- Schnitzer et al. (2012), *Local and Global Scaling Reduce Hubs in Space*, established local/global hubness correction: <https://www.jmlr.org/papers/v13/schnitzer12a.html>.
- Su et al. (2021), *Whitening Sentence Representations for Better Semantics and Faster Retrieval*, motivates train-fitted whitening for anisotropic sentence spaces: <https://arxiv.org/abs/2103.15316>.
- Peng et al. (2022), *Are Strided Embeddings and Cosine Scoring All We Need?*, motivates comparison with PLDA scoring: <https://arxiv.org/abs/2204.10523>.
- Kim et al. (2025), *Learning Universal Authorship Representations*, is the encoder-family basis rather than evidence for any particular reranker: <https://aclanthology.org/2025.emnlp-main.1766/>.
- Sakhawat et al. (2026), *Hubness, Not Anisotropy, Drives Cross-Lingual Retrieval Asymmetry*, motivates the CSLS hypothesis but studies a different retrieval task; transfer to author style is unverified: <https://arxiv.org/abs/2605.26575>.
- Habler et al. (2026), *Adversarial Hubness Detector*, motivates reporting exposure concentration and perturbation stability; it is not evidence that StyleMatch's frequent authors are adversarial hubs: <https://arxiv.org/abs/2602.22427>.

## Outputs

- `backend_metrics.json`
- `backend_test_metrics.csv`
- `backend_dev_tuning.csv`
- `backend_author_exposure.csv`
- `backend_subgroup_metrics.csv`
- `geometry_diagnostics.csv`
