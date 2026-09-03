# Ranking-backend strategy after the first whitening result

## Verdict

Do not deploy the current `recommended_backend` flag. Treat shrinkage whitening as the
single confirmatory candidate. The effect is large and profile-bootstrap-stable, but the
test set was used for method-family choice and rhetorical retrieval regressed.

## Confirmatory candidate families

### 1. Raw–whitened interpolation

For language-specific train-fitted whitening score `s_w` and raw cosine `s_c`:

```text
s_alpha = (1 - alpha) * rank_normalize(s_c) + alpha * rank_normalize(s_w)
alpha in {0.25, 0.50, 0.75, 1.00}
```

Select `alpha` on dev using macro MRR subject to corpus Recall@3 non-degradation. This tests
whether the rhetorical loss is caused by applying the full covariance correction.

### 2. Whitening + CSLS

Apply CSLS after whitening, with `k in {5, 10, 20}`. This is the most defensible combined
mechanism: whitening corrects global covariance; CSLS corrects candidate-local density.
Do not add a manual named-author penalty.

### 3. Author-balanced whitening

Estimate covariance after assigning equal total weight to each author-language profile,
and compare it with the current equal-chunk estimator. This directly tests whether corpus
volume makes some profiles define the geometry.

## Locked design

1. Fit all transforms on train sources only.
2. Select exactly one candidate family and its hyperparameters on dev.
3. Freeze code, candidate pool, source identities, and thresholds.
4. Evaluate once on newly held-out sources not used in the September 2 comparison.
5. Require:
   - paired-profile MRR lower confidence bound above zero;
   - Recall@3 improvement or non-inferiority within 0.01;
   - no supported language or corpus MRR loss larger than 0.02;
   - lower false-top3 HHI and Gini;
   - no increase in maximum false-top3 share;
   - open-set AUROC/ECE and selective risk no worse than the cosine system.

## Required diagnostics

- Report source-balanced exposure for production-observed hubs separately from aggregate
  concentration.
- Add query perturbations (sentence removal, punctuation preservation, and length-matched
  subsampling) and report top-three stability.
- Separate literary and rhetorical covariance diagnostics.
- Refit open-set thresholds after any transform; never reuse cosine calibration.

## Evidence status

- Confirmed on the opened split: whitening improves aggregate closed-set retrieval.
- Plausible: whitening + CSLS can retain recognition gains while reducing local hubs.
- Unverified: the same gains will persist on new sources, production-like inputs, open-set
  queries, or cross-language global ranking.

## References

- Conneau et al. (2017), CSLS: <https://arxiv.org/abs/1710.04087>
- Schnitzer et al. (2012), local/global hubness scaling:
  <https://www.jmlr.org/papers/v13/schnitzer12a.html>
- Su et al. (2021), sentence-embedding whitening: <https://arxiv.org/abs/2103.15316>
- Wang, Lee & Liu (2022), cosine versus constrained PLDA:
  <https://www.isca-archive.org/interspeech_2022/wang22r_interspeech.html>
- Cumani & Sarni (2023), adaptive data normalization:
  <https://www.isca-archive.org/interspeech_2023/cumani23_interspeech.html>
- Sakhawat, Sadab & Shahriar (2026), cross-lingual hubness hypothesis:
  <https://arxiv.org/abs/2605.26575>
