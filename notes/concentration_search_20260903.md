# Concentration-constrained backend search

## Mechanism families

1. **Partial covariance correction:** interpolate standardized raw cosine and whitened
   cosine. This tests whether full whitening causes the rhetorical regression.
2. **Author-balanced covariance:** give every author-language profile equal total weight
   when estimating the whitening mean and covariance.
3. **Global plus local correction:** apply CSLS after ordinary or author-balanced
   whitening. Whitening targets global covariance; CSLS targets candidate-local density.

## Search grid

- Whitening shrinkage: 0.03, 0.05, 0.10, 0.15, 0.20, 0.30.
- CSLS neighborhoods: 5, 10, 20.
- Raw/whitened blend: 0.25, 0.50, 0.75 whitened weight.

## Dev selection

An eligible candidate must satisfy all of the following relative to raw cosine:

- MRR difference at least −0.01;
- Recall@3 difference at least −0.01;
- worst-decile profile Recall@3 difference at least −0.02;
- lower false-top3 HHI, Gini, and maximum exposure share.

Among eligible candidates, minimize the arithmetic mean of each concentration measure
divided by its cosine baseline. Break ties with higher MRR. This objective is fixed before
examining the reused test diagnostic.

## Interpretation boundary

The script reports one dev-selected candidate and compares only that candidate with cosine
on the existing test split. It always writes `production_change_authorized: false` because
that test split was already used in the previous method comparison. Confirmation requires
new held-out sources and refitted open-set calibration.

## Literature

- Conneau et al. (2017), CSLS: <https://arxiv.org/abs/1710.04087>
- Schnitzer et al. (2012), local/global scaling:
  <https://www.jmlr.org/papers/v13/schnitzer12a.html>
- Su et al. (2021), embedding whitening: <https://arxiv.org/abs/2103.15316>
- Sakhawat, Sadab & Shahriar (2026), hubness versus anisotropy in cross-lingual retrieval:
  <https://arxiv.org/abs/2605.26575>
- Huang & Zhu (2026), whitening plus adaptive CSLS in a different frozen-retrieval domain:
  <https://arxiv.org/abs/2603.20738> — transfer to stylometry remains unverified.
