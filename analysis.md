# Analysis log

## 2026-09-02 — frozen-encoder ranking backends

### Current mechanism diagnosis

The recurrent-author problem is not explained by cosine alone. Shrinkage whitening (0.1)
raised profile-macro MRR from 0.444 to 0.504 and Recall@3 from 0.503 to 0.589 while reducing
false-top3 HHI from 0.00562 to 0.00539 and Gini from 0.261 to 0.236. The paired-profile MRR
difference was 0.060, 95% CI [0.043, 0.078]. This supports anisotropic covariance as a
material retrieval problem in the frozen representation.

The hub mechanism survives. Whitening moved rather than eliminated the largest hubs. For
the three repeatedly observed English profiles, source-balanced false-top3 share increased
for James Joyce (0.0060 to 0.0075) and Katherine Mansfield (0.0090 to 0.0101), but fell for
D. H. Lawrence (0.0074 to 0.0057). Offline aggregate concentration therefore does not prove
that the production-query symptom is solved.

### Where the evidence breaks

- Whitening improved every language group with at least ten evaluated profiles, but the
  rhetorical corpus fell from MRR 0.730 to 0.673 and Recall@3 0.839 to 0.817. A global
  deployment violates the predeclared subgroup non-degradation rule.
- Polish and Chinese contain only four evaluated profiles each; their negative whitening
  differences are too thin for a stable conclusion.
- The evaluator selected hyperparameters on dev, but chose the final method among families
  using test MRR. Because test informed selection, these results are exploratory and need a
  newly frozen source set for confirmation.
- Open-set rejection and score calibration were not evaluated. Whitening changes score
  distributions, so existing thresholds cannot be reused.

### Competing pathways

- **Supported:** shrinkage whitening repairs covariance geometry and materially improves
  held-out author retrieval.
- **Plausible:** neighborhood-density correction remains useful as a secondary layer. CSLS
  slightly improved MRR and reduced HHI/Gini without the recognition losses of adaptive
  S-Norm.
- **Rejected in present form:** all-but-top removal removed useful author signal; adaptive
  S-Norm worsened both retrieval and exposure; L1 and Spearman were effectively cosine
  substitutes rather than upgrades.
- **Mixed:** PLDA and PLDA + S-Norm improved MRR but substantially concentrated false
  exposure. They model identity structure better while worsening the product failure.

### Literature interpretation

Whitening is consistent with anisotropy correction in sentence representations (Su et al.
2021). CSLS/local scaling is designed for neighborhood-density imbalance rather than named
author penalties (Conneau et al. 2017; Schnitzer et al. 2012). Speaker-verification work
shows that PLDA and score normalization are domain- and language-sensitive; their transfer
to literary retrieval is not automatic. The 2026 cross-lingual hubness result is a useful
hypothesis, not direct evidence for stylometry.

### Ranked next tests

1. Dev-selected interpolation of raw and whitened cosine, then whitening + CSLS, with the
   method family selected before the next test set is opened.
2. Author-balanced covariance whitening so prolific profiles do not dominate the transform.
3. Corpus-robust selection: require non-degradation for literary and rhetorical groups;
   avoid a register router until a reliable query-side register measure exists.
4. Author-heldout open-set calibration and perturbation-stability exposure using a frozen
   production-like query bank.

## 2026-09-03 — concentration-constrained continuation

The next search treats concentration as the dev-set objective rather than a descriptive
secondary metric. Candidate eligibility requires non-inferiority on MRR, Recall@3, and
worst-decile profile Recall@3. Among eligible candidates, selection minimizes the mean
relative false-top3 HHI, Gini, and maximum candidate share. The search compares partial
raw/whitened interpolation, author-balanced covariance whitening, whitening + CSLS, and
author-balanced whitening + CSLS. Named authors are monitored but never enter the scoring
formula. Because the existing test set has already been opened, its output is diagnostic;
the selected candidate still requires new-source confirmation and new open-set calibration.
