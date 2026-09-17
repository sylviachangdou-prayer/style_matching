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

### Result

Dev selected ordinary shrinkage whitening at 0.30. Relative to cosine on dev, MRR increased
by 0.076, Recall@3 by 0.093, worst-decile Recall@3 by 0.060, and the composite concentration
index fell to 0.873. Author-balanced covariance never improved on ordinary whitening, and
no whitening + CSLS configuration satisfied the dev eligibility constraints.

On the reused test split, whitening 0.30 increased MRR by 0.057 (95% paired-profile CI
[0.040, 0.073]) and Recall@3 by 0.077 while lowering HHI, Gini, and maximum exposure share.
It nevertheless failed the subgroup gate: rhetorical MRR fell by 0.052 and Recall@3 by
0.010. The watched-profile result also remained asymmetric: source-balanced false-top3
share increased for Joyce and Mansfield and decreased only modestly for Lawrence. The
aggregate hubness mechanism therefore survives; global concentration and specific
production-visible hubs are not interchangeable outcomes.

## 2026-09-16 — ancient-text translation paths and main-readiness diagnostic

### Current mechanism diagnosis

The reported Shu Dao Nan → John Adams / George Washington / Zachary Taylor ranking is a
single user-observed example without the exact query, translation route, candidate pool,
scores, or margins. It does not validate cross-lingual style matching. The leading failure
classes are historical-language/domain shift, translation-model fingerprints, semantic or
rhetorical-content leakage, and candidate hubness. Their relative contribution is unknown.

### Literature boundary

Multilingual style embeddings already include Chinese (mStyleDistance, Findings ACL 2025),
and classical-poetry style judgments / poet profiling predate this proposal. Old-Chinese
stylometry has recent workshop work. Translation stylometry shows both translator signal
and same-original clustering across different translators (PROPOR 2024), so translation
cannot be treated as a neutral view of the source author's style. A Tang/Song poem corpus
with 32,399 expert-segmented poems exists, but whole-poem segmentation agreement is limited.
See notes/translation_path_style_strategy_20260916.md for sources and the exact novelty
boundary.

### Candidate pathway

The most defensible research question is whether variation across independent translation
paths predicts errors in human-perceived cross-lingual style similarity. The proposed
Translation-Path Consensus with Abstention (TPCA) uses pairwise-language/genre calibrated
scores, their median, translation-path MAD, and an OOD term; it abstains when agreement or
domain support is inadequate. This is a hypothesis, not yet a novel or validated method.
The source-language structural branch should remain separate from translated English
scores. Same-author attribution, translator attribution, and perceived-style similarity
must not be conflated.

### Falsifiers and next empirical steps

1. Reproduce the exact Shu Dao Nan run and record source text, detected language,
   translation text/engine, candidate index version, full ranked scores, and margins.
2. Compare original, title-masked, literal gloss, multiple independent translations,
   prose paraphrase, and topic-matched classical-poem controls. If the same presidents
   remain top matches under content-preserving prose or unrelated poems, treat the result
   as content/candidate bias.
3. Build a crossed poem × poet × translator evaluation with complete-work holdout; collect
   independent human pairwise style judgments. A single poem with many translations can
   diagnose translator-path effects but cannot support author-style retrieval.
4. Compare raw multilingual embeddings, mStyleDistance, classical stylometry,
   translation-only, and TPCA. Select thresholds on dev; report pairwise human agreement,
   risk–coverage, and source/author-clustered intervals.
5. Keep this ancient-text line separate from the ECoRe ranking claim until it has its own
   construct-validity evidence and open-set test.

### Evidence status

- **Confirmed:** the current repository separates translated-text comparison from original
  author profiles; relevant multilingual and Classical Chinese style methods already
  exist.
- **Inference:** the president matches are more likely an OOD/content/hubness warning than
  evidence of recovered Li Bai style. This cannot be resolved without the rank trace.
- **Unverified:** the input's translation path, exact cause of the matches, a sufficiently
  crossed public translation corpus, and TPCA novelty.

The new main_readiness_ecore_test.ipynb is explicitly a diagnostic on the Part 8.5 V3
source-heldout set, which Part 8.8 may already have exposed. It reports ECoRe versus
centroid retrieval, source-balanced MAUI@3, false-return Gini/max share, and subgroup
intervals. It does not establish human construct validity, two public benchmarks, or
calibrated ECoRe open-set rejection; its final readiness gate therefore cannot return
MAIN-READY.
