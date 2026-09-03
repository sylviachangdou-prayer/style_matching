# Concentration-search result

## Verdict

The best dev-selected backend is `whitened_cosine:0.3`. It improves aggregate retrieval and
aggregate exposure concentration, but it is not deployable under the existing gate.

## Test diagnostic

| Metric | Cosine | Whitening 0.30 | Change |
| --- | ---: | ---: | ---: |
| MRR | 0.444 | 0.500 | +0.057 |
| Recall@3 | 0.503 | 0.579 | +0.077 |
| Worst-decile Recall@3 | 0.060 | 0.120 | +0.060 |
| False-top3 HHI | 0.00562 | 0.00537 | −0.00025 |
| False-top3 Gini | 0.261 | 0.230 | −0.031 |
| Maximum false-top3 share | 0.0158 | 0.0116 | −0.0042 |

The MRR difference is profile-bootstrap stable: 0.057, 95% CI [0.040, 0.073].

## Failure points

- Rhetorical MRR: 0.730 to 0.679. The supported-corpus non-degradation gate fails.
- Joyce source-balanced false-top3 share: 0.0060 to 0.0111.
- Mansfield: 0.0090 to 0.0105.
- Lawrence: 0.0074 to 0.0068.
- The test split was already opened in the preceding experiment and is diagnostic only.

## Mechanism decisions

- Retain ordinary whitening as the only serious candidate.
- Drop author-balanced whitening: it added no useful correction.
- Drop the tested whitening + CSLS grids: none passed dev constraints, so their local-density
  correction did not combine cleanly with the style geometry.
- Do not introduce named-author punishment. The watched-profile results show why a global
  concentration objective can move exposure onto different authors without fixing the
  user-visible symptom.

## Next confirmatory design

Use whitening 0.30 as the fixed candidate on newly collected sources. Add a separate
production-like query bank and measure hub stability under length-matched subsampling and
sentence-removal perturbations. Keep rhetorical performance as a blocking subgroup and
refit open-set thresholds if whitening is ultimately adopted.

## Focused 2024–2026 literature record

- Alipoormolabashi, Patel, and Balasubramanian (ACL 2025), “Quantifying
  Misattribution Unfairness in Authorship Attribution,” introduce MAUI_k from excess
  false top-k appearances and show that authors nearer the author-embedding centroid face
  higher misattribution risk. This is the closest task match to the Joyce/Mansfield symptom.
  The next experiment therefore adds source-balanced MAUI@3 and reports each watched
  profile’s post-whitening centroid distance; neither author is used as a tuning target.
  <https://aclanthology.org/2025.acl-short.80/>
- Nielsen and Hansen (NLDL/PMLR 2024), “Hubness Reduction Improves Sentence-BERT
  Semantic Spaces,” find hubness in dense text embeddings and report f-norm followed by
  Mutual Proximity as their strongest tested combination. They also warn that a geometry
  transformation can damage learned structure when baseline hubness is modest. The new
  grid therefore tests f-norm and a cross-set Mutual Proximity approximation only as
  dev-gated additions to `whitened_cosine:0.3`, never as automatic replacements.
  <https://proceedings.mlr.press/v233/nielsen24a.html>
- Jaspal, Agarwal, Gupta, and Vichare (UMAP 2025), “Finding Interest Needle in
  Popularity Haystack,” subtract an exposure term from the relevance logit at inference to
  control over-retrieval. Their evidence is from recommendation, not authorship. The
  StyleMatch analogue is deliberately weaker: a train-only, source-balanced candidate
  exposure prior with a small dev-selected coefficient.
  <https://arxiv.org/abs/2503.23630>
- Habler et al. (2026), “Adversarial Hubness Detector,” combine median/MAD hub scores,
  cross-cluster spread, and stability under query perturbations. This is a hub detector for
  RAG security, not a ranking correction or authorship study. It motivates the robust
  candidate-null diagnostic now; its perturbation test still requires a production-like
  query bank and fresh encoding and is not approximated with the existing chunk vectors.
  <https://arxiv.org/abs/2602.22427>
- Huang and Zhu (2026 preprint), “SATTC,” combine whitening with mutual-neighbor and
  bidirectional-rank anchors plus a class-popularity penalty in a one-shot structural
  expert. The domain is cross-subject EEG–image retrieval. We transfer only the falsifiable
  scoring idea: a frequently retrieved candidate is penalized unless its query match also
  lies in that candidate’s high train-impostor tail. The correction is fitted from the train
  query bank, avoiding test-batch transduction and named-author punishment.
  <https://arxiv.org/abs/2603.20738>
- Sakhawat, Sadab, and Shahriar (2026 preprint) attribute cross-lingual retrieval
  asymmetry primarily to hubness and recommend CSLS. That result concerns parallel
  multilingual expressions. StyleMatch’s own whitening-plus-CSLS grid failed its dev
  constraints, so this paper supports measuring reciprocity but does not justify reviving
  CSLS here. <https://arxiv.org/abs/2605.26575>

The resulting hypotheses are narrow: candidate-conditioned score calibration may correct
Joyce/Mansfield-style false attraction left after global whitening; exposure subtraction
must be protected by reverse-rank anchors; and any reduction in concentration is rejected
if rhetorical or other supported subgroups lose more than the locked tolerance.
