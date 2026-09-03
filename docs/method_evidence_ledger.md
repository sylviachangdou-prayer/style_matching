# StyleMatch method evidence ledger

Public methodology claims must use the records below. Results from different candidate
pools, evaluation units, or artifact revisions are not directly comparable.

| Claim | Authoritative artifact | Scope | Public status |
| --- | --- | --- | --- |
| The deployed encoder is the fine-tuned multilingual authorship representation; the learned multi-view reranker was the numerical leader but its paired MRR gain was not distinguishable from zero. | `method_performance.json`; `multiview_fusion_metrics.json` | Locked independent-source test; profile-bootstrap uncertainty | Verified model-selection evidence |
| StyleMatch v2 uses per-language 0.30-shrinkage whitening followed by a source-balanced false-Top3 exposure-prior penalty of 0.01. | `../artifacts/multilingual_style_index_v2/metadata.json`; `postwhitening_dev_search.csv` | Deployed mechanism; penalty selected on dev | Deployed beta |
| On the locked diagnostic test, the exposure penalty changed MRR from 0.5003 to 0.5011 and Recall@3 from 0.5794 to 0.5780 while lowering false-Top3 Gini from 0.2303 to 0.2140 and source-balanced MAUI@3 from 0.0728 to 0.0639. | `postwhitening_test_diagnostic.csv` | Same encoder, whitening, candidate pool, queries, and source-heldout split | Verified diagnostic evidence |
| Supported language/corpus groups stayed within the predeclared non-degradation tolerance used for dev selection. | `postwhitening_subgroup_metrics.csv`; `postwhitening_dev_search.csv` | Subgroup diagnostic; very small language groups remain unstable | Verified with support caveat |
| The watched profiles became less frequent false Top-3 returns under the selected correction. | `postwhitening_watched_profiles.csv` | Pre-specified author-level diagnostic, not a universal fairness claim | Verified diagnostic evidence |
| The expanded index broadens coverage but makes retrieval harder for profiles shared with the earlier, smaller pool. | `old_vs_new_index_metrics.json` | Same frozen query embeddings; profile-grouped bootstrap | Coverage-first trade-off |
| The production scorer refits whitening and exposure priors over the complete source-prototype bank. | `../artifacts/multilingual_style_index_v2/metadata.json`; `../scripts/build_postwhitening_exposure_index.py` | Full deployed index, not the narrower locked diagnostic pool | Experimental deployment extrapolation |

Do not describe v2 scores as probabilities, calibrated confidence, forensic attribution, or
an accuracy improvement over v1. The frozen diagnostic validates the scoring design; it does
not directly estimate the complete production artifact after the full-index prototype refit.
