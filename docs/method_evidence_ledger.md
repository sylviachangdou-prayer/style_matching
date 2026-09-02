# StyleMatch method evidence ledger

Public methodology claims must point to one of the frozen records below. Results from
different candidate pools or evaluation units are not directly compared.

| Claim | Authoritative artifact | Scope | Public status |
| --- | --- | --- | --- |
| The deployed encoder is the fine-tuned multilingual authorship representation. | `method_performance.json` | Locked source-heldout model comparison | Deployed |
| The expanded index broadens candidate coverage but lowers retrieval metrics for profiles shared with the earlier, smaller candidate pool. | `old_vs_new_index_metrics.json` | Same frozen query embeddings; source/profile-grouped bootstrap | Coverage-first beta; do not describe as an accuracy improvement |
| The expanded index contains 2,577 source prototypes and 63,576 chunks across nine languages. | `../artifacts/multilingual_style_index_gutenberg_v3/metadata.json` | Served index metadata | Deployed locally |
| Micro-density reranking at 0.004 lowers false top-three HHI from .00546 to .00522 and Gini from .240 to .204. Its paired MRR and Recall@3 intervals include zero change. | `hubness_reranking_metrics.json` | Exploratory source-grouped cross-fitting; no re-encoding | Experimental; not deployed |
| The density correction reduces the tracked over-exposure of some central profiles but not all of them. | `hubness_reranking_author_exposure.csv` | Per-profile false top-three exposure | Experimental; do not generalize to universal fairness improvement |

The production index must retain `hubness_correction: none` until a locked confirmatory
evaluation passes both retrieval and concentration gates.
