# StyleMatch model card

## Intended use

StyleMatch retrieves curated author-language profiles by stylistic resemblance. It supports
literary discovery and research demonstrations, not forensic attribution, plagiarism detection,
or claims about who wrote a passage. V2 scores are similarities, not probabilities; formal
open-set rejection is not calibrated.

## Versioned components

- Encoder: [`stylematch-authorship:v2`](https://huggingface.co/sylviachangdou-prayer/stylematch-authorship/tree/v2), a frozen 1024-dimensional multilingual authorship representation. Its weights are unchanged from model v1.
- Ranking artifact: [`stylematch-index:v2`](https://huggingface.co/datasets/sylviachangdou-prayer/stylematch-index/tree/v2).
- Ranking backend: per-language 0.30-shrinkage whitening plus a 0.01 source-balanced false-Top3 exposure-prior penalty.
- Topic/tone: `intfloat/multilingual-e5-base`, reported separately and excluded from Style Match rank.
- Decade: a separate experimental prototype task; it is unavailable when dated support is insufficient.

## Training constraints

- Use original-language primary texts only.
- Positive pairs must come from different independent sources by the same author.
- Hard negatives prioritize matched language, topic, register, and period.
- Registry membership alone does not create a training example.
- Language-aware batches are used; machine translation does not supply training text.

## Evaluation and selection

Complete sources are assigned to train, dev, or test, and profile-level bootstrap intervals
account for dependent chunks. The fine-tuned authorship encoder was selected because the
numerically stronger multi-view reranker did not establish a positive paired MRR gain and failed
the subgroup non-degradation rule. The v2 exposure penalty was dev-selected under retrieval
non-inferiority and concentration-reduction gates. See
[`method_evidence_ledger.md`](method_evidence_ledger.md).

## Limitations

- Style, topic, register, period, and translation history remain partly entangled.
- Evidence and performance vary across languages and corpora.
- Cross-language scores lack ordered-pair calibration.
- V2 has no calibrated probability or universal rejection threshold.
- The production calibration is refitted on the complete source-prototype bank; locked diagnostic
  results validate the scoring design rather than the exact full deployment artifact.
- Public-domain source availability creates historical and genre selection bias.
