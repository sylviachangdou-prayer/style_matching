# StyleMatch model card

## Intended use

StyleMatch ranks curated author-language profiles by stylistic resemblance. It is an open-set,
public-facing discovery tool, not forensic authorship attribution. It must return a low-confidence
or no-match state when the input is not adequately represented.

## Model channels

- Style backbone: selected by `artifacts/model_comparison_v1.json`; mStyleDistance remains the
  reproducible baseline.
- Topic/tone: multilingual-e5, kept separate from the Style Match Score.
- Implemented style views: source prototypes, character/function-word models, stylometric
  rhythm, compression distance, and discourse features. Parser-based POS/dependency scores may
  be added as aligned candidates, but are absent unless multilingual parser artifacts are recorded.
- Decade: separate author-heldout-validated `language × register × decade` prototypes.

## Training constraints

- Original-language primary texts only.
- Positive pairs use the same author in different independent sources.
- Registry membership alone does not create a training example.
- Language-aware batches are used for multilingual contrastive adaptation.

## Required evaluation

Report source/topic/domain/time-heldout ranking, ordered cross-language pairs, open-set AUROC/EER,
calibration, selective risk, per-author misattribution exposure, and deployment latency. Fusion is
adopted only when paired-bootstrap evidence beats the best single model without worsening
calibration or selective precision.

## Known limitations

- Style, register, topic, period, and translation history can remain partially entangled.
- Profiles with few sources are exploratory and excluded from headline evidence.
- Cross-language cosine scores are not globally comparable without ordered-pair calibration.
- A decade match is resemblance to the dated corpus, not proof that a text was written then.
- Public figures near the center of embedding space may be over-returned; monitor MAUI-style risk.

## Release status

Read `artifacts/baseline_v1/release_gates.json`. A model is not private-beta ready unless every
gate passes. Raw cosine values must never be described as probabilities.
