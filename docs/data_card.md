# StyleMatch data card

## Corpus definition

The registry contains literary and rhetorical author-language profiles. The registry is the target
universe, not proof of text availability. Actual eligibility is computed from independent source
rows and reported in `coverage_matrix.csv`.

## Admission tiers

- `formal`: at least three independent sources and a valid source-heldout train/dev/test split.
- `exploratory`: one or two independent sources; retrievable but excluded from headline metrics.
- `catalog_only`: no usable source text; never ranked.

## Source policy

- Preserve original-language primary text.
- Exclude translations, subtitles, adaptations, summaries, paraphrases, and generated imitation.
- Treat duplicate editions, mirrors, and chunk repetitions as one source.
- Record year/date, topic, domain, register, source type, delivered language, licence, display
  permission, canonical URL, and local provenance.
- Display excerpts only when `display_allowed=true` and licence status is public domain, licensed,
  or permission granted.

## Splits and independence

Chunks from one source never cross source-heldout partitions. Topic/domain/register/time splits
assign the grouping variable globally to one partition. Decade evaluation holds out complete
authors; chunks are repeated measurements, not independent evaluation units.

## Gaps

Modern rhetoric requires archive-specific collection and delivered-language verification.
Copyrighted literary and public-figure texts may be usable for internal measurement but not excerpt
display. Missing metadata remains an explicit audit failure and is not imputed from author era.
