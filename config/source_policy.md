# StyleMatch Source Policy

## Inclusion rule

Only ingest original-language source text. Do not ingest translations, adapted transcripts, subtitles, paraphrases, summaries, or LLM-generated imitation text.

## Language rule

- If the person originally wrote or spoke in English, collect English originals.
- If the person originally wrote or spoke in another language, collect that source language.
- If only translations are available for a person in the current round, skip that person for the current round.

## Corpus rule

- `literary`: prose by named authors, stored in the author's original language.
- `rhetorical`: public rhetoric by named speakers/writers, stored in the speaker's original language.
- Campaign material remains out of scope unless explicitly added later as a separate corpus.

## Required minimal metadata

- `name`
- `corpus`
- `original_language`
- `source_family`
- `notes`

