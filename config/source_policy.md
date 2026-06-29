# StyleMatch Source Policy

## Inclusion rule

Only ingest text into `data/{literary,rhetorical}` when the source is public-domain or clearly licensed for reuse. Public visibility is not enough.

## V1 scope

- Language: English.
- Literary corpus: public-domain prose, primarily Project Gutenberg texts.
- Rhetorical corpus: public-domain or official government public rhetoric. Campaign material is excluded.

## Source tiers

- `approved`: can be fetched into corpus now.
- `candidate`: useful public figure, but license/source still needs review before ingestion.
- `rejected`: do not use for training.

## Hard exclusions for training data

- LLM-generated imitation text.
- Scraped copyrighted books.
- Modern speech transcripts from media sites, YouTube captions, TED, corporate blogs, or fan transcript sites unless the license explicitly permits reuse.
- Translated speeches unless translator copyright is explicitly clear.

