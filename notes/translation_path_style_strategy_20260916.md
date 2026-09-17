# Translation-path style matching: method strategy

Date: 2026-09-16  
Status: exploratory proposal; novelty and efficacy are not established.

## Question and current observation

The user reports that feeding the classical Chinese poem Shu Dao Nan to StyleMatch returned John Adams, George Washington, and Zachary Taylor. This is one product observation, not an evaluation: the exact input form, translation path, candidate universe, scores, and rank margins were not supplied.

The result should initially be treated as a possible out-of-domain or content-leakage failure, not evidence that the system detected Li Bai's style in American presidents. Plausible causes include (i) classical Chinese / regulated verse being outside the encoder's tested distribution, (ii) the query being translated or implicitly mapped into modern English, with translator/model style entering the representation, (iii) semantics or rhetoric dominating style, (iv) a high-hub candidate or candidate-pool artifact, and (v) direct memorization/recognition of a famous poem. These are hypotheses until the exact run is reproduced with score traces and controls.

The construct must be stated before modeling. There are at least three different tasks:

1. **Source-author attribution:** identify the writer of an original classical text.
2. **Translator attribution:** identify recurring stylistic choices in a translation.
3. **Cross-lingual perceived-style similarity:** rank texts that independent readers judge stylistically similar, without claiming common authorship.

The reported US-president matches concern task 3 at most. They are not a valid author-attribution outcome.

## What prior work changes

- Multilingual style embeddings already exist: mStyleDistance constructs embeddings for nine languages, including Chinese, and evaluates multilingual style similarity and authorship verification. Therefore “apply a multilingual style encoder to Chinese” is not a defensible novelty claim. Its coverage of modern Chinese does not itself validate Classical Chinese, historical language shift, or Tang poetry. [Qiu et al., Findings ACL 2025](https://aclanthology.org/2025.findings-acl.869/)
- Classical-poetry style identification is also prior art. A 2022 arXiv paper studies Tang/Song poetry-school style judgments with a pretrained poetry model; a 2023 poet-profiling paper combines style features with literary domain knowledge. Old-Chinese stylometry using POS 4-grams is reported in the 2025 Ancient Language Processing workshop. A contribution must go beyond “use a pretrained model for ancient-poetry style.” [Pretrained-model style judgment](https://arxiv.org/abs/2211.04657), [poet profiling](https://www.mdpi.com/2079-9292/12/3/656), [Old-Chinese stylometry](https://aclanthology.org/2025.alp-1.15/)
- Translation is not a neutral conversion. A 2024 stylometric study of translated texts finds partial translator attribution, but also clustering of translations of the same original despite different translators. This makes original text, translator, and translation period crossed causes of the observed style signal. [Pagano et al., PROPOR 2024](https://aclanthology.org/2024.propor-2.15.pdf)
- There is a large Tang/Song poem resource with 32,399 expert-segmented poems, but segmentation agreement was much stronger at the line than whole-poem level. A method should retain line/form information and report segmentation uncertainty rather than treat one tokenization as ground truth. [Liu et al., NLP4DH 2022](https://aclanthology.org/2022.nlp4dh-1.17/)
- Contemporary Chinese cross-topic attribution has a reproducible benchmark (CCTAA), but it is newswire prose, not ancient poetry. It is a useful domain-shift control, not a substitute for a Classical Chinese benchmark. [Wang & Riddell, LREC 2022](https://aclanthology.org/2022.lrec-1.633/)
- A small parallel text with six English translations of the same Chinese classic can isolate translator-path variation, but one source text cannot establish author-style retrieval. [Six translations of Daxue](https://doi.org/10.3389/fpsyg.2022.1069697)

## Candidate contribution: Translation-Path Consensus with Abstention (TPCA)

The defensible target is *human-perceived cross-lingual style similarity under translation variation*, not recovery of a source author's invariant essence. Translation may preserve, suppress, or introduce stylistic properties.

For a query \(q\), candidate profile \(a\), and independently sourced translation path \(j\), let \(r_j(q,a)\) be a score calibrated on development data for the ordered language pair, genre, and translation condition. Define:

\[
S_{\mathrm{TPCA}}(q,a)=\operatorname{median}_j r_j(q,a)
- \lambda\,\operatorname{MAD}_j(r_j(q,a))
- \gamma\,O(q,a),
\]

where MAD is disagreement across translation paths and \(O\) is a separately calibrated out-of-domain penalty. Return a match only if the best calibrated score exceeds a development-set threshold and translation-path disagreement is below a predeclared bound; otherwise abstain. Raw cosine values from different languages must not be averaged.

This is a proposal, not a novelty claim. Its potentially publishable element is whether *translation-path instability predicts invalid style matches* and whether selective abstention improves human-judgment precision at fixed coverage. A simple multi-translation average is not enough; it is an ensemble baseline.

For same-language Classical Chinese retrieval, use a distinct native-form branch: character/function-character distributions, line length and metrical form, parallelism, repetition, and rhetorical/discourse structure. Do not force those features into the English-translated score. Keep semantic/topic embeddings in a separate diagnostic channel.

## Minimal controlled study

### Data design

- Start with a multi-poet Tang/Song corpus. Require multiple independent poems per author and keep poem/source IDs, dynasty, form, title, edition, punctuation/segmentation provenance, and disputed-attribution flags.
- For a translation pilot, use only poems with multiple independently authored published translations where permissions allow. Add controlled machine translations as a separately labeled condition, not as human gold. Translators must be crossed with source poets/texts; a single famous poem with many translations estimates translation differences, not author identity.
- Include the reported Shu Dao Nan as a case study, not as the full test set. Extend to a set of Li Bai poems and matched poems by other poets, with the entire poem/work held out.
- Candidate pools must be explicit: (a) same-language classical poets, (b) modern Chinese prose/poetry, and (c) English literary/rhetorical profiles. The US presidents belong only to a cross-domain stress test, not the primary candidate set.

### Conditions and controls

1. Original Classical Chinese.
2. Expert modern-Chinese gloss.
3. Literal English translation.
4. Multiple human/published English translations.
5. Multiple machine-translation systems/prompts, labeled by engine and version.
6. Meaning-preserving prose paraphrase that removes verse form.
7. Content controls: topic-matched poems by other poets, shuffled-line/couplet controls, and title/name-masked inputs.

Report raw multilingual encoder, mStyleDistance, classical stylometry, translation-only scoring, and TPCA. Use author/work-disjoint splits; tune \(\lambda,\gamma\), calibration, and abstention only on development data. Cluster uncertainty by source poem and author, not by generated translation variant.

### Construct-validity target

Collect independent pairwise judgments of stylistic similarity. Separate Chinese-expert judgments on original texts from English-reader judgments on translated texts; do not silently pool them into one universal scale. Keep authorship labels hidden during judgment and ask annotators to distinguish style from shared topic/content. Measure agreement and report judgments by original poem and translation path. The external benchmark's human-label process must remain distinct from the evaluation method's author labels.

### Primary outcomes and falsifiers

- Primary: pairwise agreement / ranking correlation with independent human style judgments, stratified by original vs translation path.
- Secondary: within-language author retrieval on held-out works; this is not a substitute for the human-similarity outcome.
- Selective prediction: risk–coverage curve and precision at prespecified coverage; test whether path disagreement identifies false confident matches.
- Content-leakage falsifier: if the same English presidents remain top-ranked for unrelated topic-matched classical poems or prose glosses, the apparent cross-language style signal is likely semantic/register/candidate bias.
- Translation-instability falsifier: if rankings swing across faithful translations and TPCA abstention does not improve human-judgment precision, drop the method.
- OOD falsifier: if scores are confident on Classical Chinese despite poor human agreement, the score calibration is not detecting domain shift.

## Ranked route

1. **First, reproduce the product observation.** Save the exact source string, whether it was translated, translation output/model, detected language, candidate pool/version, top scores and margins, and the original/translated text fingerprints. Compare the original, title-masked text, literal gloss, and at least two independent translation paths. No conclusion from rank labels alone.
2. **Then run the small translation-path diagnostic.** Use a crossed set of poems/translators and matched content controls. Ask whether translation-path disagreement predicts disagreement with human judgments.
3. **Only if the diagnostic survives, build TPCA.** Calibrate ordered language-pair and genre-specific scores on dev; fit an abstention threshold; freeze the full test before any method comparison.
4. **Keep the main StyleMatch ECoRe story separate.** The ancient-text study could become a separate paper or a domain-shift/construct-validity section only after it has the data and human judgments. Do not combine it with a same-author retrieval claim by narrative alone.

## Confirmed / inferred / unverified

- **Confirmed from project artifacts:** the deployed evidence distinguishes original-language retrieval from translation-mediated results; translations are not used to populate author profiles. Existing documentation says translation-mediated comparison must remain separately labeled.
- **Confirmed from literature:** multilingual style encoders, classical-poetry style identification, Old-Chinese stylometry, and translator-style analysis all have prior work, so each is a novelty boundary.
- **Inference:** the three-president result is more plausibly a domain/content/hubness warning than proof of stylistic equivalence. Exact cause is not identified.
- **Unverified:** the user-reported query's actual translation path and score margins; public availability/licensing and translator metadata for a sufficiently crossed Tang-poetry translation corpus; novelty of TPCA after a full literature review.

## Immediate project outputs

- main_readiness_ecore_test.ipynb runs an ECoRe-vs-centroid diagnostic with false-return and subgroup reporting. It explicitly cannot award main readiness because its V3 test may already have been opened and it does not supply human judgments, public benchmark results, or calibrated ECoRe open-set rejection.
- The reported Shu Dao Nan matches are not included as measured evidence; the original query and ranking trace are still required to reproduce them.
