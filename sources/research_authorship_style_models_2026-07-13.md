# Authorship/style representation methods and model scan

Date: 2026-07-13

## Main findings

- Per-language raw Recall@k and MRR are not comparable when candidate counts, author diversity,
  domains, registers, and source counts differ. English can be harder despite having more text.
- Authorship-representation models optimize author discrimination; style-embedding models optimize
  content-independent stylistic similarity. Neither objective alone fully identifies literary style.
- Topic leakage and named-entity shortcuts remain documented failure modes. Source-, topic-,
  domain-, and author-heldout evaluation is required.
- Learned late fusion of neural authorship embeddings and explicit stylometric views is supported,
  but weights must be learned on held-out data and accepted only under subgroup and calibration
  non-inferiority tests.

## Primary papers and official model cards

1. Rivera-Soto et al. (2021), *Learning Universal Authorship Representations*.
   https://aclanthology.org/2021.emnlp-main.70/
2. Sawatphol et al. (2022), *Topic-Regularized Authorship Representation Learning*.
   https://aclanthology.org/2022.emnlp-main.70/
3. Brad et al. (2022), *Rethinking the Authorship Verification Experimental Setups*.
   https://aclanthology.org/2022.emnlp-main.380/
4. Wang et al. (2023), *Can Authorship Representation Learning Capture Stylistic Features?*
   https://aclanthology.org/2023.tacl-1.80/
5. Sawatphol et al. (2024), *Addressing Topic Leakage in Cross-Topic Evaluation for Authorship
   Verification* (HITS/RAVEN).
   https://aclanthology.org/2024.tacl-1.75/
6. Patel et al. (2025), *StyleDistance: Stronger Content-Independent Style Embeddings with
   Synthetic Parallel Examples*.
   https://aclanthology.org/2025.naacl-long.436/
7. Qiu et al. (2025), *mStyleDistance: Multilingual Style Embeddings and their Evaluation*.
   https://aclanthology.org/2025.findings-acl.869/
8. Kim et al. (2025), *Leveraging Multilingual Training for Authorship Representation: Enhancing
   Generalization across Languages and Domains*.
   https://aclanthology.org/2025.emnlp-main.1766/
9. Man et al. (2026), *Explainable Disentangled Representation Learning for Generalizable
   Authorship Attribution in the Era of Generative AI*.
   https://aclanthology.org/2026.acl-long.2018/
10. Fabien et al. (2020), *BertAA: BERT fine-tuning for Authorship Attribution*; reports gains
    from combining BERT and stylometric/hybrid features.
    https://aclanthology.org/2020.icon-main.16/
11. Official multilingual authorship-representation checkpoint.
    https://huggingface.co/Blablablab/multilingual-style-representation
12. Official mStyleDistance checkpoint.
    https://huggingface.co/StyleDistance/mstyledistance
13. Official English LUAR-MUD checkpoint.
    https://huggingface.co/rrivera1849/LUAR-MUD
14. Released multilingual authorship-representation training code, including PCM and
    language-aware sampling.
    https://github.com/junghwanjkim/multilingual_aa

## Model implications for StyleMatch

- Primary challenger: `Blablablab/multilingual-style-representation` (XLM-R large,
  multilingual authorship retrieval, PCM and language-aware batching).
- Content-independence anchor: `StyleDistance/mstyledistance`.
- English-only diagnostic: `rrivera1849/LUAR-MUD`; do not use it for multilingual headline
  comparisons or as a drop-in SentenceTransformer without respecting its episode input format.
- Executable English diagnostic: `gabrielloiseau/LUAR-MUD-sentence-transformers`, an unofficial
  standard SentenceTransformer conversion with safetensors. Treat it as a diagnostic rather than
  evidence about the official episode-level LUAR implementation.
- Research-stage alternative: EAVAE-style explicit content/style disentanglement; no production
  adoption without reproducible checkpoint and identical frozen-split evaluation.

## Implemented experiment contract

`colab_stylematch_method_exploration.ipynb` operationalizes the feasible comparisons. It uses
independent sources as the primary unit; matched candidate counts and theoretical chance baselines
for language comparisons; independent-source evidence curves; source-level elastic-net late
fusion; and disjoint profile-heldout dev/test verification. PCM protects the 300 most frequent
non-special subword tokens and masks other sampled-training tokens at rate 0.4, matching the
released code defaults. ARR/EAVAE are not presented as completed experiments because there is no
compatible maintained multilingual SentenceTransformer checkpoint in this pipeline.
