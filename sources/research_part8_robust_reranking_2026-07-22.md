# Part 8 expanded-corpus top-four retraining: literature audit

Date: 2026-07-22

## Implementation-to-literature map

| Part 8 decision | Academic basis | Scope in StyleMatch |
| --- | --- | --- |
| Refit trainable representations after corpus expansion, then reuse their frozen scores for reranker search | Cawley & Talbot (2010) | Encoder and classical views fit train sources; neural reranker selection is grouped within dev; the pretrained encoder remains an unchanged control. |
| Keep positive pairs source-separated | Project training protocol; Kim, Zhang & Jurgens (2025) | A profile contributes fine-tuning pairs only when at least two independent train sources exist. Lower-support profiles remain zero-shot candidates. |
| Normalize heterogeneous retrieval outputs before combining them | Montague & Aslam (2001); Kittler et al. (1998) | The learned reranker receives within-language z-scores and candidate percentiles. These are rank features, not calibrated probabilities. |
| Select architecture and stopping epoch by profile-grouped dev cross-validation; open test once | Cawley & Talbot (2010) | Prevents optimizing the reported test result through repeated architecture or epoch search. |
| Optimize candidate lists rather than independent binary labels | Cao et al. (2007) | A listwise softmax objective is combined with an anchor-hard-negative pairwise loss. |
| Learn conditional evidence weights while bounding deviation from the strongest view | Jacobs et al. (1991); project safeguard | A small gating network weights views per candidate; its output remains a residual around the fine-tuned centroid score. |
| Compare systems on paired evidence | Dror et al. (2018) | MRR changes are computed for the same profiles under the base and fused systems. |
| Preserve within-profile dependence in uncertainty estimates | Cameron, Gelbach & Miller (2008) | The author-language profile is resampled as the cluster. The code uses a nonparametric paired profile bootstrap, not the paper’s wild cluster bootstrap-t procedure. |
| Guard performance across language and corpus groups | Sagawa et al. (2020) | A validation constraint inspired by worst-group robustness. StyleMatch does not train with Group DRO. |
| Require calibration non-degradation | Guo et al. (2017) | ECE is checked separately from ranking performance. |
| Require selective precision non-degradation | Geifman & El-Yaniv (2017) | Precision at 50% coverage operationalizes the risk–coverage trade-off. |

## Primary sources

1. Kittler, J., Hatef, M., Duin, R. P. W., & Matas, J. (1998). On Combining Classifiers. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 20(3), 226–239. https://doi.org/10.1109/34.667881
2. Montague, M. H., & Aslam, J. A. (2001). Relevance Score Normalization for Metasearch. *Proceedings of CIKM 2001*, 427–433. https://doi.org/10.1145/502585.502657
3. Cameron, A. C., Gelbach, J. B., & Miller, D. L. (2008). Bootstrap-Based Improvements for Inference with Clustered Errors. *The Review of Economics and Statistics*, 90(3), 414–427. https://doi.org/10.1162/rest.90.3.414
4. Cawley, G. C., & Talbot, N. L. C. (2010). On Over-fitting in Model Selection and Subsequent Selection Bias in Performance Evaluation. *Journal of Machine Learning Research*, 11, 2079–2107. https://www.jmlr.org/papers/v11/cawley10a.html
5. Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). On Calibration of Modern Neural Networks. *Proceedings of ICML 2017*, 1321–1330. https://proceedings.mlr.press/v70/guo17a.html
6. Geifman, Y., & El-Yaniv, R. (2017). Selective Classification for Deep Neural Networks. *Advances in Neural Information Processing Systems 30*, 4885–4894. https://proceedings.neurips.cc/paper/2017/hash/4a8423d5e91fda00bb7e46540e2b0cf1-Abstract.html
7. Dror, R., Baumer, G., Shlomov, S., & Reichart, R. (2018). The Hitchhiker’s Guide to Testing Statistical Significance in Natural Language Processing. *Proceedings of ACL 2018*, 1383–1392. https://doi.org/10.18653/v1/P18-1128
8. Sagawa, S., Koh, P. W., Hashimoto, T. B., & Liang, P. (2020). Distributionally Robust Neural Networks for Group Shifts: On the Importance of Regularization for Worst-Case Generalization. *ICLR 2020*. https://openreview.net/forum?id=ryxGuJrFvS
9. Kim, H., Zhang, Y., & Jurgens, D. (2025). Leveraging Multilingual Training for Authorship Representation: Enhancing Generalization across Languages and Domains. *Proceedings of EMNLP 2025*. https://aclanthology.org/2025.emnlp-main.1766/

## Claims deliberately not made

- Percentile conversion does not calibrate scores into probabilities.
- Subgroup non-degradation is not a formal Group DRO objective or guarantee.
- The profile bootstrap does not make dependent chunks or sources independent; it preserves their profile-level grouping during resampling.
- These sources motivate the protocol. They do not establish that fusion improves StyleMatch; only the locked Part 8 artifact can do that.

## Frontier indicator audit

The next reranker should test a small set of conditionally complementary views rather than add every available score:

1. **Work/source multi-prototypes and profile dispersion.** These retain within-author heterogeneity hidden by a single centroid and directly target failures where a known passage is closer to a neighboring author profile.
2. **Boundary-consistency features.** Score several sentence-aligned crops of the same query and measure both mean similarity and rank instability. This is a robustness diagnostic and potential rank feature; it must not force copied text to score 1.0.
3. **Delexicalized or probabilistically content-masked representations.** These target topic leakage. Kim, Zhang & Jurgens (2025) motivate probabilistic content masking and language-aware batching for multilingual authorship representations.
4. **Layer-mixture representations.** A lightweight learned mixture of transformer layers is a challenger for out-of-domain robustness, not an automatic replacement for the selected encoder.
5. **Disaggregated classical views.** Delexicalized character patterns, function words, punctuation/rhythm, and syntax should enter separately only when grouped out-of-fold ablations show residual value beyond the neural scores.

Topic similarity, exact-source overlap, publication decade, demographic metadata, and generated style labels must remain outside the author Style Match rank. Source overlap can support a separate provenance warning; margin, dispersion, and instability are primarily confidence or rejection features.

Additional primary sources:

10. Wegmann, A., Schraagen, M., & Nguyen, D. (2024). Addressing Topic Leakage in Cross-Topic Evaluation for Authorship Verification. *Transactions of the Association for Computational Linguistics*. https://aclanthology.org/2024.tacl-1.75/
11. Wegmann, A., Schraagen, M., & Nguyen, D. (2023). Can Authorship Representation Learning Capture Stylistic Features? *Transactions of the Association for Computational Linguistics*. https://aclanthology.org/2023.tacl-1.80/
12. Halvani, O., Winter, C., & Graner, L. (2018). On the Usefulness of Compression Models for Authorship Verification. Related delexicalization evidence is summarized by *What Represents “Style” in Authorship Attribution?* https://aclanthology.org/C18-1238/
13. Rivera-Soto, R. A., Miano, A., Ordonez, J., Chen, B. Y., Khan, A., Bishop, M., & Andrews, N. (2021). Learning Universal Authorship Representations. https://aclanthology.org/2021.emnlp-main.712/
14. Zhang, Y., Kim, H., & Jurgens, D. (2025). Layered Insights: Exploring the Representation of Writing Style in Transformer Models. *Proceedings of EMNLP 2025*. https://aclanthology.org/2025.emnlp-main.521/
15. Cao, Z., Qin, T., Liu, T.-Y., Tsai, M.-F., & Li, H. (2007). Learning to Rank: From Pairwise Approach to Listwise Approach. *Proceedings of ICML 2007*. https://doi.org/10.1145/1273496.1273513
16. Jacobs, R. A., Jordan, M. I., Nowlan, S. J., & Hinton, G. E. (1991). Adaptive Mixtures of Local Experts. *Neural Computation*, 3(1), 79–87. https://doi.org/10.1162/neco.1991.3.1.79
