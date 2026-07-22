# Part 8 expanded-corpus top-four retraining: literature audit

Date: 2026-07-22

## Implementation-to-literature map

| Part 8 decision | Academic basis | Scope in StyleMatch |
| --- | --- | --- |
| Refit the trainable top-four method families after corpus expansion | Cawley & Talbot (2010) | Encoder and classical views fit train sources; the reranker fits dev scores; the pretrained encoder remains an unchanged control. |
| Keep positive pairs source-separated | Project training protocol; Kim, Zhang & Jurgens (2025) | A profile contributes fine-tuning pairs only when at least two independent train sources exist. Lower-support profiles remain zero-shot candidates. |
| Normalize heterogeneous retrieval outputs before combining them | Montague & Aslam (2001); Kittler et al. (1998) | The learned reranker receives within-language z-scores and candidate percentiles. These are rank features, not calibrated probabilities. |
| Select weights on dev and open test once | Cawley & Talbot (2010) | Prevents optimizing the reported test result through repeated weight search. |
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
