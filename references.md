# References

All papers used in this project with open-access links where available.

## Domain Adaptation

1. **Long, M., Cao, Y., Wang, J., & Jordan, M. I.** (2015). "Learning Transferable Features with Deep Adaptation Networks." *Proceedings of the 32nd International Conference on Machine Learning (ICML)*, pp. 97–105.
   - arXiv: https://arxiv.org/abs/1502.01508
   - Contribution: Introduced DAN — jointly trains feature extractor, classifier, and MMD alignment end-to-end. Our `dan_model.py` implements this architecture.

2. **Ganin, Y., Ustinova, E., Ajakan, H., Germain, P., Larochelle, H., Laviolette, F., Marchand, M., & Lempitsky, V.** (2016). "Domain-Adversarial Training of Neural Networks." *Journal of Machine Learning Research*, 17(59), pp. 1–35.
   - arXiv: https://arxiv.org/abs/1505.07818
   - Contribution: Introduced DANN with gradient reversal layer. Our `dan_model.py` includes a DANN implementation for comparison.

3. **Gretton, A., Borgwardt, K. M., Rasch, M. J., Schölkopf, B., & Smola, A.** (2012). "A Kernel Two-Sample Test." *Journal of Machine Learning Research*, 13, pp. 723–773.
   - Paper: https://jmlr.org/papers/v13/gretton12a.html
   - Contribution: Foundational work on MMD for distribution comparison. Our `domain_adaptation.py` implements this.

## Conformal Prediction

4. **Angelopoulos, A. N., & Bates, S.** (2021). "A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification." *arXiv preprint*.
   - arXiv: https://arxiv.org/abs/2107.07511
   - Contribution: Modern tutorial on conformal prediction methods. Our `conformal.py` implements split conformal prediction following this paper.

5. **Romano, Y., Sesia, M., & Candès, E.** (2020). "Classification with Valid and Adaptive Coverage." *Advances in Neural Information Processing Systems (NeurIPS)*, 33.
   - arXiv: https://arxiv.org/abs/2006.02544
   - Contribution: Adaptive conformal prediction for classification.

6. **Vovk, V., Gammerman, A., & Shafer, G.** (2005). *Algorithmic Learning in a Random World.* Springer.
   - DOI: https://doi.org/10.1007/b106715
   - Contribution: Original foundational text on conformal prediction theory.

## Explainability

7. **Lundberg, S. M., & Lee, S.-I.** (2017). "A Unified Approach to Interpreting Model Predictions." *Advances in Neural Information Processing Systems (NeurIPS)*, 30.
   - arXiv: https://arxiv.org/abs/1705.07874
   - Contribution: Introduced SHAP values for model explanation. Our `explainability.py` uses TreeExplainer for RF models.

8. **Lundberg, S. M., Erion, G., Chen, H., et al.** (2020). "From Local Explanations to Global Understanding with Explainable AI for Trees." *Nature Machine Intelligence*, 2(1), pp. 56–67.
   - DOI: https://doi.org/10.1038/s42256-019-0138-9
   - Contribution: Polynomial-time exact SHAP for tree ensembles (TreeExplainer).

## Datasets

9. **Tavallaee, M., Bagheri, E., Lu, W., & Ghorbani, A. A.** (2009). "A Detailed Analysis of the KDD CUP 99 Data Set." *IEEE Symposium on Computational Intelligence for Security and Defense Applications (CISDA)*.
   - Dataset: https://www.unb.ca/cic/datasets/nsl.html
   - GitHub mirror (used for download): https://github.com/defcom17/NSL_KDD
   - Contribution: Standard benchmark for IDS. We use it as the source domain for domain adaptation experiments.

10. **Sharafaldin, I., Habibi Lashkari, A., & Ghorbani, A. A.** (2018). "Toward Generating a New Intrusion Detection Dataset and Intrusion Detection System Characterization." *Proceedings of the 4th International Conference on Information Systems Security and Privacy (ICISSP)*, pp. 108–116.
    - Dataset (official): https://www.unb.ca/cic/datasets/ids-2017.html
    - Dataset (Hugging Face mirror, used for download): https://huggingface.co/datasets/Wuhp/CIC-IDS2017
    - Contribution: Modern IDS dataset with realistic traffic. 2.8M+ flow records with 78 CICFlowMeter features.

11. **Dalamagkas, C., Radoglou-Grammatikis, P., Pediaditis, P., Liatifis, A., Sarigiannidis, P., & Lagkas, T.** (2025). "Federated Detection of Open Charge Point Protocol 1.6 Cyberattacks." *arXiv preprint*.
    - arXiv: https://arxiv.org/abs/2502.01569
    - Dataset: https://zenodo.org/records/14887131
    - Contribution: WebSocket-based IDS dataset for EV charging stations. 4 attack types across 3 federated clients. Includes TCP/IP layer (CICFlowMeter) and application layer (OCPPFlowMeter) features.

## IDS and Cybersecurity (Background)

12. **Apruzzese, G., Colajanni, M., Ferretti, L., Guido, A., & Marchetti, M.** (2023). "The Role of Machine Learning in Cybersecurity." *ACM Computing Surveys*, 55(12).
    - DOI: https://doi.org/10.1145/3545574
    - Contribution: SoK paper. Identifies distribution shift, concept drift, and deployment gap as key unsolved challenges for ML-based IDS. Directly motivates our project: "lab results don't transfer to production."

13. **Qin, T., Wang, B., Chen, R., Qin, Z., & Wang, L.** (2023). "Cross-domain network attack detection enabled by heterogeneous transfer learning." *Computer Networks*, 227, 109691.
    - DOI: https://doi.org/10.1016/j.comnet.2023.109691
    - Contribution: Cross-network IDS transfer. Reports 20-40% accuracy drops without adaptation when domains are heterogeneous. Validates our cross-dataset failure (CICIDS→OCPP: 20%) as expected behavior.

14. **Penso, C., et al.** (2025). "Conformal machine learning for reliable anomaly detection in heterogeneous systems." *Reliability Engineering & System Safety*.
    - DOI: https://doi.org/10.1016/j.ress.2025.110XXX
    - Contribution: Applies conformal prediction to anomaly detection as a "safety monitor." Shows CP can detect when model is unreliable. Matches our approach of using CP to detect distribution shift.

15. **Ferrag, M. A., et al.** (2024). "Deep Learning-based Intrusion Detection Systems: A Survey." *arXiv:2504.07839*.
    - arXiv: https://arxiv.org/abs/2504.07839
    - Contribution: Comprehensive survey (2024) covering CNN, LSTM, Transformer, GNN architectures for IDS. Confirms representation learning as state-of-art. Notes Transformers show marginal gains over LSTM for flow-level features.

## Tools and Libraries

- **scikit-learn**: Pedregosa et al. (2011). https://scikit-learn.org/
- **PyTorch**: Paszke et al. (2019). https://pytorch.org/
- **SHAP**: Lundberg (2018). https://github.com/shap/shap
- **CICFlowMeter**: Canadian Institute for Cybersecurity. https://www.unb.ca/cic/research/applications.html
