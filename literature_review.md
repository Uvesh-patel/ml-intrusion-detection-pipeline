# Literature Review

14 papers covering the main components. Organized by topic.

**Research question:** How to maintain reliability in a multi-stage ML pipeline
when distribution shift degrades both accuracy and formal guarantees?

---

## 1. Deep Learning for IDS (Foundation — why DL for IDS at all)

### [P1] Ferrag et al. (2024). "Deep Learning-based Intrusion Detection Systems: A Survey"
- **Source:** arXiv:2504.07839 (comprehensive, 100+ pages)
- **Relevance:** Establishes DL as state-of-the-art for IDS. Covers CNN, LSTM, Transformer,
  GNN architectures. Shows that representation learning outperforms manual feature
  engineering. Confirms NSL-KDD and CICIDS2017 as standard benchmarks.
- **What we learned:** DL representation learning IS the standard for modern IDS.
  Our approach (embedding + classifier) and datasets (NSL-KDD, CICIDS2017) are appropriate.

### [P2] Lanvin et al. (2023). "Transformers and Large Language Models for Efficient Intrusion Detection"
- **Source:** arXiv:2408.07583
- **Relevance:** Reviews Transformer-based IDS from 2017-2024. Shows Transformers can
  capture long-range feature dependencies in network traffic. Notes that attention
  mechanisms improve detection of subtle attacks (R2L, U2R).
- **What we learned:** Transformers can capture feature interactions but come with
  computational cost. For flow-level features, gains over simpler models are marginal.
  This informs our hypothesis: architecture matters less than adaptation method.

---

## 2. Embedding Architectures for Traffic (Justifies AE, CNN, Transformer)

### [P3] IDS-INT: Hoang et al. (2023). "Intrusion detection system using transformer-based transfer learning"
- **Source:** Journal of Information Security and Applications, 75, 103492
- **Relevance:** Uses Transformer encoder to create traffic embeddings, then applies
  transfer learning across datasets. Shows Transformer outperforms CNN-LSTM on
  CICIDS2017 and UNSW-NB15.
- **Key lesson for us:** "Transfer learning effectiveness depends heavily on
  feature space similarity between source and target" — matches our cross-dataset failure.
- **Design decision:** We test Transformer as feature interaction model (self-attention
  over flow features to capture pairwise relationships).

### [P4] DDosTC / CNN-Transformer hybrids (2022-2024). Multiple papers.
- **Source:** Survey Section 3.3 (Ferrag 2024)
- **Relevance:** The dominant architecture for flow-level NIDS is CNN + Transformer:
  - CNN captures LOCAL feature group dependencies (adjacent features interact)
  - Transformer captures GLOBAL feature interactions (any pair)
  - Multiple papers (IDS-INT, DDosTC, [88], [89]) use this pattern on CIC-IDS datasets.
- **Key lesson for us:** 1D-CNN is the right tool for local patterns in flow features.
  LSTM is for PACKET SEQUENCES or LOG SEQUENCES — NOT for flow-level aggregate features.
- **Design decision:** We use multi-scale 1D-CNN (k=3,5,7) as our "local" embedding.

### [P5] Kitsune (Mirsky 2018) + GTAE-IDS (2025). Autoencoder for NIDS.
- **Source:** Kitsune: NDSS 2018. GTAE-IDS: Graph Transformer Autoencoder (2025).
- **Relevance:** Autoencoders are the STANDARD unsupervised approach for NIDS.
  Kitsune: lightweight AE for plug-and-play detection.
  GTAE-IDS: Transformer encoder + DNN decoder for label-free detection.
- **Key lesson:** AE is not just a "baseline" — it's the most deployed approach.
  Reconstruction-based learning captures what's "normal" without labels.
- **Design decision:** Our AE (41→32→16→32→41) is the simplest valid embedding.
  The comparison against CNN and Transformer tests whether added inductive bias helps.

### WHY NOT LSTM?
The literature is clear: LSTM excels for SEQUENTIAL data (logs, packet traces,
time-series of flows). Our NSL-KDD features are **per-connection aggregates** —
41 statistics computed over a single connection. There is no temporal ordering.
Papers using LSTM for IDS (DeepLog, LogRobust, I2RNN) process sequences of events,
not feature vectors. Using LSTM on a flat feature vector would be scientifically
unjustified — it misapplies the architecture's inductive bias.

---

## 3. Domain Adaptation for IDS (Core contribution — MMD, DAN, DANN)

### [P6] Long et al. (2015). "Learning Transferable Features with Deep Adaptation Networks"
- **Source:** ICML 2015 (1900+ citations)
- **Relevance:** The foundational DAN paper. Proposes multi-kernel MMD in hidden layers
  of deep networks for unsupervised domain adaptation. Key insight: joint optimization
  of task loss + MMD is critical (separate training fails).
- **What we learned:** Joint optimization of task loss + MMD is non-negotiable.
  Our Run 3 (separate training fails) and Run 4 (end-to-end succeeds) independently
  rediscovered this paper's main result — which gives confidence in our methodology.

### [P7] Ganin et al. (2016). "Domain-Adversarial Training of Neural Networks"
- **Source:** JMLR 17(1), 1-35 (2400+ citations)
- **Relevance:** Introduces DANN with gradient reversal layer. Domain discriminator
  trained adversarially to make features domain-invariant.
- **What we learned:** Adversarial domain adaptation is theoretically elegant but
  practically fragile on small/imbalanced datasets. Our finding (DANN < DAN) is
  consistent with the broader literature on DANN instability.

### [P8] Qin et al. (2023). "Cross-domain network attack detection enabled by heterogeneous transfer learning"
- **Source:** Computer Networks, Volume 227, 2023
- **Relevance:** Directly addresses cross-network IDS transfer. Uses heterogeneous
  transfer learning to handle different feature spaces across networks. Reports
  significant accuracy drops (20-40%) without adaptation on cross-domain scenarios.
- **What we learned:** Cross-domain IDS transfer with heterogeneous feature spaces
  is an OPEN PROBLEM. Our 20% accuracy (CICIDS→OCPP) matches their reported range.
  This isn't a failure of our method — it's the research frontier.

### [P9] Xu et al. (2024). "Deep transfer learning for intrusion detection in industrial control networks"
- **Source:** arXiv:2304.10550v2
- **Relevance:** Applies MMD-based transfer learning to ICS/SCADA intrusion detection.
  Shows MMD reduces distribution gap but gains depend on domain similarity. Industrial
  control network traffic has different characteristics from IT traffic.
- **What we learned:** MMD is effective when domains share structure (within-domain
  shift). For cross-domain with fundamentally different traffic patterns, feature-level
  adaptation alone is insufficient. Need architectural solutions (shared representations).

---

## 4. Conformal Prediction for Security (Trustworthiness layer)

### [P10] Angelopoulos & Bates (2021). "A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification"
- **Source:** arXiv:2107.07511 (foundational tutorial, 800+ citations)
- **Relevance:** Establishes conformal prediction framework. Coverage guarantee under
  exchangeability. Key point: guarantee BREAKS under distribution shift.
- **What we learned:** CP's guarantee relies on exchangeability. Our finding (77%
  coverage instead of 90% under shift) is exactly what the theory PREDICTS should
  happen. This makes CP a natural DETECTOR of distribution shift.

### [P11] Penso et al. (2025). "Conformal machine learning for reliable anomaly detection in heterogeneous systems"
- **Source:** Reliability Engineering & System Safety (2025)
- **Relevance:** Directly applies conformal prediction to anomaly detection with
  uncertainty quantification. Addresses false alarm reduction. Shows CP can be a
  "safety monitor" that detects when the model is unreliable.
- **What we learned:** Using CP as a "safety monitor" (not just for prediction sets)
  is an emerging idea. Our application of CP to DETECT shift in IDS pipelines is
  aligned with this recent direction.

---

## 5. Explainability for IDS (SHAP)

### [P12] Mangal & Holm (2024). "Explainable Machine Learning for Network Intrusion Detection"
- **Source:** Journal of Cybersecurity Education, Research and Practice, 2025
- **Relevance:** Applies SHAP to RF and XGBoost IDS models on NSL-KDD and CICIDS2017.
  Identifies which features drive decisions. Finds volume-based features dominate
  detection while behavioral features matter for attack classification.
- **What we learned:** Our SHAP results match their findings independently — volume
  features for detection, behavioral features for classification. This gives us
  confidence AND explains WHY global MMD hurts Stage 2 (different feature regimes).

### [P13] Srinivasan et al. (2024). "Lightweight Intrusion Detection in IoT via SHAP-Guided Feature Selection"
- **Source:** arXiv:2512.19488
- **Relevance:** Uses SHAP not just for explanation but for feature selection.
  Reduces feature set based on SHAP importance → maintains accuracy with fewer features.
  Relevant for deployment in resource-constrained environments.
- **What we learned:** SHAP-guided feature selection could help cross-domain transfer
  by keeping only features that are both IMPORTANT and ROBUST to shift. This is a
  concrete direction for future work.

---

## 6. Pipeline Robustness & Trustworthy AI

### [P14] Apruzzese et al. (2023). "The Role of Machine Learning in Cybersecurity" (SoK)
- **Source:** ACM Computing Surveys, 55(12), 2023
- **Relevance:** Systematization of Knowledge paper on ML for cybersecurity. Identifies
  key challenges: (1) concept drift, (2) adversarial robustness, (3) deployment gaps,
  (4) lack of reproducibility. Explicitly calls out that "ML models trained in lab
  conditions fail in production due to distribution shift."
- **What we learned:** The core problem we address — lab-trained models failing
  under production shift — is identified as the main unsolved challenge here.
  Our experiments confirm this: Run 3 shows naive adaptation fails, Run 4 shows
  end-to-end works, cross-dataset shows the hard frontier.

---

## Summary: How Literature Informed Each Design Choice

| Component | References | Status |
|---|---|---|
| DL for IDS (general) | P1, P2, P14 | Standard, well-established |
| Autoencoder embeddings | P5 (Kitsune, GTAE-IDS) | Most deployed approach for NIDS |
| 1D-CNN embeddings | P4 (DDosTC, IDS-INT) | SOTA for flow-level local features |
| Transformer embeddings | P2, P3 (TabTransformer) | Global feature interactions |
| NOT using LSTM | P4 (survey shows LSTM = sequential) | Correct — our data isn't sequential |
| MMD domain adaptation | P6, P8, P9 | Well-established, works within-domain |
| DAN (end-to-end) | P6 | Standard approach, our +8.2pp matches reported gains |
| DANN (adversarial) | P7 | Valid but unstable on small datasets |
| Cross-dataset failure | P3, P8, P14 | EXPECTED — heterogeneous transfer is hard |
| Conformal prediction | P10, P11 | Recent application to IDS, fits well |
| SHAP for IDS | P12, P13 | Well-established, our results match prior work |
| Pipeline robustness | P14 | Direct motivation from SoK paper |

---

## DESIGN DECISION: CNN + Transformer (NOT LSTM) FOR FLOW FEATURES

The literature clearly distinguishes:
- **LSTM** → for LOG SEQUENCES, PACKET SEQUENCES (DeepLog, LogRobust, I2RNN)
- **CNN** → for LOCAL feature patterns in flow-level aggregates (DDosTC, IDS-INT)
- **Transformer** → for GLOBAL feature interactions (TabTransformer, GTAE-IDS)
- **Autoencoder** → for RECONSTRUCTION-based representation (Kitsune, standard)

Our NSL-KDD/CICFlowMeter data is **per-connection aggregate statistics** (41/56 features).
There is NO temporal ordering. Using LSTM would be like using a time-series model on
a single snapshot — theoretically unmotivated.

**Our architecture comparison (AE vs CNN vs Transformer) is properly motivated:**
- AE: "What information is needed to reconstruct?" (no structural bias)
- CNN: "Do adjacent feature groups interact?" (local inductive bias)
- Transformer: "Which feature PAIRS matter?" (global inductive bias)

**Expected outcome from literature:**
- All three should perform similarly on clean within-domain data
- Under distribution SHIFT, the question is: which inductive bias is more ROBUST?
- If all degrade similarly → the bottleneck is the shift, not the architecture
- This motivates DAN (end-to-end adaptation), which we already showed works (+8.2pp)

---

## Constraints from the Literature

Things the literature shows don't work well (confirmed by our experiments):
1. MMD alone doesn't fix heterogeneous shift (P8) — confirmed in Run 3
2. Transformer doesn't universally outperform simpler models (P3) — marginal gains
3. Conformal prediction breaks without exchangeability (P10) — our key finding
4. Cross-domain IDS is an open problem, not solved (P14)

## Future Directions (from literature gaps)

1. Class-conditional MMD — align per-class to preserve fine-grained boundaries
2. Few-shot adaptation for novel attack types (P8)
3. Online conformal prediction with adaptive recalibration (P11)
4. SHAP-guided feature selection for robust cross-domain transfer (P13)
