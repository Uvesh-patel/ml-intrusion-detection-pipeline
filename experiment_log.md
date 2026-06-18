# Experiment Log

Tracking what was tried, what happened, and what was learned.


## Run 1: Baseline Pipeline (main.py)

**Goal:** Establish baseline performance of two-stage IDS pipeline on NSL-KDD.

**Setup:** Stage 1 (binary: normal/attack) with RF, SVM, MLP. Stage 2 (multi-class: DoS/Probe/R2L/U2R) with same models. Evaluate individually and end-to-end.

**Results:**
- Stage 1 best: SVM at 79.8% accuracy, 79.7% F1
- Stage 2 best: RF at 78.1% accuracy, 78.0% F1
- End-to-end: only 62% of attacks correctly detected AND classified
- Gap between Stage 2 isolated (93%) and pipeline (62%) = Stage 1 misses propagate

**Learned:** Errors cascade non-linearly through multi-stage systems. A 20% miss rate at Stage 1 causes 31% degradation at the pipeline level. This is the core problem.


## Run 2: Robustness Analysis

**Goal:** Quantify how the pipeline degrades under realistic noise conditions.

**Results:**
- Gaussian noise sigma=0.1: Stage 1 drops 4%, Stage 2 drops 25%
- Gaussian noise sigma=0.5: pipeline nearly useless (28% E2E)
- Feature dropout 30%: Stage 2 holds (82%) but Stage 1 drops, dragging E2E down

**Learned:** Stage 2 is fragile to noise but tolerant of missing features. Stage 1 is the opposite. The pipeline's weakest link determines overall reliability. This motivates domain adaptation (reduce the shift) and confidence estimation (know when predictions are unreliable).


## Run 3: Domain Adaptation — Separate AE + MMD

**Goal:** Use autoencoder embeddings + MMD alignment to reduce distribution shift between train (source) and test (target) sets.

**Setup:** Train autoencoder with MMD loss (lambda=1.0) separately, then classify embeddings with RF.

**Results:**
- Stage 1: AE+MMD 77.2% vs baseline 77.3% (no improvement, just recovered compression loss)
- Stage 2: AE+MMD 77.0% vs baseline 78.1% (slightly worse)
- MMD shift reduced 95.9% (0.022 → 0.0009)

**Learned:** Reducing MMD between source and target embeddings does NOT automatically improve classification. The problem: the encoder was trained for reconstruction + alignment, but NOT for discrimination. The features are domain-invariant but not necessarily useful for classification. Need end-to-end training where classification loss and MMD are optimized jointly.


## Run 4: Domain Adaptation — DAN and DANN (end-to-end)

**Goal:** Fix the separate training problem by jointly optimizing encoder + classifier + domain alignment.

**Setup:**
- DAN: shared feature extractor (41→64→32) + classifier (32→16→classes) + MMD penalty (lambda=0.5), trained 80 epochs
- DANN: same architecture + domain discriminator with gradient reversal layer, trained 80 epochs

**Results:**

Stage 1 (Binary Detection):
- Baseline (RF, raw):   77.3% acc, 77.0% F1
- AE+MMD (separate):    78.4% acc, 78.1% F1
- DAN (end-to-end):     85.5% acc, 85.6% F1  ← +8.2pp over baseline
- DANN (end-to-end):    80.4% acc, 80.3% F1  ← +3.1pp over baseline

Stage 2 (Attack Classification):
- Baseline (RF, raw):   78.1% acc, 78.0% F1
- AE+MMD (separate):    77.2% acc, 73.6% F1
- DAN (end-to-end):     76.0% acc, 70.0% F1
- DANN (end-to-end):    77.7% acc, 73.3% F1

**Learned:**
1. End-to-end training confirmed as superior. DAN massively improved Stage 1 detection (+8.2pp). The encoder now learns features that are both discriminative and domain-invariant.
2. DANN also improved Stage 1 but less than DAN. The domain discriminator reached near-chance accuracy (0.693 ≈ log(2)), meaning features became domain-invariant, but the adversarial signal was weaker than the direct MMD penalty.
3. Stage 2 was NOT improved by any adaptation method. The 4-class problem with extreme imbalance (DoS: 45k vs U2R: 52) makes global distribution alignment counterproductive — it blurs the fine-grained boundaries between rare attack types.
4. Practical conclusion: use DAN for Stage 1, keep RF for Stage 2. Each stage gets the best method for its characteristics.

**Open question for Stage 2:** Would class-conditional MMD (align distributions per-class) help? Or is the fundamental issue that there aren't enough samples of rare classes to align?


## Run 5: Confidence-Aware Pipeline (completed)

**Goal:** Add uncertainty estimation so the pipeline knows when it might be wrong.

**Setup:** Compare RF baseline vs DAN+RF best combo with confidence thresholding.

**Results:**
- Progression: RF baseline (77.3%) → DAN (84.1%) → DAN + gating t=0.90 (86.3%)
- DAN flags only 5.3% of samples for human review to gain that extra +2.2pp
- RF needs to flag 27.8% to reach 94.6% — much less operationally efficient
- DAN is better calibrated: ECE = 0.14 vs RF's ECE = 0.37

**Learned:** Domain-adapted models are not only more accurate, they're more honest about uncertainty. DAN's softmax outputs better reflect true correctness probability.


## Run 6: Conformal Prediction + SHAP Explainability (completed)

**Questions arising from Runs 4-5:**
- Run 4 showed DAN helps Stage 1 but NOT Stage 2. Why? What's different?
- Run 5 showed DAN is better calibrated. But can we get FORMAL guarantees?
- If the pipeline is deployed, how does an operator KNOW it's unreliable?

**Goal:** (a) Conformal prediction: formal coverage guarantees that DETECT shift.
(b) SHAP: explain WHY Stage 2 doesn't benefit from global domain adaptation.

**Setup:**
- Conformal prediction: two modes compared
  - Source-calibrated: cross-validation scores from training data
  - Target-calibrated: split test into 30% calibration + 70% evaluation
- SHAP: TreeExplainer on RF models for Stage 1 and Stage 2

**Conformal Prediction Results:**

Source-calibrated (alpha=0.10, target=90%):
  Stage 1 coverage: 77.3% — GUARANTEE BROKEN
  Stage 2 coverage: 78.1% — GUARANTEE BROKEN
  All sets are singletons (model overconfident on source domain)
  Calibration scores: mean=0.0038 (model almost never wrong on training data)

Target-calibrated (alpha=0.10, target=90%):
  Stage 1 coverage: 90.7% — GUARANTEE HOLDS ✓
  Stage 2 coverage: 89.7% — GUARANTEE HOLDS (marginal) ✓
  Stage 1: set_size=1.20, singleton=80.5% (informative predictions)
  Stage 2: set_size=1.43, singleton=64.0% (more uncertain, expected)

Main finding: conformal prediction's coverage guarantee requires exchangeability.
Distribution shift between NSL-KDD train/test violates this assumption,
causing coverage to fail (77% instead of 90%). This shows quantitatively
that domain adaptation is needed for safety guarantees to hold.

**SHAP Results:**

Stage 1 top features: src_bytes (0.104), dst_bytes (0.061), dst_host_srv_count (0.042)
Stage 2 top features: count (0.034), dst_host_serror_rate (0.031), dst_host_same_src_port_rate (0.025)

Key insight: Stages 1 and 2 rely on DIFFERENT features. Stage 1 uses volume
(bytes transferred), Stage 2 uses behavior patterns (connection counts, error rates).

This explains Run 4's result: global MMD alignment works for Stage 1 because
it aligns the volume-dominated distributions. But Stage 2 needs fine-grained
behavioral patterns preserved — global alignment disrupts class boundaries.
A possible fix: class-conditional MMD (align per-class, not globally).

Misclassification analysis: 5119/22544 errors (22.7%). Top error-driving features
are the same volume features, suggesting the model struggles with low-volume
attacks (R2L, U2R) that don't have the clear byte-count signature of DoS/Probe.

**Learned:**
1. Conformal prediction exposes distribution shift quantitatively. Not just
   "accuracy drops" but "mathematical guarantees become invalid." Stronger argument.
2. SHAP reveals the model's decision logic. Volume features dominate detection
   but behavioral features matter for classification. Different signal per stage.
3. Together, domain adaptation + conformal prediction + SHAP give the pipeline
   improved accuracy, formal guarantees, and interpretability.


## Run 7: Does Encoder Architecture Affect Domain Adaptation?

**Question arising from Run 4:** DAN works because it jointly trains encoder +
classifier + MMD alignment. But Run 4 used a simple feedforward encoder (41→64→32).
Does the encoder's inductive bias matter? If the encoder can't capture the RIGHT
structure, can MMD still align the distributions usefully?

**Hypothesis:** The encoder architecture should NOT significantly affect results
for flow-level features, because:
1. Flow features are aggregate statistics (not sequential → LSTM inappropriate)
2. The 41 features are low-dimensional → even simple encoders can represent them
3. Run 3 already showed: the problem is joint optimization, not encoder capacity

If confirmed, this supports the Run 4 conclusion: the training procedure
(end-to-end DAN) matters more than the encoder architecture.

**Setup:**
- Three architectures, same embedding_dim=16, same training (reconstruction + MMD):
  - Autoencoder: feedforward, no structural bias (our baseline from Run 3)
  - 1D-CNN (multi-scale k=3,5,7): captures local feature group patterns
  - Transformer encoder: captures global feature interactions via self-attention
- Each trained: (a) reconstruction only, (b) reconstruction + MMD (lambda=1.0)
- Downstream: RF classifier on embeddings → accuracy/F1 on NSL-KDD test set
- Measure: MMD between train/test embeddings

**Why NOT LSTM:** Flow features are per-connection aggregates — 41 statistics
computed over one connection. No temporal ordering exists. LSTM is for sequential
data (log sequences, packet traces). Literature confirms: CNN and Transformer are
appropriate for flow-level features (DDosTC, IDS-INT, GTAE-IDS).

**Results:** (to be filled after running)

**Interpretation guide:**
- If all three perform similarly → architecture is NOT the bottleneck → confirms Run 4
- If Transformer+MMD > AE+MMD → self-attention captures shift-invariant structure
- If CNN+MMD > AE+MMD → local feature groups contain shift-invariant patterns
- ANY result is informative for the core question


---

## Summary of Findings

1. Multi-stage IDS pipelines suffer from cascading errors (77% → 62% E2E)
2. Separate embedding + alignment does not improve classification (Run 3)
3. End-to-end training (DAN) is required: +8.2pp for binary detection
4. Global alignment helps detection but hurts multi-class — different feature
   regimes require different adaptation strategies
5. Conformal prediction coverage breaks under shift (77% vs 90% target)
   — quantitative evidence that exchangeability violations are the core issue
6. Architecture choice (AE vs CNN vs Transformer) is expected to be secondary
   to the training procedure; Run 7 tests this hypothesis

## Open Questions

- Class-conditional MMD for Stage 2 (align per-class instead of globally)
- Online recalibration of conformal prediction without target labels
- Cross-dataset transfer with heterogeneous feature spaces (CICIDS→OCPP: 20%)
