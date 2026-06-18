"""
CICIDS2017 experiment: validate pipeline on modern dataset.

Runs the full pipeline (baseline RF + conformal prediction + SHAP)
on CICIDS2017 to check if results hold on a different, modern dataset.

CICIDS2017 contains 2.8M+ flow records from 2017 with modern attacks:
    DoS/DDoS, Brute Force, Web Attacks, Port Scanning, Botnet, etc.

We sample 30% for manageable runtime and run:
    1. Baseline RF (binary + multi-class)
    2. Conformal prediction (target-calibrated)
    3. SHAP feature importance

Dataset source:
    Sharafaldin, Lashkari, Ghorbani. "Toward Generating a New Intrusion
    Detection Dataset and Intrusion Traffic Characterization" (ICISSP 2018).
    Canadian Institute for Cybersecurity, University of New Brunswick.

Author: Uvesh Patel
"""

import numpy as np
import os
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report

from src.data_loader_cicids import load_cicids2017, preprocess_cicids
from src.conformal import run_conformal_pipeline
from src.explainability import compute_shap_values, global_feature_importance


def main():
    print("=" * 60)
    print("CICIDS2017 EXPERIMENT")
    print("Validating pipeline on modern dataset")
    print("=" * 60)

    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)

    # Load CICIDS2017
    print("\n[1] Loading CICIDS2017...")
    df = load_cicids2017(data_dir="data", sample_fraction=0.3)

    print("\n[2] Preprocessing...")
    data = preprocess_cicids(df, test_fraction=0.3)

    X_train = data['X_train']
    X_test = data['X_test']
    y_train_binary = data['y_train_binary']
    y_test_binary = data['y_test_binary']
    y_train_multi = data['y_train_multi']
    y_test_multi = data['y_test_multi']
    feature_names = data['feature_names']
    class_names = data['class_names']

    attack_mask_train = y_train_binary == 1

    # ============================================================
    # BASELINE RF
    # ============================================================
    print("\n" + "=" * 60)
    print("[3] BASELINE: Random Forest")
    print("=" * 60)

    # Stage 1: Binary
    print("\n  Training Stage 1 (binary)...")
    rf_s1 = RandomForestClassifier(n_estimators=100, max_depth=20, n_jobs=-1, random_state=42)
    rf_s1.fit(X_train, y_train_binary)
    preds_s1 = rf_s1.predict(X_test)
    acc_s1 = accuracy_score(y_test_binary, preds_s1)
    f1_s1 = f1_score(y_test_binary, preds_s1, average='weighted')
    print(f"  Stage 1: acc={acc_s1:.4f}, F1={f1_s1:.4f}")

    # Stage 2: Multi-class
    print("\n  Training Stage 2 (multi-class on attack samples)...")
    rf_s2 = RandomForestClassifier(n_estimators=100, max_depth=20, n_jobs=-1, random_state=42)
    rf_s2.fit(X_train[attack_mask_train], y_train_multi[attack_mask_train])
    attack_test_mask = y_test_binary == 1
    preds_s2 = rf_s2.predict(X_test[attack_test_mask])
    acc_s2 = accuracy_score(y_test_multi[attack_test_mask], preds_s2)
    f1_s2 = f1_score(y_test_multi[attack_test_mask], preds_s2, average='weighted')
    print(f"  Stage 2: acc={acc_s2:.4f}, F1={f1_s2:.4f}")
    print(f"\n  Classification report (Stage 2):")
    print(classification_report(y_test_multi[attack_test_mask], preds_s2,
                                target_names=[c for i, c in enumerate(class_names)
                                              if i != class_names.index('Benign')]
                                if 'Benign' in class_names else class_names))

    # ============================================================
    # CONFORMAL PREDICTION
    # ============================================================
    print("\n" + "=" * 60)
    print("[4] CONFORMAL PREDICTION")
    print("=" * 60)

    print("\n  Stage 1 (binary)...")
    rf_cp = RandomForestClassifier(n_estimators=100, max_depth=20, n_jobs=-1, random_state=42)
    cp_results = run_conformal_pipeline(
        rf_cp, X_train, y_train_binary, X_test, y_test_binary,
        alpha_values=[0.05, 0.10, 0.15]
    )

    print(f"\n  Target-calibrated results:")
    print(f"  {'Alpha':<8} {'Coverage':<12} {'Target':<12} {'Set Size':<12} {'Valid?'}")
    print(f"  {'-'*8} {'-'*12} {'-'*12} {'-'*12} {'-'*6}")
    for key, metrics in cp_results['target_calibrated'].items():
        print(f"  {metrics['alpha']:<8.2f} "
              f"{metrics['empirical_coverage']:<12.4f} "
              f"{metrics['target_coverage']:<12.2f} "
              f"{metrics['avg_set_size']:<12.3f} "
              f"{'YES' if metrics['coverage_valid'] else 'NO'}")

    # ============================================================
    # SHAP EXPLAINABILITY
    # ============================================================
    print("\n" + "=" * 60)
    print("[5] SHAP EXPLAINABILITY")
    print("=" * 60)

    print("\n  Computing SHAP values (Stage 1)...")
    shap_result = compute_shap_values(rf_s1, X_test, feature_names=feature_names,
                                      max_samples=200)
    importance = global_feature_importance(shap_result, top_k=10)
    print("\n  Top 10 features driving detection on CICIDS2017:")
    for fname, imp in importance:
        print(f"    {fname:<35} importance: {imp:.4f}")

    # ============================================================
    # SUMMARY
    # ============================================================
    print("\n" + "=" * 60)
    print("SUMMARY: CICIDS2017 Results")
    print("=" * 60)
    print(f"\n  Stage 1 (Binary): acc={acc_s1:.4f}, F1={f1_s1:.4f}")
    print(f"  Stage 2 (Multi):  acc={acc_s2:.4f}, F1={f1_s2:.4f}")
    print(f"  Conformal (alpha=0.10): coverage={cp_results['target_calibrated'].get('alpha_0.1', {}).get('empirical_coverage', 0):.4f}")
    print(f"  Top features: {', '.join([f[0] for f in importance[:3]])}")

    # Save
    summary = {
        'dataset': 'CICIDS2017',
        'baseline': {
            'stage1_acc': float(acc_s1),
            'stage1_f1': float(f1_s1),
            'stage2_acc': float(acc_s2),
            'stage2_f1': float(f1_s2),
        },
        'conformal': cp_results,
        'shap_top_features': importance,
    }

    output_path = os.path.join(results_dir, 'cicids2017_results.json')
    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\n  Results saved to {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
