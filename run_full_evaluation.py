"""
Full evaluation: conformal prediction + SHAP explainability.

Runs on NSL-KDD dataset (CICIDS2017 handled separately due to download size).
Evaluates the complete trustworthy IDS pipeline:
    1. Baseline RF performance
    2. Conformal prediction sets (guaranteed coverage)
    3. SHAP feature explanations (why each decision was made)

Combines conformal prediction (coverage guarantees) with SHAP
(per-feature attribution) for both pipeline stages.

Author: Uvesh Patel
"""

import numpy as np
import os
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report

from src.data_loader import load_nsl_kdd, preprocess
from src.conformal import run_conformal_pipeline
from src.explainability import (
    compute_shap_values,
    global_feature_importance,
    explain_misclassified,
)


def main():
    print("=" * 60)
    print("FULL TRUSTWORTHY IDS EVALUATION")
    print("Conformal Prediction + SHAP Explainability")
    print("=" * 60)

    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)

    # Load data
    print("\n[1] Loading NSL-KDD...")
    df_train, df_test = load_nsl_kdd(data_dir="data")
    data = preprocess(df_train, df_test)

    X_train = data['X_train']
    X_test = data['X_test']
    y_train_binary = data['y_train_binary']
    y_test_binary = data['y_test_binary']
    y_train_multi = data['y_train_multi']
    y_test_multi = data['y_test_multi']
    feature_names = data.get('feature_cols', data.get('feature_names', None))

    attack_mask_train = y_train_binary == 1

    # ============================================================
    # PART 1: CONFORMAL PREDICTION
    # ============================================================
    print("\n" + "=" * 60)
    print("PART 1: CONFORMAL PREDICTION")
    print("Prediction sets with guaranteed coverage")
    print("=" * 60)

    # Stage 1: Binary detection
    print("\n  --- Stage 1: Binary Detection ---")
    rf_s1 = RandomForestClassifier(n_estimators=100, max_depth=20, n_jobs=-1, random_state=42)
    conformal_s1 = run_conformal_pipeline(
        rf_s1, X_train, y_train_binary, X_test, y_test_binary,
        alpha_values=[0.01, 0.05, 0.10, 0.15, 0.20]
    )

    # Show score statistics
    stats = conformal_s1['score_stats']
    print(f"\n  Calibration score statistics:")
    print(f"    Source (training CV): mean={stats['source_mean']:.4f}, std={stats['source_std']:.4f}")
    print(f"    Target (test split):  mean={stats['target_mean']:.4f}, std={stats['target_std']:.4f}")

    # Mode A: Source-calibrated (expected to fail)
    print(f"\n  MODE A: Source-calibrated (demonstrates shift problem):")
    print(f"  {'Alpha':<8} {'Coverage':<12} {'Target':<12} {'Set Size':<12} {'Singleton%':<12} {'Valid?'}")
    print(f"  {'-'*8} {'-'*12} {'-'*12} {'-'*12} {'-'*12} {'-'*6}")
    for key, metrics in conformal_s1['source_calibrated'].items():
        print(f"  {metrics['alpha']:<8.2f} "
              f"{metrics['empirical_coverage']:<12.4f} "
              f"{metrics['target_coverage']:<12.2f} "
              f"{metrics['avg_set_size']:<12.3f} "
              f"{metrics['singleton_rate']*100:<12.1f} "
              f"{'YES' if metrics['coverage_valid'] else 'NO'}")

    # Mode B: Target-calibrated (expected to work)
    print(f"\n  MODE B: Target-calibrated (valid guarantees):")
    print(f"  {'Alpha':<8} {'Coverage':<12} {'Target':<12} {'Set Size':<12} {'Singleton%':<12} {'Valid?'}")
    print(f"  {'-'*8} {'-'*12} {'-'*12} {'-'*12} {'-'*12} {'-'*6}")
    for key, metrics in conformal_s1['target_calibrated'].items():
        print(f"  {metrics['alpha']:<8.2f} "
              f"{metrics['empirical_coverage']:<12.4f} "
              f"{metrics['target_coverage']:<12.2f} "
              f"{metrics['avg_set_size']:<12.3f} "
              f"{metrics['singleton_rate']*100:<12.1f} "
              f"{'YES' if metrics['coverage_valid'] else 'NO'}")

    # Stage 2: Multi-class attack classification
    print("\n  --- Stage 2: Attack Classification ---")
    rf_s2 = RandomForestClassifier(n_estimators=100, max_depth=20, n_jobs=-1, random_state=42)
    conformal_s2 = run_conformal_pipeline(
        rf_s2, X_train[attack_mask_train], y_train_multi[attack_mask_train],
        X_test[y_test_binary == 1], y_test_multi[y_test_binary == 1],
        alpha_values=[0.01, 0.05, 0.10, 0.15, 0.20]
    )

    print(f"\n  MODE A: Source-calibrated:")
    print(f"  {'Alpha':<8} {'Coverage':<12} {'Target':<12} {'Set Size':<12} {'Singleton%':<12} {'Valid?'}")
    print(f"  {'-'*8} {'-'*12} {'-'*12} {'-'*12} {'-'*12} {'-'*6}")
    for key, metrics in conformal_s2['source_calibrated'].items():
        print(f"  {metrics['alpha']:<8.2f} "
              f"{metrics['empirical_coverage']:<12.4f} "
              f"{metrics['target_coverage']:<12.2f} "
              f"{metrics['avg_set_size']:<12.3f} "
              f"{metrics['singleton_rate']*100:<12.1f} "
              f"{'YES' if metrics['coverage_valid'] else 'NO'}")

    print(f"\n  MODE B: Target-calibrated:")
    print(f"  {'Alpha':<8} {'Coverage':<12} {'Target':<12} {'Set Size':<12} {'Singleton%':<12} {'Valid?'}")
    print(f"  {'-'*8} {'-'*12} {'-'*12} {'-'*12} {'-'*12} {'-'*6}")
    for key, metrics in conformal_s2['target_calibrated'].items():
        print(f"  {metrics['alpha']:<8.2f} "
              f"{metrics['empirical_coverage']:<12.4f} "
              f"{metrics['target_coverage']:<12.2f} "
              f"{metrics['avg_set_size']:<12.3f} "
              f"{metrics['singleton_rate']*100:<12.1f} "
              f"{'YES' if metrics['coverage_valid'] else 'NO'}")

    # ============================================================
    # PART 2: SHAP EXPLAINABILITY
    # ============================================================
    print("\n" + "=" * 60)
    print("PART 2: SHAP EXPLAINABILITY")
    print("Which features drive each decision?")
    print("=" * 60)

    # Train fresh RF models for SHAP (on full training data)
    print("\n  Training models for SHAP analysis...")
    rf_shap_s1 = RandomForestClassifier(n_estimators=100, max_depth=20, n_jobs=-1, random_state=42)
    rf_shap_s1.fit(X_train, y_train_binary)

    rf_shap_s2 = RandomForestClassifier(n_estimators=100, max_depth=20, n_jobs=-1, random_state=42)
    rf_shap_s2.fit(X_train[attack_mask_train], y_train_multi[attack_mask_train])

    # Stage 1 SHAP
    print("\n  --- Stage 1: SHAP Analysis ---")
    print("  Computing SHAP values (may take a minute)...")
    shap_s1 = compute_shap_values(rf_shap_s1, X_test, feature_names=feature_names,
                                  max_samples=300)
    importance_s1 = global_feature_importance(shap_s1, top_k=10)
    print("\n  Top 10 features driving binary detection:")
    for fname, imp in importance_s1:
        print(f"    {fname:<30} importance: {imp:.4f}")

    # Stage 2 SHAP
    print("\n  --- Stage 2: SHAP Analysis ---")
    print("  Computing SHAP values...")
    attack_test = X_test[y_test_binary == 1]
    shap_s2 = compute_shap_values(rf_shap_s2, attack_test, feature_names=feature_names,
                                  max_samples=300)
    importance_s2 = global_feature_importance(shap_s2, top_k=10)
    print("\n  Top 10 features driving attack classification:")
    for fname, imp in importance_s2:
        print(f"    {fname:<30} importance: {imp:.4f}")

    # Explain misclassified samples
    print("\n  --- Misclassification Analysis ---")
    preds_s1 = rf_shap_s1.predict(X_test)
    mis_analysis = explain_misclassified(
        rf_shap_s1, X_test, y_test_binary, preds_s1,
        feature_names=feature_names, max_samples=50
    )
    print(f"  Total misclassified: {mis_analysis['n_misclassified']} / {len(y_test_binary)}")
    if mis_analysis['n_misclassified'] > 0:
        print("  Top features causing errors:")
        for fname, imp in mis_analysis['top_error_drivers'][:5]:
            print(f"    {fname:<30} error contribution: {imp:.4f}")

    # ============================================================
    # SUMMARY
    # ============================================================
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    # Pick the alpha=0.10 result for summary
    cp_s1_src = conformal_s1['source_calibrated'].get('alpha_0.1', {})
    cp_s1_tgt = conformal_s1['target_calibrated'].get('alpha_0.1', {})
    cp_s2_tgt = conformal_s2['target_calibrated'].get('alpha_0.1', {})

    print(f"\n  CONFORMAL PREDICTION (alpha=0.10, target=90% coverage):")
    print(f"    Source-calibrated Stage 1: coverage={cp_s1_src.get('empirical_coverage', 0):.4f} "
          f"(FAILS — shift breaks guarantee)")
    print(f"    Target-calibrated Stage 1: coverage={cp_s1_tgt.get('empirical_coverage', 0):.4f}, "
          f"set_size={cp_s1_tgt.get('avg_set_size', 0):.3f}, "
          f"singleton={cp_s1_tgt.get('singleton_rate', 0)*100:.1f}%")
    print(f"    Target-calibrated Stage 2: coverage={cp_s2_tgt.get('empirical_coverage', 0):.4f}, "
          f"set_size={cp_s2_tgt.get('avg_set_size', 0):.3f}, "
          f"singleton={cp_s2_tgt.get('singleton_rate', 0)*100:.1f}%")
    print(f"\n  CP guarantees break under distribution shift.")
    print(f"  Either adapt the model (DAN) or calibrate on target domain.")

    print(f"\n  EXPLAINABILITY:")
    print(f"    Stage 1 driven by: {', '.join([f[0] for f in importance_s1[:3]])}")
    print(f"    Stage 2 driven by: {', '.join([f[0] for f in importance_s2[:3]])}")

    # Save results
    summary = {
        'conformal_stage1': conformal_s1,
        'conformal_stage2': conformal_s2,
        'shap_stage1_top_features': [(f, float(v)) for f, v in importance_s1],
        'shap_stage2_top_features': [(f, float(v)) for f, v in importance_s2],
        'misclassification_analysis': {
            'n_misclassified': mis_analysis['n_misclassified'],
            'top_error_drivers': [(f, float(v)) for f, v in mis_analysis.get('top_error_drivers', [])[:10]],
        },
    }

    output_path = os.path.join(results_dir, 'full_evaluation.json')
    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\n  Results saved to {output_path}")
    print("=" * 60)
    print("DONE")
    print("=" * 60)


if __name__ == "__main__":
    main()
