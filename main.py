"""
Two-stage ML pipeline for network intrusion detection.

Stage 1 flags suspicious traffic (binary: normal vs attack).
Stage 2 classifies the attack type (DoS, Probe, R2L, U2R).

The interesting part is what happens when Stage 1 makes mistakes --
those errors propagate into Stage 2, and we measure exactly how much
that degrades the overall system. We also test what happens when the
input data itself is noisy or incomplete.

Uses the NSL-KDD dataset.

Author: Uvesh Mahebub Patel
"""

import numpy as np
import os
import json
from src.data_loader import load_nsl_kdd, preprocess
from src.pipeline import train_stage1, train_stage2, run_full_pipeline
from src.robustness import (
    analyze_noise_robustness,
    analyze_feature_dropout,
    plot_confusion_matrices,
    plot_model_comparison,
    feature_importance_analysis,
)


def main():
    print("=" * 60)
    print("MULTI-STAGE ML PIPELINE FOR CYBER DEFENSE")
    print("Robustness Analysis of Cascading ML Components")
    print("=" * 60)

    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)

    # ---- Step 1: Load and preprocess data ----
    df_train, df_test = load_nsl_kdd(data_dir="data")
    data = preprocess(df_train, df_test)

    X_train = data['X_train']
    X_test = data['X_test']
    y_train_binary = data['y_train_binary']
    y_test_binary = data['y_test_binary']
    y_train_multi = data['y_train_multi']
    y_test_multi = data['y_test_multi']
    class_names = data['class_names']
    feature_cols = data['feature_cols']

    # ---- Step 2: Train Stage 1 (Binary: normal vs attack) ----
    s1_results, best_s1_model, best_s1_name = train_stage1(
        X_train, y_train_binary, X_test, y_test_binary
    )

    # ---- Step 3: Train Stage 2 (Multi-class: attack type) ----
    # Train only on attack samples
    attack_mask_train = y_train_binary == 1
    attack_mask_test = y_test_binary == 1

    # Stage 2 never sees 'normal' — only the 4 attack categories
    attack_class_names = [c for c in class_names if c != 'normal']

    s2_results, best_s2_model, best_s2_name = train_stage2(
        X_train[attack_mask_train], y_train_multi[attack_mask_train],
        X_test[attack_mask_test], y_test_multi[attack_mask_test],
        attack_class_names
    )

    # ---- Step 4: Run full pipeline (Stage 1 -> Stage 2) ----
    pipeline_metrics = run_full_pipeline(
        best_s1_model, best_s2_model, X_test,
        y_test_binary, y_test_multi, class_names
    )

    # ---- Step 5: Generate visualizations ----
    print("\n" + "=" * 60)
    print("GENERATING VISUALIZATIONS")
    print("=" * 60)

    plot_confusion_matrices(s1_results, s2_results, attack_class_names, results_dir)
    plot_model_comparison(s1_results, s2_results, results_dir)

    # Feature importance (use Random Forest models)
    s1_rf = s1_results['Random Forest']['model']
    s2_rf = s2_results['Random Forest']['model']
    feature_importance_analysis(s1_rf, s2_rf, feature_cols, class_names, results_dir)

    # ---- Step 6: Robustness analysis ----
    noise_results = analyze_noise_robustness(
        best_s1_model, best_s2_model, X_test,
        y_test_binary, y_test_multi, class_names, results_dir
    )

    dropout_results = analyze_feature_dropout(
        best_s1_model, best_s2_model, X_test,
        y_test_binary, y_test_multi, class_names, results_dir
    )

    # ---- Step 7: Save summary ----
    summary = {
        'best_stage1_model': best_s1_name,
        'best_stage2_model': best_s2_name,
        'stage1_results': {
            name: {
                'accuracy': r['metrics']['accuracy'],
                'f1_weighted': r['metrics']['f1_weighted'],
                'precision_macro': r['metrics']['precision_macro'],
                'recall_macro': r['metrics']['recall_macro'],
                'train_time': r['metrics']['train_time'],
            } for name, r in s1_results.items()
        },
        'stage2_results': {
            name: {
                'accuracy': r['metrics']['accuracy'],
                'f1_weighted': r['metrics']['f1_weighted'],
                'precision_macro': r['metrics']['precision_macro'],
                'recall_macro': r['metrics']['recall_macro'],
                'train_time': r['metrics']['train_time'],
            } for name, r in s2_results.items()
        },
        'pipeline_metrics': pipeline_metrics,
    }

    with open(os.path.join(results_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)
    print(f"\nResults saved to: {os.path.abspath(results_dir)}/")
    print("  - confusion_matrices.png")
    print("  - model_comparison.png")
    print("  - feature_importance.png")
    print("  - noise_robustness.png")
    print("  - dropout_robustness.png")
    print("  - summary.json")


if __name__ == "__main__":
    main()
