"""
Confidence-aware pipeline analysis.

Compares two setups:
  A) RF baseline: RF for Stage 1, RF for Stage 2
  B) Best combo: DAN for Stage 1 (85.5%), RF for Stage 2 (78.1%)

For each, evaluates:
  - Accuracy at different coverage levels (selective prediction)
  - Calibration (are confidence scores reliable?)
  - Threshold sweep (tradeoff between accuracy and human workload)
  - Confidence-gated full pipeline

Author: Uvesh Patel
"""

import numpy as np
import os
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

from src.data_loader import load_nsl_kdd, preprocess
from src.confidence import (
    get_confidence,
    get_confidence_neural,
    accuracy_at_coverage,
    calibration_analysis,
    sweep_thresholds,
)
from src.dan_model import train_dan, predict


def analyze_stage1(name, preds, confidence, y_true):
    """Run full confidence analysis for a Stage 1 model."""
    print(f"\n  --- {name} ---")

    acc = accuracy_score(y_true, preds)
    f1 = f1_score(y_true, preds, average='weighted')
    print(f"  Accuracy (all): {acc:.4f}, F1: {f1:.4f}")
    print(f"  Confidence: mean={confidence.mean():.3f}, "
          f"min={confidence.min():.3f}, median={np.median(confidence):.3f}")

    # Accuracy-coverage
    print(f"\n  Accuracy-Coverage:")
    coverage_data = accuracy_at_coverage(y_true, preds, confidence)
    for item in coverage_data:
        cov = item['coverage']
        if abs(cov - 0.5) < 0.01 or abs(cov - 0.7) < 0.01 or \
           abs(cov - 0.8) < 0.01 or abs(cov - 0.9) < 0.01 or abs(cov - 1.0) < 0.01:
            print(f"    Coverage {cov:.0%}: acc={item['accuracy']:.4f}")

    # Calibration
    cal_data = calibration_analysis(y_true, confidence, preds)
    print(f"\n  Calibration (ECE): {cal_data['ece']:.4f}")

    # Threshold sweep
    print(f"\n  Threshold sweep:")
    print(f"    {'Thresh':<8} {'Accuracy':<10} {'Flagged%':<10}")
    print(f"    {'-'*8} {'-'*10} {'-'*10}")

    n = len(y_true)
    threshold_data = []
    for t in np.arange(0.5, 0.99, 0.05):
        mask = confidence >= t
        if mask.sum() == 0:
            continue
        t_acc = accuracy_score(y_true[mask], preds[mask])
        flagged_pct = (1 - mask.mean()) * 100
        threshold_data.append({
            'threshold': float(t),
            'accuracy': float(t_acc),
            'flagged_percent': float(flagged_pct),
            'n_classified': int(mask.sum()),
        })
        print(f"    {t:<8.2f} {t_acc:<10.4f} {flagged_pct:<10.1f}")

    # Find best operating point
    best = None
    for t in threshold_data:
        if t['accuracy'] > acc + 0.02 and t['flagged_percent'] < 30:
            best = t
            break

    return {
        'accuracy_all': float(acc),
        'f1_all': float(f1),
        'confidence_mean': float(confidence.mean()),
        'accuracy_coverage': coverage_data,
        'calibration': cal_data,
        'threshold_sweep': threshold_data,
        'best_operating_point': best,
    }


def main():
    print("=" * 60)
    print("CONFIDENCE-AWARE PIPELINE ANALYSIS")
    print("Comparing: RF baseline vs DAN+RF best combo")
    print("=" * 60)

    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)

    # Load data
    df_train, df_test = load_nsl_kdd(data_dir="data")
    data = preprocess(df_train, df_test)

    X_train = data['X_train']
    X_test = data['X_test']
    y_train_binary = data['y_train_binary']
    y_test_binary = data['y_test_binary']
    y_train_multi = data['y_train_multi']
    y_test_multi = data['y_test_multi']

    attack_mask_train = y_train_binary == 1

    # ============================================================
    # PART A: RF Baseline
    # ============================================================
    print("\n" + "=" * 60)
    print("PART A: RF Baseline Pipeline")
    print("=" * 60)

    print("\n  Training RF Stage 1...")
    rf_s1 = RandomForestClassifier(n_estimators=100, max_depth=20, n_jobs=-1, random_state=42)
    rf_s1.fit(X_train, y_train_binary)

    print("  Training RF Stage 2...")
    rf_s2 = RandomForestClassifier(n_estimators=100, max_depth=20, n_jobs=-1, random_state=42)
    rf_s2.fit(X_train[attack_mask_train], y_train_multi[attack_mask_train])

    preds_rf, conf_rf = get_confidence(rf_s1, X_test)
    rf_results = analyze_stage1("RF Stage 1", preds_rf, conf_rf, y_test_binary)

    # ============================================================
    # PART B: DAN + RF (best combo from Run 4)
    # ============================================================
    print("\n" + "=" * 60)
    print("PART B: DAN Stage 1 + RF Stage 2 (best combo)")
    print("=" * 60)

    print("\n  Training DAN Stage 1 (end-to-end, 80 epochs)...")
    dan_feat, dan_clf = train_dan(
        X_train, y_train_binary, X_test, num_classes=2,
        epochs=80, mmd_weight=0.5
    )

    preds_dan, conf_dan = get_confidence_neural(dan_feat, dan_clf, X_test)
    dan_results = analyze_stage1("DAN Stage 1", preds_dan, conf_dan, y_test_binary)

    # ============================================================
    # COMPARISON
    # ============================================================
    print("\n" + "=" * 60)
    print("COMPARISON: RF vs DAN with Confidence Gating")
    print("=" * 60)

    print(f"\n  {'Metric':<35} {'RF':<12} {'DAN':<12}")
    print(f"  {'-'*35} {'-'*12} {'-'*12}")
    print(f"  {'Accuracy (all samples)':<35} {rf_results['accuracy_all']:<12.4f} {dan_results['accuracy_all']:<12.4f}")
    print(f"  {'F1 (all samples)':<35} {rf_results['f1_all']:<12.4f} {dan_results['f1_all']:<12.4f}")
    print(f"  {'Mean confidence':<35} {rf_results['confidence_mean']:<12.3f} {dan_results['confidence_mean']:<12.3f}")
    print(f"  {'Calibration error (ECE)':<35} {rf_results['calibration']['ece']:<12.4f} {dan_results['calibration']['ece']:<12.4f}")

    # Best operating points
    print(f"\n  Best operating points (>2pp improvement, <30% flagged):")
    if rf_results['best_operating_point']:
        bp = rf_results['best_operating_point']
        print(f"    RF:  threshold={bp['threshold']:.2f}, acc={bp['accuracy']:.4f}, "
              f"flagged={bp['flagged_percent']:.1f}%")
    else:
        print(f"    RF:  no significant improvement at reasonable flagging rates")

    if dan_results['best_operating_point']:
        bp = dan_results['best_operating_point']
        print(f"    DAN: threshold={bp['threshold']:.2f}, acc={bp['accuracy']:.4f}, "
              f"flagged={bp['flagged_percent']:.1f}%")
    else:
        print(f"    DAN: no significant improvement at reasonable flagging rates")

    # Final story
    print(f"\n  PROGRESSION:")
    print(f"    1. RF baseline (no adaptation):        {rf_results['accuracy_all']:.4f}")
    print(f"    2. DAN (domain adaptation):            {dan_results['accuracy_all']:.4f}")
    if dan_results['best_operating_point']:
        bp = dan_results['best_operating_point']
        print(f"    3. DAN + confidence gating (t={bp['threshold']:.2f}): {bp['accuracy']:.4f}")
        print(f"       (flagging {bp['flagged_percent']:.1f}% for human review)")

    # Save
    summary = {
        'rf_baseline': rf_results,
        'dan_best_combo': dan_results,
    }

    output_path = os.path.join(results_dir, 'confidence_analysis.json')
    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n  Results saved to {output_path}")
    print("=" * 60)
    print("DONE")
    print("=" * 60)


if __name__ == "__main__":
    main()
