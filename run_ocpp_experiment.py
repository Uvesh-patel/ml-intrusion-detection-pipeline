"""
OCPP 1.6 WebSocket domain adaptation experiment.

This experiment demonstrates domain adaptation on real WebSocket traffic
from EV charging stations. The OCPP dataset has 3 separate clients
(charging stations), creating a natural cross-environment shift.

Experiment structure:
    1. Baseline: train on Client 1, test on Client 2 (no adaptation)
    2. DAN adaptation: align Client 1 and Client 2 embeddings via MMD
    3. Conformal prediction: test coverage guarantees with/without shift
    4. SHAP: explain which features drive detection in WebSocket traffic

This validates that the domain adaptation pipeline works on modern
WebSocket-based IoT protocol traffic, not just traditional benchmarks.

Dataset:
    Dalamagkas et al. (2025) "Federated Detection of Open Charge Point
    Protocol 1.6 Cyberattacks" arXiv:2502.01569

Author: Uvesh Patel
"""

import os
import json
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import train_test_split

from src.data_loader_ocpp import load_cross_station, load_ocpp_combined, preprocess_ocpp
from src.dan_model import train_dan, predict
from src.conformal import run_conformal_pipeline
from src.explainability import compute_shap_values, global_feature_importance


def run_baseline(data, task="binary"):
    """Train RF on source, test on target without adaptation."""
    if task == "binary":
        y_train = data["y_source_binary"]
        y_test = data["y_target_binary"]
    else:
        y_train = data["y_source_multi"]
        y_test = data["y_target_multi"]

    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(data["X_source"], y_train)

    y_pred = rf.predict(data["X_target"])
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")

    return rf, acc, f1, y_pred


def run_same_station_baseline(data):
    """Train and test on the same station (no shift). Upper bound."""
    X = data["X_source"]
    y = data["y_source_binary"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")
    return acc, f1


def run_dan_adaptation(data, task="binary"):
    """Apply DAN to align source and target distributions."""
    if task == "binary":
        y_source = data["y_source_binary"]
        y_target = data["y_target_binary"]
        n_classes = 2
    else:
        y_source = data["y_source_multi"]
        y_target = data["y_target_multi"]
        n_classes = len(data["class_names"])

    feat_ext, classifier = train_dan(
        X_source=data["X_source"],
        y_source=y_source,
        X_target=data["X_target"],
        num_classes=n_classes,
        epochs=100,
        lr=0.001,
        mmd_weight=1.0,
        batch_size=64,
    )

    y_pred = predict(feat_ext, classifier, data["X_target"])
    acc = accuracy_score(y_target, y_pred)
    f1 = f1_score(y_target, y_pred, average="weighted")
    return (feat_ext, classifier), acc, f1, y_pred


def run_conformal_on_ocpp(data):
    """Run conformal prediction to test coverage guarantees under shift."""
    X_source = data["X_source"]
    y_source = data["y_source_binary"]
    X_target = data["X_target"]
    y_target = data["y_target_binary"]

    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)

    results = run_conformal_pipeline(
        model=rf,
        X_train=X_source,
        y_train=y_source,
        X_test=X_target,
        y_test=y_target,
    )
    return results


def run_shap_analysis(data, rf_model):
    """Run SHAP to explain which WebSocket/network features matter."""
    X_test = data["X_target"][:200]  # subsample for speed
    feature_names = data["feature_names"]

    shap_result = compute_shap_values(rf_model, X_test, feature_names)
    top_features = global_feature_importance(shap_result, top_k=10)
    return top_features


def main():
    print("=" * 60)
    print("OCPP 1.6 WebSocket Domain Adaptation Experiment")
    print("=" * 60)

    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)

    # --- Cross-station experiment: Client 1 -> Client 2 ---
    print("\n[1] Loading OCPP data: Client 1 (source) -> Client 2 (target)")
    data = load_cross_station(
        source_client=1, target_client=2, layer="tcp"
    )

    # Same-station baseline (upper bound, no shift)
    print("\n[2] Same-station baseline (no shift)...")
    same_acc, same_f1 = run_same_station_baseline(data)
    print(f"    Accuracy: {same_acc:.4f}, F1: {same_f1:.4f}")

    # Cross-station baseline (with shift, no adaptation)
    print("\n[3] Cross-station baseline (shift, no adaptation)...")
    rf_model, cross_acc, cross_f1, _ = run_baseline(data, task="binary")
    print(f"    Accuracy: {cross_acc:.4f}, F1: {cross_f1:.4f}")
    print(f"    Performance drop due to shift: {same_acc - cross_acc:.4f}")

    # DAN adaptation
    print("\n[4] DAN domain adaptation (Client 1 -> Client 2)...")
    _, dan_acc, dan_f1, _ = run_dan_adaptation(data, task="binary")
    print(f"    Accuracy: {dan_acc:.4f}, F1: {dan_f1:.4f}")
    print(f"    Recovery from adaptation: +{dan_acc - cross_acc:.4f}")

    # Multi-class
    print("\n[5] Multi-class attack classification...")
    _, multi_acc_base, multi_f1_base, _ = run_baseline(data, task="multi")
    print(f"    Baseline (no adapt): Acc={multi_acc_base:.4f}, F1={multi_f1_base:.4f}")
    _, multi_acc_dan, multi_f1_dan, _ = run_dan_adaptation(data, task="multi")
    print(f"    DAN adapted:         Acc={multi_acc_dan:.4f}, F1={multi_f1_dan:.4f}")

    # Conformal prediction
    print("\n[6] Conformal prediction (coverage under shift)...")
    cp_results = run_conformal_on_ocpp(data)
    src_cov = cp_results['source_calibrated'].get('alpha_0.1', {}).get('empirical_coverage', 'N/A')
    tgt_cov = cp_results['target_calibrated'].get('alpha_0.1', {}).get('empirical_coverage', 'N/A')
    print(f"    Source-calibrated coverage (alpha=0.10): {src_cov}")
    print(f"    Target-calibrated coverage (alpha=0.10): {tgt_cov}")

    # SHAP
    print("\n[7] SHAP feature importance...")
    top_features = run_shap_analysis(data, rf_model)
    print("    Top features for WebSocket IDS:")
    for feat, importance in top_features[:5]:
        print(f"      {feat}: {importance:.4f}")

    # --- Also test Client 1 -> Client 3 (different shift) ---
    print("\n[8] Robustness: Client 1 -> Client 3...")
    data_13 = load_cross_station(source_client=1, target_client=3, layer="tcp")
    _, acc_13_base, _, _ = run_baseline(data_13, task="binary")
    _, acc_13_dan, _, _ = run_dan_adaptation(data_13, task="binary")
    print(f"    Baseline: {acc_13_base:.4f}")
    print(f"    DAN:      {acc_13_dan:.4f}")
    print(f"    Recovery: +{acc_13_dan - acc_13_base:.4f}")

    # Save results
    results = {
        "dataset": "OCPP 1.6 WebSocket (Zenodo, Dalamagkas et al. 2025)",
        "experiment": "cross_station_domain_adaptation",
        "source": "Client 1",
        "target": "Client 2",
        "feature_layer": "TCP/IP (CICFlowMeter)",
        "n_features": len(data["feature_names"]),
        "same_station_accuracy": round(same_acc, 4),
        "cross_station_baseline_accuracy": round(cross_acc, 4),
        "cross_station_dan_accuracy": round(dan_acc, 4),
        "performance_drop_from_shift": round(same_acc - cross_acc, 4),
        "recovery_from_dan": round(dan_acc - cross_acc, 4),
        "multiclass_baseline_f1": round(multi_f1_base, 4),
        "multiclass_dan_f1": round(multi_f1_dan, 4),
        "conformal_prediction": cp_results,
        "top_shap_features": top_features[:10],
        "client_1_to_3_baseline": round(acc_13_base, 4),
        "client_1_to_3_dan": round(acc_13_dan, 4),
    }

    out_path = os.path.join(results_dir, "ocpp_experiment_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Same-station (no shift):      {same_acc:.4f}")
    print(f"  Cross-station (with shift):   {cross_acc:.4f}")
    print(f"  Cross-station + DAN:          {dan_acc:.4f}")
    print(f"  Shift causes drop of:         {same_acc - cross_acc:.4f}")
    print(f"  DAN recovers:                 +{dan_acc - cross_acc:.4f}")


if __name__ == "__main__":
    main()
