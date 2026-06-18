"""
Cross-dataset domain adaptation: CICIDS2017 -> OCPP WebSocket.

Tests domain adaptation under severe distribution shift.
Both datasets use CICFlowMeter for feature extraction (58 shared features),
but they come from completely different networks:
    - CICIDS2017: enterprise office network (2017)
    - OCPP: EV charging station WebSocket traffic (2024)

The distribution shift is real: different protocols, traffic volumes,
attack strategies, and network environments. A model trained on one
should degrade on the other without adaptation.

Experiment:
    1. Baseline RF: train on CICIDS2017 (source), test on OCPP (target)
    2. DAN adaptation: align CICIDS2017 and OCPP embeddings
    3. Conformal prediction: test if coverage guarantees hold across datasets
    4. Compare with in-domain (OCPP-only) baseline as upper bound

If adaptation works here, it means models trained on one network can be
reused in a different environment without complete retraining.

Author: Uvesh Patel
"""

import os
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import train_test_split

from src.data_loader_ocpp import (
    load_ocpp_combined, get_shared_features_with_cicids, OCPP_DATA_DIR
)
from src.data_loader_cicids import load_cicids2017, preprocess_cicids
from src.dan_model import train_dan, predict
from src.conformal import run_conformal_pipeline


def load_cicids_shared_features(sample_fraction=0.3):
    """Load CICIDS2017 with only the features shared with OCPP."""
    print("  Loading CICIDS2017...")
    df = load_cicids2017(data_dir="data", sample_fraction=sample_fraction)

    # Clean column names
    df.columns = df.columns.str.strip()

    # Get shared features
    shared = get_shared_features_with_cicids()
    available = [c for c in shared if c in df.columns]
    print(f"  Shared features found: {len(available)}/{len(shared)}")

    # Extract features
    X = df[available].copy()
    X = X.apply(pd.to_numeric, errors="coerce")
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    X.fillna(0, inplace=True)

    # Binary label: BENIGN=0, anything else=1
    labels = df["Label"].str.strip() if "Label" in df.columns else df["label"].str.strip()
    y_binary = (labels != "BENIGN").astype(int).values

    return X.values, y_binary, available


def load_ocpp_shared_features():
    """Load OCPP (combined) with only the features shared with CICIDS2017."""
    print("  Loading OCPP combined data...")
    train_df, test_df = load_ocpp_combined(layer="tcp")
    df = pd.concat([train_df, test_df], ignore_index=True)

    shared = get_shared_features_with_cicids()
    available = [c for c in shared if c in df.columns]
    print(f"  Shared features found: {len(available)}/{len(shared)}")

    X = df[available].copy()
    X = X.apply(pd.to_numeric, errors="coerce")
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    X.fillna(0, inplace=True)

    # Binary label
    labels = df["Label"].str.strip().str.lower()
    y_binary = (labels != "normal").astype(int).values

    return X.values, y_binary, available


def main():
    print("=" * 60)
    print("CROSS-DATASET EXPERIMENT: CICIDS2017 -> OCPP WebSocket")
    print("=" * 60)

    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)

    # Load both datasets with shared features
    print("\n[1] Loading datasets with shared CICFlowMeter features...")
    X_cicids, y_cicids, feat_cicids = load_cicids_shared_features(sample_fraction=0.1)
    X_ocpp, y_ocpp, feat_ocpp = load_ocpp_shared_features()

    # Use only features present in BOTH (intersection)
    common = [f for f in feat_cicids if f in feat_ocpp]
    idx_cicids = [feat_cicids.index(f) for f in common]
    idx_ocpp = [feat_ocpp.index(f) for f in common]
    X_cicids = X_cicids[:, idx_cicids]
    X_ocpp = X_ocpp[:, idx_ocpp]
    feat_names = common

    print(f"\n  CICIDS2017 (source): {X_cicids.shape[0]} samples, {X_cicids.shape[1]} features")
    print(f"  OCPP (target): {X_ocpp.shape[0]} samples, {X_ocpp.shape[1]} features")
    print(f"  Common features: {len(feat_names)}")
    print(f"  CICIDS2017 attack ratio: {y_cicids.mean():.3f}")
    print(f"  OCPP attack ratio: {y_ocpp.mean():.3f}")

    # Scale both with source scaler
    scaler = StandardScaler()
    X_source = scaler.fit_transform(X_cicids)
    X_target = scaler.transform(X_ocpp)

    # --- Upper bound: train and test on OCPP (no shift) ---
    print("\n[2] Upper bound: train/test within OCPP (no shift)...")
    X_ocpp_train, X_ocpp_test, y_ocpp_train, y_ocpp_test = train_test_split(
        X_target, y_ocpp, test_size=0.3, random_state=42, stratify=y_ocpp
    )
    rf_upper = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf_upper.fit(X_ocpp_train, y_ocpp_train)
    upper_acc = accuracy_score(y_ocpp_test, rf_upper.predict(X_ocpp_test))
    upper_f1 = f1_score(y_ocpp_test, rf_upper.predict(X_ocpp_test), average="weighted")
    print(f"    Accuracy: {upper_acc:.4f}, F1: {upper_f1:.4f}")

    # --- Baseline: train on CICIDS2017, test on OCPP ---
    print("\n[3] Cross-dataset baseline: CICIDS2017 -> OCPP (no adaptation)...")
    rf_cross = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf_cross.fit(X_source, y_cicids)
    y_pred_cross = rf_cross.predict(X_target)
    cross_acc = accuracy_score(y_ocpp, y_pred_cross)
    cross_f1 = f1_score(y_ocpp, y_pred_cross, average="weighted")
    print(f"    Accuracy: {cross_acc:.4f}, F1: {cross_f1:.4f}")
    print(f"    Drop from shift: {upper_acc - cross_acc:.4f}")

    # --- DAN: adapt CICIDS2017 -> OCPP ---
    print("\n[4] DAN domain adaptation: CICIDS2017 -> OCPP...")
    feat_ext, classifier = train_dan(
        X_source=X_source,
        y_source=y_cicids,
        X_target=X_target,
        num_classes=2,
        epochs=80,
        lr=0.001,
        mmd_weight=1.0,
        batch_size=256,
    )
    y_pred_dan = predict(feat_ext, classifier, X_target)
    dan_acc = accuracy_score(y_ocpp, y_pred_dan)
    dan_f1 = f1_score(y_ocpp, y_pred_dan, average="weighted")
    print(f"    Accuracy: {dan_acc:.4f}, F1: {dan_f1:.4f}")
    print(f"    Recovery: +{dan_acc - cross_acc:.4f}")

    # --- Conformal prediction under cross-dataset shift ---
    print("\n[5] Conformal prediction (cross-dataset shift)...")
    rf_cp = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    cp_results = run_conformal_pipeline(
        model=rf_cp,
        X_train=X_source,
        y_train=y_cicids,
        X_test=X_target,
        y_test=y_ocpp,
    )

    src_cov = cp_results['source_calibrated'].get('alpha_0.1', {}).get('empirical_coverage', None)
    tgt_cov = cp_results['target_calibrated'].get('alpha_0.1', {}).get('empirical_coverage', None)
    print(f"    Source-calibrated coverage (alpha=0.10): {src_cov:.4f}")
    print(f"    Target-calibrated coverage (alpha=0.10): {tgt_cov:.4f}")
    if src_cov and src_cov < 0.90:
        print(f"    >>> COVERAGE GUARANTEE VIOLATED under cross-dataset shift!")

    # Save results
    results = {
        "experiment": "cross_dataset_domain_adaptation",
        "source": "CICIDS2017 (enterprise network, 2017)",
        "target": "OCPP 1.6 WebSocket (EV charging, 2024)",
        "shared_features": len(feat_names),
        "feature_names": feat_names,
        "source_samples": int(X_source.shape[0]),
        "target_samples": int(X_target.shape[0]),
        "upper_bound_accuracy": round(upper_acc, 4),
        "cross_dataset_baseline_accuracy": round(cross_acc, 4),
        "cross_dataset_dan_accuracy": round(dan_acc, 4),
        "shift_drop": round(upper_acc - cross_acc, 4),
        "dan_recovery": round(dan_acc - cross_acc, 4),
        "conformal_source_calibrated_coverage": round(src_cov, 4) if src_cov else None,
        "conformal_target_calibrated_coverage": round(tgt_cov, 4) if tgt_cov else None,
        "conformal_full": cp_results,
    }

    out_path = os.path.join(results_dir, "cross_dataset_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Results saved to {out_path}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY: Cross-Dataset Domain Adaptation")
    print("=" * 60)
    print(f"  In-domain (OCPP only):        {upper_acc:.4f}")
    print(f"  Cross-dataset (no adapt):     {cross_acc:.4f}")
    print(f"  Cross-dataset + DAN:          {dan_acc:.4f}")
    print(f"  Performance drop (shift):     -{upper_acc - cross_acc:.4f}")
    print(f"  DAN recovery:                 +{dan_acc - cross_acc:.4f}")
    print(f"  CP source-calibrated:         {src_cov:.4f} (target: 0.90)")
    print(f"  CP target-calibrated:         {tgt_cov:.4f} (target: 0.90)")


if __name__ == "__main__":
    main()
