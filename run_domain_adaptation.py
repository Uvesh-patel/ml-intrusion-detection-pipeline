"""
Domain adaptation experiment.

Compares four approaches:
1. Baseline: Random Forest on raw features (no adaptation)
2. AE + MMD (separate): autoencoder trained with MMD, then RF on embeddings
3. DAN (end-to-end): shared encoder + classifier + MMD, jointly trained
4. DANN (end-to-end): shared encoder + classifier + domain discriminator
   with gradient reversal, jointly trained

The NSL-KDD test set has a natural domain shift (different attack proportions,
unseen attack types). End-to-end methods should outperform the separate approach
because the encoder learns features that are both discriminative AND domain-invariant.

Author: Uvesh Patel
"""

import numpy as np
import os
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

from src.data_loader import load_nsl_kdd, preprocess
from src.embedding import train_autoencoder, get_embeddings
from src.domain_adaptation import train_with_mmd, measure_domain_shift
from src.dan_model import train_dan, train_dann, predict


def eval_rf(X_train, y_train, X_test, y_test, label=""):
    """Train RF on given features and evaluate."""
    clf = RandomForestClassifier(n_estimators=100, max_depth=20, n_jobs=-1, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    print(f"    {label} — Acc: {acc:.4f}, F1: {f1:.4f}")
    return acc, f1


def eval_neural(feat_ext, classifier, X_test, y_test, label=""):
    """Evaluate a DAN/DANN model."""
    y_pred = predict(feat_ext, classifier, X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    print(f"    {label} — Acc: {acc:.4f}, F1: {f1:.4f}")
    return acc, f1


def run_stage(stage_name, X_train, y_train, X_test, y_test, num_classes):
    """Run all 4 methods on one stage and return results dict."""
    print(f"\n{'='*60}")
    print(f"  {stage_name}")
    print(f"{'='*60}")

    results = {}

    # [A] Baseline
    print("\n  [A] Baseline (RF on raw features):")
    acc, f1 = eval_rf(X_train, y_train, X_test, y_test, "Baseline")
    results['baseline'] = {'accuracy': acc, 'f1': f1}

    # [B] AE + MMD (separate)
    print("\n  [B] AE + MMD (separate encoder, then RF on embeddings):")
    mmd_model = train_with_mmd(X_train, X_test, epochs=50, embedding_dim=16, mmd_weight=1.0)
    X_tr_emb = get_embeddings(mmd_model, X_train)
    X_te_emb = get_embeddings(mmd_model, X_test)
    acc, f1 = eval_rf(X_tr_emb, y_train, X_te_emb, y_test, "AE+MMD separate")
    results['ae_mmd_separate'] = {'accuracy': acc, 'f1': f1}

    # [C] DAN (end-to-end)
    print("\n  [C] DAN (end-to-end: encoder + classifier + MMD):")
    dan_feat, dan_clf = train_dan(
        X_train, y_train, X_test, num_classes=num_classes,
        epochs=80, mmd_weight=0.5
    )
    acc, f1 = eval_neural(dan_feat, dan_clf, X_test, y_test, "DAN")
    results['dan'] = {'accuracy': acc, 'f1': f1}

    # [D] DANN (end-to-end with gradient reversal)
    print("\n  [D] DANN (end-to-end: encoder + classifier + domain discriminator):")
    dann_feat, dann_clf = train_dann(
        X_train, y_train, X_test, num_classes=num_classes,
        epochs=80
    )
    acc, f1 = eval_neural(dann_feat, dann_clf, X_test, y_test, "DANN")
    results['dann'] = {'accuracy': acc, 'f1': f1}

    return results


def main():
    print("=" * 60)
    print("DOMAIN ADAPTATION EXPERIMENT")
    print("Baseline vs AE+MMD vs DAN vs DANN")
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
    class_names = data['class_names']

    attack_mask_train = y_train_binary == 1
    attack_mask_test = y_test_binary == 1

    # Stage 1: Binary detection
    s1 = run_stage(
        "STAGE 1: Binary Detection (normal vs attack)",
        X_train, y_train_binary, X_test, y_test_binary,
        num_classes=2
    )

    # Stage 2: Attack classification (attack samples only)
    n_attack_classes = len([c for c in class_names if c != 'normal'])
    s2 = run_stage(
        "STAGE 2: Attack Classification (DoS/Probe/R2L/U2R)",
        X_train[attack_mask_train], y_train_multi[attack_mask_train],
        X_test[attack_mask_test], y_test_multi[attack_mask_test],
        num_classes=n_attack_classes
    )

    # Summary
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    print(f"\n  {'Method':<30} {'S1 Acc':>8} {'S1 F1':>8} {'S2 Acc':>8} {'S2 F1':>8}")
    print(f"  {'-'*30} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    for key in ['baseline', 'ae_mmd_separate', 'dan', 'dann']:
        label = {
            'baseline': 'Baseline (RF, raw)',
            'ae_mmd_separate': 'AE+MMD (separate)',
            'dan': 'DAN (end-to-end)',
            'dann': 'DANN (end-to-end)',
        }[key]
        print(f"  {label:<30} {s1[key]['accuracy']:>8.4f} {s1[key]['f1']:>8.4f} "
              f"{s2[key]['accuracy']:>8.4f} {s2[key]['f1']:>8.4f}")

    # Save
    summary = {'stage1': s1, 'stage2': s2}
    with open(os.path.join(results_dir, 'domain_adaptation_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n  Saved to {results_dir}/domain_adaptation_summary.json")
    print("=" * 60)
    print("DONE")
    print("=" * 60)


if __name__ == "__main__":
    main()
