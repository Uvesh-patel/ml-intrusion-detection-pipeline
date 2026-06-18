"""
Embedding Architecture Comparison — Autoencoder vs 1D-CNN vs Transformer

Compares three embedding strategies with different inductive biases:
1. Autoencoder: reconstruction-based, no structural assumption
2. Multi-scale 1D-CNN: captures local feature group dependencies
3. Transformer encoder: captures global feature interactions via attention

Each is evaluated on:
  - Downstream classification accuracy (RF on embeddings)
  - Domain shift (MMD between train/test embeddings)
  - Effect of MMD alignment during training

The goal is to check whether the encoder architecture matters for
robustness under distribution shift, or whether the training procedure
(end-to-end adaptation) is what actually makes the difference.

Usage: python run_embedding_comparison.py
"""

import numpy as np
import json
import os
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
import torch

from src.data_loader import load_nsl_kdd, preprocess
from src.embedding import TrafficAutoencoder, train_autoencoder, get_embeddings
from src.embedding_sequential import (
    train_sequential_embedding, get_embeddings_sequential
)
from src.domain_adaptation import compute_mmd


def compute_mmd_numpy(X_source, X_target, bandwidth=1.0):
    """Compute MMD between numpy arrays (for evaluation)."""
    source = torch.FloatTensor(X_source)
    target = torch.FloatTensor(X_target)
    mmd_val = compute_mmd(source, target, bandwidth=bandwidth)
    return mmd_val.item()


def evaluate_embedding(emb_train, emb_test, y_train, y_test, name):
    """Train RF on embeddings and evaluate."""
    rf = RandomForestClassifier(n_estimators=100, max_depth=20,
                                n_jobs=-1, random_state=42)
    rf.fit(emb_train, y_train)
    y_pred = rf.predict(emb_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
    return acc, f1


def main():
    print("=" * 70)
    print("EMBEDDING ARCHITECTURE COMPARISON")
    print("Autoencoder vs 1D-CNN vs Transformer for IDS Pipeline")
    print("=" * 70)

    # Load data
    print("\n[1/5] Loading NSL-KDD dataset...")
    df_train, df_test = load_nsl_kdd()
    data = preprocess(df_train, df_test)
    X_train = data['X_train']
    X_test = data['X_test']
    y_train_binary = data['y_train_binary']
    y_test_binary = data['y_test_binary']
    feature_names = data['feature_cols']
    input_dim = X_train.shape[1]
    print(f"  Train: {X_train.shape}, Test: {X_test.shape}")
    print(f"  Input dimension: {input_dim}")

    # Baseline: RF on raw features
    print("\n[2/5] Baseline (RF on raw features)...")
    rf_raw = RandomForestClassifier(n_estimators=100, max_depth=20,
                                     n_jobs=-1, random_state=42)
    rf_raw.fit(X_train, y_train_binary)
    raw_acc = accuracy_score(y_test_binary, rf_raw.predict(X_test))
    raw_f1 = f1_score(y_test_binary, rf_raw.predict(X_test), average='macro', zero_division=0)
    print(f"  Raw features → Acc: {raw_acc:.4f}, F1: {raw_f1:.4f}")

    # MMD between raw train and test (quantify shift)
    raw_mmd = compute_mmd_numpy(X_train[:2000], X_test[:2000])
    print(f"  Raw feature MMD (train vs test): {raw_mmd:.6f}")

    # ---- Compare embeddings ----
    embedding_dim = 16
    epochs = 50
    results = {
        'raw_features': {
            'accuracy': raw_acc, 'f1': raw_f1, 'mmd': raw_mmd,
            'params': 0, 'train_time': 0
        }
    }

    # 3A: Autoencoder
    print("\n[3/5] Training embedding models...")
    print("\n  --- Autoencoder ---")
    t0 = time.time()
    ae_model = train_autoencoder(X_train, epochs=epochs, embedding_dim=embedding_dim)
    ae_time = time.time() - t0
    emb_train_ae = get_embeddings(ae_model, X_train)
    emb_test_ae = get_embeddings(ae_model, X_test)
    ae_acc, ae_f1 = evaluate_embedding(emb_train_ae, emb_test_ae,
                                        y_train_binary, y_test_binary, "AE")
    ae_mmd = compute_mmd_numpy(emb_train_ae[:2000], emb_test_ae[:2000])
    ae_params = sum(p.numel() for p in ae_model.parameters())
    print(f"  AE → Acc: {ae_acc:.4f}, F1: {ae_f1:.4f}, MMD: {ae_mmd:.6f}")
    print(f"  Parameters: {ae_params:,}, Time: {ae_time:.1f}s")
    results['autoencoder'] = {
        'accuracy': ae_acc, 'f1': ae_f1, 'mmd': ae_mmd,
        'params': ae_params, 'train_time': ae_time
    }

    # 3B: 1D-CNN (multi-scale)
    print("\n  --- 1D-CNN (Multi-Scale) ---")
    t0 = time.time()
    cnn_model, cnn_history = train_sequential_embedding(
        X_train, model_type='cnn', epochs=epochs, embedding_dim=embedding_dim
    )
    cnn_time = time.time() - t0
    emb_train_cnn = get_embeddings_sequential(cnn_model, X_train)
    emb_test_cnn = get_embeddings_sequential(cnn_model, X_test)
    cnn_acc, cnn_f1 = evaluate_embedding(emb_train_cnn, emb_test_cnn,
                                          y_train_binary, y_test_binary, "CNN")
    cnn_mmd = compute_mmd_numpy(emb_train_cnn[:2000], emb_test_cnn[:2000])
    cnn_params = sum(p.numel() for p in cnn_model.parameters())
    print(f"  CNN → Acc: {cnn_acc:.4f}, F1: {cnn_f1:.4f}, MMD: {cnn_mmd:.6f}")
    print(f"  Parameters: {cnn_params:,}, Time: {cnn_time:.1f}s")
    results['cnn'] = {
        'accuracy': cnn_acc, 'f1': cnn_f1, 'mmd': cnn_mmd,
        'params': cnn_params, 'train_time': cnn_time
    }

    # 3C: Transformer
    print("\n  --- Transformer Encoder ---")
    t0 = time.time()
    tf_model, tf_history = train_sequential_embedding(
        X_train, model_type='transformer', epochs=epochs, embedding_dim=embedding_dim
    )
    tf_time = time.time() - t0
    emb_train_tf = get_embeddings_sequential(tf_model, X_train)
    emb_test_tf = get_embeddings_sequential(tf_model, X_test)
    tf_acc, tf_f1 = evaluate_embedding(emb_train_tf, emb_test_tf,
                                        y_train_binary, y_test_binary, "Transformer")
    tf_mmd = compute_mmd_numpy(emb_train_tf[:2000], emb_test_tf[:2000])
    tf_params = sum(p.numel() for p in tf_model.parameters())
    print(f"  Transformer → Acc: {tf_acc:.4f}, F1: {tf_f1:.4f}, MMD: {tf_mmd:.6f}")
    print(f"  Parameters: {tf_params:,}, Time: {tf_time:.1f}s")
    results['transformer'] = {
        'accuracy': tf_acc, 'f1': tf_f1, 'mmd': tf_mmd,
        'params': tf_params, 'train_time': tf_time
    }

    # ---- With MMD alignment ----
    print("\n[4/5] Training with MMD domain adaptation (lambda=1.0)...")

    # CNN + MMD
    print("\n  --- CNN + MMD ---")
    t0 = time.time()
    cnn_mmd_model, cnn_mmd_hist = train_sequential_embedding(
        X_train, model_type='cnn', epochs=epochs, embedding_dim=embedding_dim,
        X_target=X_test, mmd_lambda=1.0
    )
    cnn_mmd_time = time.time() - t0
    emb_train_cnn_mmd = get_embeddings_sequential(cnn_mmd_model, X_train)
    emb_test_cnn_mmd = get_embeddings_sequential(cnn_mmd_model, X_test)
    cnn_mmd_acc, cnn_mmd_f1 = evaluate_embedding(
        emb_train_cnn_mmd, emb_test_cnn_mmd,
        y_train_binary, y_test_binary, "CNN+MMD"
    )
    cnn_mmd_val = compute_mmd_numpy(emb_train_cnn_mmd[:2000], emb_test_cnn_mmd[:2000])
    print(f"  CNN+MMD → Acc: {cnn_mmd_acc:.4f}, F1: {cnn_mmd_f1:.4f}, MMD: {cnn_mmd_val:.6f}")
    results['cnn_mmd'] = {
        'accuracy': cnn_mmd_acc, 'f1': cnn_mmd_f1, 'mmd': cnn_mmd_val,
        'params': cnn_params, 'train_time': cnn_mmd_time
    }

    # Transformer + MMD
    print("\n  --- Transformer + MMD ---")
    t0 = time.time()
    tf_mmd_model, tf_mmd_hist = train_sequential_embedding(
        X_train, model_type='transformer', epochs=epochs, embedding_dim=embedding_dim,
        X_target=X_test, mmd_lambda=1.0
    )
    tf_mmd_time = time.time() - t0
    emb_train_tf_mmd = get_embeddings_sequential(tf_mmd_model, X_train)
    emb_test_tf_mmd = get_embeddings_sequential(tf_mmd_model, X_test)
    tf_mmd_acc, tf_mmd_f1 = evaluate_embedding(
        emb_train_tf_mmd, emb_test_tf_mmd,
        y_train_binary, y_test_binary, "Transformer+MMD"
    )
    tf_mmd_val = compute_mmd_numpy(emb_train_tf_mmd[:2000], emb_test_tf_mmd[:2000])
    print(f"  Transformer+MMD → Acc: {tf_mmd_acc:.4f}, F1: {tf_mmd_f1:.4f}, MMD: {tf_mmd_val:.6f}")
    results['transformer_mmd'] = {
        'accuracy': tf_mmd_acc, 'f1': tf_mmd_f1, 'mmd': tf_mmd_val,
        'params': tf_params, 'train_time': tf_mmd_time
    }

    # ---- Summary ----
    print("\n[5/5] SUMMARY")
    print("=" * 70)
    print(f"{'Method':<20} {'Accuracy':>10} {'F1':>10} {'MMD':>12} {'Params':>10}")
    print("-" * 70)
    for name, r in results.items():
        print(f"{name:<20} {r['accuracy']:>10.4f} {r['f1']:>10.4f} "
              f"{r['mmd']:>12.6f} {r['params']:>10,}")
    print("=" * 70)

    # Key observations for the log
    print("\n  Observations:")
    best_method = max(results.items(), key=lambda x: x[1]['accuracy'])
    print(f"  Best accuracy: {best_method[0]} ({best_method[1]['accuracy']:.4f})")
    lowest_mmd = min(results.items(), key=lambda x: x[1]['mmd'])
    print(f"  Lowest MMD: {lowest_mmd[0]} ({lowest_mmd[1]['mmd']:.6f})")

    print("\n  Shift robustness: accuracy drop from raw features")
    print("  (lower drop = more robust to distribution shift):")
    for name, r in results.items():
        if name == 'raw_features':
            continue
        delta = r['accuracy'] - raw_acc
        direction = "+" if delta >= 0 else ""
        print(f"    {name}: {direction}{delta*100:.1f}pp vs raw features")

    # Save results
    os.makedirs('results', exist_ok=True)
    # Convert numpy types for JSON
    json_results = {}
    for k, v in results.items():
        json_results[k] = {kk: float(vv) if isinstance(vv, (np.floating, float)) else vv
                           for kk, vv in v.items()}
    with open('results/embedding_comparison.json', 'w') as f:
        json.dump(json_results, f, indent=2)
    print("\n  Results saved to results/embedding_comparison.json")


if __name__ == '__main__':
    main()
