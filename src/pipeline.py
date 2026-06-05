"""
Training and evaluation for the two-stage detection pipeline.

Stage 1: Binary detection (is this traffic normal or an attack?)
Stage 2: Attack classification (what kind of attack is it?)

Stage 2 only processes traffic that Stage 1 flagged, so Stage 1's
mistakes directly affect what Stage 2 sees.
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
import time


def get_models():
    """Return a dictionary of models to train."""
    return {
        'Random Forest': RandomForestClassifier(
            n_estimators=100, max_depth=20, n_jobs=-1, random_state=42
        ),
        'SVM (RBF)': SVC(
            kernel='rbf', C=10, gamma='scale', random_state=42,
            probability=True, max_iter=-1
        ),
        'MLP': MLPClassifier(
            hidden_layer_sizes=(128, 64), max_iter=200, random_state=42,
            early_stopping=True, validation_fraction=0.1
        ),
    }


def evaluate_model(y_true, y_pred, class_names=None):
    """Compute evaluation metrics."""
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
        'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0),
        'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
        'f1_weighted': f1_score(y_true, y_pred, average='weighted', zero_division=0),
        'confusion_matrix': confusion_matrix(y_true, y_pred),
    }
    return metrics


def train_stage1(X_train, y_train, X_test, y_test):
    """
    Stage 1: Binary anomaly detection (normal=0, attack=1).
    Trains multiple models and returns the best one.
    """
    print("\n[3/4] Training Stage 1: Binary Anomaly Detection")
    print("=" * 60)

    models = get_models()
    results = {}
    best_f1 = 0
    best_model = None
    best_name = None

    for name, model in models.items():
        print(f"\n  Training {name}...")
        start = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - start

        y_pred = model.predict(X_test)
        metrics = evaluate_model(y_test, y_pred)
        metrics['train_time'] = train_time

        results[name] = {'model': model, 'metrics': metrics, 'predictions': y_pred}

        print(f"    Accuracy:  {metrics['accuracy']:.4f}")
        print(f"    F1 (macro): {metrics['f1_macro']:.4f}")
        print(f"    F1 (weighted): {metrics['f1_weighted']:.4f}")
        print(f"    Precision: {metrics['precision_macro']:.4f}")
        print(f"    Recall:    {metrics['recall_macro']:.4f}")
        print(f"    Time:      {train_time:.1f}s")

        if metrics['f1_weighted'] > best_f1:
            best_f1 = metrics['f1_weighted']
            best_model = model
            best_name = name

    print(f"\n  >> Best Stage 1 model: {best_name} (F1={best_f1:.4f})")
    return results, best_model, best_name


def train_stage2(X_train, y_train, X_test, y_test, class_names):
    """
    Stage 2: Multi-class attack classification.
    Trains on all attack samples, evaluated on attack samples.
    """
    print("\n[4/4] Training Stage 2: Attack Classification")
    print("=" * 60)

    # Only use attack samples (not normal traffic)
    models = get_models()
    results = {}
    best_f1 = 0
    best_model = None
    best_name = None

    for name, model in models.items():
        print(f"\n  Training {name}...")
        start = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - start

        y_pred = model.predict(X_test)
        metrics = evaluate_model(y_test, y_pred, class_names)
        metrics['train_time'] = train_time

        results[name] = {'model': model, 'metrics': metrics, 'predictions': y_pred}

        print(f"    Accuracy:  {metrics['accuracy']:.4f}")
        print(f"    F1 (macro): {metrics['f1_macro']:.4f}")
        print(f"    F1 (weighted): {metrics['f1_weighted']:.4f}")
        print(f"    Time:      {train_time:.1f}s")
        print(f"\n    Per-class report:")
        report = classification_report(y_test, y_pred, target_names=class_names, zero_division=0)
        for line in report.split('\n'):
            if line.strip():
                print(f"      {line}")

        if metrics['f1_weighted'] > best_f1:
            best_f1 = metrics['f1_weighted']
            best_model = model
            best_name = name

    print(f"\n  >> Best Stage 2 model: {best_name} (F1={best_f1:.4f})")
    return results, best_model, best_name


def run_full_pipeline(stage1_model, stage2_model, X_test, y_test_binary,
                      y_test_multi, class_names):
    """
    Run both stages back-to-back on the test set.
    Stage 1 decides normal vs attack, then Stage 2 classifies the attack
    type -- but only on the traffic that Stage 1 actually flagged.
    So if Stage 1 misses something, Stage 2 never gets a chance to see it.
    """
    print("\n" + "=" * 60)
    print("FULL PIPELINE EVALUATION (Stage 1 -> Stage 2)")
    print("=" * 60)

    # Stage 1: Binary detection
    stage1_pred = stage1_model.predict(X_test)

    # Indices where Stage 1 says "attack"
    flagged_attack = stage1_pred == 1
    flagged_normal = stage1_pred == 0

    # Stage 1 performance
    s1_acc = accuracy_score(y_test_binary, stage1_pred)
    s1_tp = np.sum((stage1_pred == 1) & (y_test_binary == 1))  # True attacks caught
    s1_fn = np.sum((stage1_pred == 0) & (y_test_binary == 1))  # Attacks missed
    s1_fp = np.sum((stage1_pred == 1) & (y_test_binary == 0))  # Normal flagged as attack
    s1_tn = np.sum((stage1_pred == 0) & (y_test_binary == 0))  # Normal correctly passed

    print(f"\n  Stage 1 Results:")
    print(f"    Accuracy: {s1_acc:.4f}")
    print(f"    True Positives (attacks caught): {s1_tp}")
    print(f"    False Negatives (attacks missed): {s1_fn}")
    print(f"    False Positives (normal flagged): {s1_fp}")
    print(f"    True Negatives (normal passed):  {s1_tn}")
    print(f"    Detection Rate: {s1_tp / (s1_tp + s1_fn):.4f}")

    # Stage 2: Classify only flagged traffic
    if np.sum(flagged_attack) == 0:
        print("\n  Stage 1 flagged no traffic as attack — Stage 2 has no input.")
        return {}

    X_flagged = X_test[flagged_attack]
    y_true_multi_flagged = y_test_multi[flagged_attack]
    y_true_binary_flagged = y_test_binary[flagged_attack]

    stage2_pred = stage2_model.predict(X_flagged)

    # Of the flagged traffic, how many were actually attacks?
    actual_attacks_in_flagged = np.sum(y_true_binary_flagged == 1)
    false_positives_in_flagged = np.sum(y_true_binary_flagged == 0)

    print(f"\n  Stage 2 Input (traffic flagged by Stage 1):")
    print(f"    Total flagged: {np.sum(flagged_attack)}")
    print(f"    Actually attacks: {actual_attacks_in_flagged}")
    print(f"    False positives from Stage 1: {false_positives_in_flagged}")

    # Evaluate Stage 2 only on actual attack traffic that was correctly flagged
    actual_attack_mask = y_true_binary_flagged == 1
    if np.sum(actual_attack_mask) > 0:
        y_true_attacks = y_true_multi_flagged[actual_attack_mask]
        y_pred_attacks = stage2_pred[actual_attack_mask]

        # Filter to only attack classes (exclude 'normal' index)
        normal_idx = class_names.index('normal') if 'normal' in class_names else -1
        attack_class_names = [c for i, c in enumerate(class_names) if c != 'normal']

        s2_acc = accuracy_score(y_true_attacks, y_pred_attacks)
        s2_f1 = f1_score(y_true_attacks, y_pred_attacks, average='weighted', zero_division=0)

        print(f"\n  Stage 2 Results (on correctly flagged attacks):")
        print(f"    Accuracy: {s2_acc:.4f}")
        print(f"    F1 (weighted): {s2_f1:.4f}")

    # End-to-end metrics
    total_attacks = np.sum(y_test_binary == 1)
    attacks_detected = s1_tp
    attacks_correctly_classified = 0
    if np.sum(actual_attack_mask) > 0:
        attacks_correctly_classified = np.sum(y_true_attacks == y_pred_attacks)

    print(f"\n  End-to-End Pipeline Performance:")
    print(f"    Total attacks in test set:          {total_attacks}")
    print(f"    Attacks detected by Stage 1:        {attacks_detected} ({attacks_detected/total_attacks:.1%})")
    print(f"    Attacks correctly classified (E2E):  {attacks_correctly_classified} ({attacks_correctly_classified/total_attacks:.1%})")
    print(f"    Attacks lost at Stage 1:            {s1_fn} ({s1_fn/total_attacks:.1%})")

    return {
        'stage1_accuracy': s1_acc,
        'stage1_detection_rate': s1_tp / (s1_tp + s1_fn),
        'stage1_false_positive_rate': s1_fp / (s1_fp + s1_tn) if (s1_fp + s1_tn) > 0 else 0,
        'stage2_accuracy_on_flagged': s2_acc if np.sum(actual_attack_mask) > 0 else 0,
        'e2e_detection_rate': attacks_detected / total_attacks,
        'e2e_classification_rate': attacks_correctly_classified / total_attacks,
    }
