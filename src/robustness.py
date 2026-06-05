"""
Robustness testing for the pipeline.

Tests what happens when:
1. Features get noisy (gaussian noise at varying levels)
2. Features go missing (random dropout, like a sensor failing)
3. Stage 1 errors bleed into Stage 2

Generates plots showing how each stage and the full pipeline degrade.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
)
import os


def inject_gaussian_noise(X, noise_level):
    """Add Gaussian noise to features. noise_level = std dev relative to feature range."""
    noise = np.random.normal(0, noise_level, X.shape)
    X_noisy = X + noise
    return np.clip(X_noisy, 0, 1)  # Features are MinMax scaled to [0,1]


def inject_feature_dropout(X, dropout_rate):
    """Randomly set features to 0 (simulating missing data)."""
    mask = np.random.binomial(1, 1 - dropout_rate, X.shape)
    return X * mask


def analyze_noise_robustness(stage1_model, stage2_model, X_test,
                              y_test_binary, y_test_multi, class_names,
                              results_dir="results"):
    """
    Analyze how pipeline performance degrades with increasing feature noise.
    """
    print("\n" + "=" * 60)
    print("ROBUSTNESS ANALYSIS: Feature Noise")
    print("=" * 60)

    noise_levels = [0.0, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]

    s1_accuracies = []
    s1_f1_scores = []
    s1_detection_rates = []
    s2_accuracies = []
    e2e_rates = []

    for noise in noise_levels:
        # Inject noise
        X_noisy = inject_gaussian_noise(X_test, noise) if noise > 0 else X_test.copy()

        # Stage 1
        s1_pred = stage1_model.predict(X_noisy)
        s1_acc = accuracy_score(y_test_binary, s1_pred)
        s1_f1 = f1_score(y_test_binary, s1_pred, average='weighted', zero_division=0)
        tp = np.sum((s1_pred == 1) & (y_test_binary == 1))
        fn = np.sum((s1_pred == 0) & (y_test_binary == 1))
        det_rate = tp / (tp + fn) if (tp + fn) > 0 else 0

        # Stage 2 on flagged traffic
        flagged = s1_pred == 1
        s2_acc = 0.0
        e2e_rate = 0.0
        if np.sum(flagged) > 0:
            X_flagged = X_noisy[flagged]
            y_true_multi_flagged = y_test_multi[flagged]
            y_true_binary_flagged = y_test_binary[flagged]
            s2_pred = stage2_model.predict(X_flagged)

            actual_attacks = y_true_binary_flagged == 1
            if np.sum(actual_attacks) > 0:
                s2_acc = accuracy_score(
                    y_true_multi_flagged[actual_attacks],
                    s2_pred[actual_attacks]
                )
                correct = np.sum(
                    y_true_multi_flagged[actual_attacks] == s2_pred[actual_attacks]
                )
                total_attacks = np.sum(y_test_binary == 1)
                e2e_rate = correct / total_attacks if total_attacks > 0 else 0

        s1_accuracies.append(s1_acc)
        s1_f1_scores.append(s1_f1)
        s1_detection_rates.append(det_rate)
        s2_accuracies.append(s2_acc)
        e2e_rates.append(e2e_rate)

        print(f"  Noise={noise:.2f}: S1_Acc={s1_acc:.4f}, S1_Det={det_rate:.4f}, "
              f"S2_Acc={s2_acc:.4f}, E2E={e2e_rate:.4f}")

    # Plot results
    os.makedirs(results_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].plot(noise_levels, s1_accuracies, 'b-o', label='Stage 1 Accuracy')
    axes[0].plot(noise_levels, s1_detection_rates, 'r-s', label='Stage 1 Detection Rate')
    axes[0].set_xlabel('Noise Level (σ)')
    axes[0].set_ylabel('Score')
    axes[0].set_title('Stage 1: Binary Detection vs Noise')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim([0, 1.05])

    axes[1].plot(noise_levels, s2_accuracies, 'g-^', label='Stage 2 Accuracy')
    axes[1].set_xlabel('Noise Level (σ)')
    axes[1].set_ylabel('Score')
    axes[1].set_title('Stage 2: Attack Classification vs Noise')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim([0, 1.05])

    axes[2].plot(noise_levels, e2e_rates, 'm-D', label='End-to-End Rate', linewidth=2)
    axes[2].plot(noise_levels, s1_detection_rates, 'r--', label='Stage 1 Only', alpha=0.5)
    axes[2].plot(noise_levels, s2_accuracies, 'g--', label='Stage 2 Only', alpha=0.5)
    axes[2].set_xlabel('Noise Level (σ)')
    axes[2].set_ylabel('Score')
    axes[2].set_title('Pipeline Degradation: Cascading Effect')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    axes[2].set_ylim([0, 1.05])

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'noise_robustness.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Saved: {results_dir}/noise_robustness.png")

    return {
        'noise_levels': noise_levels,
        's1_accuracies': s1_accuracies,
        's1_detection_rates': s1_detection_rates,
        's2_accuracies': s2_accuracies,
        'e2e_rates': e2e_rates,
    }


def analyze_feature_dropout(stage1_model, stage2_model, X_test,
                             y_test_binary, y_test_multi, class_names,
                             results_dir="results"):
    """
    Analyze how pipeline performance degrades with increasing feature dropout.
    """
    print("\n" + "=" * 60)
    print("ROBUSTNESS ANALYSIS: Feature Dropout")
    print("=" * 60)

    dropout_rates = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]

    s1_accuracies = []
    s2_accuracies = []
    e2e_rates = []

    for rate in dropout_rates:
        X_dropped = inject_feature_dropout(X_test, rate) if rate > 0 else X_test.copy()

        # Stage 1
        s1_pred = stage1_model.predict(X_dropped)
        s1_acc = accuracy_score(y_test_binary, s1_pred)
        tp = np.sum((s1_pred == 1) & (y_test_binary == 1))
        fn = np.sum((s1_pred == 0) & (y_test_binary == 1))

        # Stage 2
        flagged = s1_pred == 1
        s2_acc = 0.0
        e2e_rate = 0.0
        if np.sum(flagged) > 0:
            s2_pred = stage2_model.predict(X_dropped[flagged])
            actual_attacks = y_test_binary[flagged] == 1
            if np.sum(actual_attacks) > 0:
                s2_acc = accuracy_score(
                    y_test_multi[flagged][actual_attacks],
                    s2_pred[actual_attacks]
                )
                correct = np.sum(
                    y_test_multi[flagged][actual_attacks] == s2_pred[actual_attacks]
                )
                total_attacks = np.sum(y_test_binary == 1)
                e2e_rate = correct / total_attacks if total_attacks > 0 else 0

        s1_accuracies.append(s1_acc)
        s2_accuracies.append(s2_acc)
        e2e_rates.append(e2e_rate)

        print(f"  Dropout={rate:.2f}: S1_Acc={s1_acc:.4f}, S2_Acc={s2_acc:.4f}, E2E={e2e_rate:.4f}")

    # Plot
    os.makedirs(results_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(dropout_rates, s1_accuracies, 'b-o', label='Stage 1 Accuracy')
    ax.plot(dropout_rates, s2_accuracies, 'g-^', label='Stage 2 Accuracy')
    ax.plot(dropout_rates, e2e_rates, 'm-D', label='End-to-End Rate', linewidth=2)
    ax.set_xlabel('Feature Dropout Rate')
    ax.set_ylabel('Score')
    ax.set_title('Pipeline Degradation Under Feature Dropout')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.05])
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'dropout_robustness.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Saved: {results_dir}/dropout_robustness.png")

    return {
        'dropout_rates': dropout_rates,
        's1_accuracies': s1_accuracies,
        's2_accuracies': s2_accuracies,
        'e2e_rates': e2e_rates,
    }


def plot_confusion_matrices(stage1_results, stage2_results, class_names,
                             results_dir="results"):
    """Plot confusion matrices for the best models at each stage."""
    os.makedirs(results_dir, exist_ok=True)

    # Stage 1 confusion matrix (best model)
    best_s1_name = max(stage1_results, key=lambda k: stage1_results[k]['metrics']['f1_weighted'])
    cm1 = stage1_results[best_s1_name]['metrics']['confusion_matrix']

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    sns.heatmap(cm1, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Normal', 'Attack'], yticklabels=['Normal', 'Attack'],
                ax=axes[0])
    axes[0].set_title(f'Stage 1: {best_s1_name}\nBinary Detection')
    axes[0].set_ylabel('True Label')
    axes[0].set_xlabel('Predicted Label')

    # Stage 2 confusion matrix (best model)
    best_s2_name = max(stage2_results, key=lambda k: stage2_results[k]['metrics']['f1_weighted'])
    cm2 = stage2_results[best_s2_name]['metrics']['confusion_matrix']

    sns.heatmap(cm2, annot=True, fmt='d', cmap='Oranges',
                xticklabels=class_names, yticklabels=class_names,
                ax=axes[1])
    axes[1].set_title(f'Stage 2: {best_s2_name}\nAttack Classification')
    axes[1].set_ylabel('True Label')
    axes[1].set_xlabel('Predicted Label')

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'confusion_matrices.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Saved: {results_dir}/confusion_matrices.png")


def plot_model_comparison(stage1_results, stage2_results, results_dir="results"):
    """Bar chart comparing all models at each stage."""
    os.makedirs(results_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Stage 1
    names = list(stage1_results.keys())
    metrics_list = ['accuracy', 'precision_macro', 'recall_macro', 'f1_weighted']
    x = np.arange(len(names))
    width = 0.2

    for i, metric in enumerate(metrics_list):
        values = [stage1_results[n]['metrics'][metric] for n in names]
        axes[0].bar(x + i * width, values, width, label=metric.replace('_', ' ').title())

    axes[0].set_xticks(x + width * 1.5)
    axes[0].set_xticklabels(names, rotation=15)
    axes[0].set_ylabel('Score')
    axes[0].set_title('Stage 1: Model Comparison')
    axes[0].legend(fontsize=8)
    axes[0].set_ylim([0, 1.05])
    axes[0].grid(True, alpha=0.3, axis='y')

    # Stage 2
    names2 = list(stage2_results.keys())
    for i, metric in enumerate(metrics_list):
        values = [stage2_results[n]['metrics'][metric] for n in names2]
        axes[1].bar(x + i * width, values, width, label=metric.replace('_', ' ').title())

    axes[1].set_xticks(x + width * 1.5)
    axes[1].set_xticklabels(names2, rotation=15)
    axes[1].set_ylabel('Score')
    axes[1].set_title('Stage 2: Model Comparison')
    axes[1].legend(fontsize=8)
    axes[1].set_ylim([0, 1.05])
    axes[1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'model_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {results_dir}/model_comparison.png")


def feature_importance_analysis(stage1_model, stage2_model, feature_cols,
                                 class_names, results_dir="results"):
    """Plot feature importance from Random Forest models."""
    os.makedirs(results_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for ax, model, title in [
        (axes[0], stage1_model, 'Stage 1: Binary Detection'),
        (axes[1], stage2_model, 'Stage 2: Attack Classification')
    ]:
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            indices = np.argsort(importances)[-15:]  # Top 15 features

            ax.barh(range(len(indices)), importances[indices], color='steelblue')
            ax.set_yticks(range(len(indices)))
            ax.set_yticklabels([feature_cols[i] for i in indices], fontsize=8)
            ax.set_xlabel('Importance')
            ax.set_title(f'{title}\nTop 15 Features')
            ax.grid(True, alpha=0.3, axis='x')
        else:
            ax.text(0.5, 0.5, 'Feature importance\nnot available\nfor this model type',
                    ha='center', va='center', transform=ax.transAxes)
            ax.set_title(title)

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'feature_importance.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {results_dir}/feature_importance.png")
