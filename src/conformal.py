"""
Conformal prediction for the IDS pipeline.

Provides distribution-free prediction sets with guaranteed coverage.
Instead of outputting a single class label, the model outputs a SET of
possible classes. The guarantee: the true class is in the set with
probability >= 1 - alpha, regardless of the data distribution.

This is strictly stronger than threshold-based confidence (which offers
no formal guarantee). The only assumption is exchangeability of data.

Algorithm (Split/Inductive Conformal Prediction):
    1. Split training data into proper training set and calibration set
    2. Train model on proper training set
    3. Compute nonconformity scores on calibration set:
       s_i = 1 - f(x_i)[y_i]  (1 minus model's prob for true class)
    4. Compute quantile threshold:
       q_hat = quantile(scores, ceil((n+1)(1-alpha)) / n)
    5. At test time, prediction set includes all classes k where:
       f(x_test)[k] >= 1 - q_hat

References:
    - Vovk, Gammerman, Shafer. "Algorithmic Learning in a Random World"
      (Springer, 2005). Foundational text on conformal prediction.
    - Angelopoulos & Bates. "A Gentle Introduction to Conformal
      Prediction and Distribution-Free Uncertainty Quantification"
      (arXiv:2107.07511, 2021). Modern tutorial.
    - Romano, Sesia, Candes. "Classification with Valid and Adaptive
      Coverage" (NeurIPS 2020). Adaptive conformal methods.

Author: Uvesh Patel
"""

import numpy as np
from sklearn.model_selection import train_test_split


def compute_nonconformity_scores(model, X_cal, y_cal):
    """
    Compute nonconformity scores on a calibration set.

    Score for sample i = 1 - model's predicted probability for the TRUE class.
    Higher score = the model was LESS confident about the correct answer.

    Args:
        model: trained sklearn model with predict_proba
        X_cal: calibration features
        y_cal: calibration true labels

    Returns:
        scores: array of nonconformity scores, one per calibration sample
    """
    proba = model.predict_proba(X_cal)
    # Clip to avoid degenerate scores when RF gives exact 1.0 or 0.0
    # (tree ensembles can give p=1.0 via unanimous voting)
    proba = np.clip(proba, 1e-4, 1 - 1e-4)
    # Renormalize after clipping
    proba = proba / proba.sum(axis=1, keepdims=True)

    # Get the probability assigned to the true class for each sample
    # model.classes_ maps index to class label
    class_to_idx = {c: i for i, c in enumerate(model.classes_)}
    true_class_probs = np.array([
        proba[i, class_to_idx[y_cal[i]]] for i in range(len(y_cal))
    ])
    scores = 1.0 - true_class_probs
    return scores


def compute_quantile_threshold(scores, alpha=0.1):
    """
    Compute the conformal quantile threshold.

    q_hat = the ceil((n+1)(1-alpha))/n quantile of the calibration scores.
    This ensures the marginal coverage guarantee.

    Args:
        scores: nonconformity scores from calibration set
        alpha: desired miscoverage rate (e.g., 0.1 for 90% coverage)

    Returns:
        q_hat: the quantile threshold
    """
    n = len(scores)
    # Finite-sample correction: use ceil((n+1)(1-alpha))/n quantile
    quantile_level = min(np.ceil((n + 1) * (1 - alpha)) / n, 1.0)
    q_hat = np.quantile(scores, quantile_level)
    return q_hat


def predict_sets(model, X_test, q_hat):
    """
    Generate prediction sets for test data.

    A class k is included in the prediction set if:
        model_probability(k) >= 1 - q_hat

    Equivalently, if the nonconformity score for that class would be <= q_hat.

    Args:
        model: trained model
        X_test: test features
        q_hat: conformal threshold from calibration

    Returns:
        prediction_sets: list of sets, one per test sample
    """
    proba = model.predict_proba(X_test)
    proba = np.clip(proba, 1e-4, 1 - 1e-4)
    proba = proba / proba.sum(axis=1, keepdims=True)
    classes = model.classes_
    threshold = 1.0 - q_hat

    prediction_sets = []
    for i in range(len(X_test)):
        pset = set()
        for j, cls in enumerate(classes):
            if proba[i, j] >= threshold:
                pset.add(cls)
        # If empty (shouldn't happen with proper alpha), include argmax
        if len(pset) == 0:
            pset.add(classes[np.argmax(proba[i])])
        prediction_sets.append(pset)

    return prediction_sets


def evaluate_conformal(prediction_sets, y_true, alpha):
    """
    Evaluate conformal prediction performance.

    Metrics:
        - Coverage: fraction of samples where true class is in the set
          (should be >= 1 - alpha by guarantee)
        - Average set size: mean number of classes in prediction sets
          (smaller = more informative)
        - Singleton rate: fraction of sets with exactly 1 class
          (higher = more decisive predictions)
        - Empty rate: fraction of empty sets (should be 0)

    Args:
        prediction_sets: list of prediction sets
        y_true: true labels
        alpha: target miscoverage rate

    Returns:
        dict with evaluation metrics
    """
    n = len(y_true)

    covered = sum(1 for i in range(n) if y_true[i] in prediction_sets[i])
    coverage = covered / n

    set_sizes = [len(s) for s in prediction_sets]
    avg_size = np.mean(set_sizes)

    singletons = sum(1 for s in set_sizes if s == 1)
    singleton_rate = singletons / n

    empty = sum(1 for s in set_sizes if s == 0)
    empty_rate = empty / n

    return {
        'alpha': float(alpha),
        'target_coverage': float(1 - alpha),
        'empirical_coverage': float(coverage),
        'coverage_valid': coverage >= (1 - alpha),
        'avg_set_size': float(avg_size),
        'singleton_rate': float(singleton_rate),
        'empty_rate': float(empty_rate),
        'n_samples': n,
    }


def run_conformal_pipeline(model, X_train, y_train, X_test, y_test,
                           alpha_values=None, cal_fraction=0.3, random_state=42):
    """
    Full conformal prediction pipeline — two modes compared:

    Mode A (source-calibrated): calibrate on training data cross-validation.
        This WILL fail under distribution shift (important finding).

    Mode B (target-calibrated): calibrate on a held-out portion of test data.
        This simulates having some labeled target data (realistic in
        deployment: you'd label some examples from the target network).
        Should satisfy the coverage guarantee.

    Comparing both modes demonstrates WHY domain adaptation matters:
    without it, even statistical guarantees break.

    Args:
        model: untrained sklearn classifier (will be cloned internally)
        X_train: full training features
        y_train: full training labels
        X_test: test features
        y_test: test labels
        alpha_values: list of miscoverage rates to evaluate
        cal_fraction: fraction for calibration splits
        random_state: for reproducibility

    Returns:
        results: dict with source_calibrated and target_calibrated results
    """
    from sklearn.model_selection import StratifiedKFold
    from sklearn.base import clone

    if alpha_values is None:
        alpha_values = [0.01, 0.05, 0.10, 0.15, 0.20]

    # === MODE A: Source-calibrated (cross-conformal on training data) ===
    n_folds = 5
    kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)

    all_scores_source = []
    for fold_idx, (train_idx, cal_idx) in enumerate(kf.split(X_train, y_train)):
        fold_model = clone(model)
        fold_model.fit(X_train[train_idx], y_train[train_idx])
        fold_scores = compute_nonconformity_scores(
            fold_model, X_train[cal_idx], y_train[cal_idx]
        )
        all_scores_source.append(fold_scores)

    scores_source = np.concatenate(all_scores_source)

    # Train final model on all training data
    model.fit(X_train, y_train)

    # === MODE B: Target-calibrated (split test into cal + eval) ===
    # Use 30% of test data for calibration, evaluate on remaining 70%
    n_test = len(X_test)
    rng = np.random.RandomState(random_state)
    perm = rng.permutation(n_test)
    n_cal = int(n_test * cal_fraction)

    cal_idx = perm[:n_cal]
    eval_idx = perm[n_cal:]

    X_test_cal = X_test[cal_idx]
    y_test_cal = y_test[cal_idx]
    X_test_eval = X_test[eval_idx]
    y_test_eval = y_test[eval_idx]

    scores_target = compute_nonconformity_scores(model, X_test_cal, y_test_cal)

    # Evaluate both modes
    results = {
        'source_calibrated': {},
        'target_calibrated': {},
        'score_stats': {
            'source_mean': float(scores_source.mean()),
            'source_std': float(scores_source.std()),
            'source_max': float(scores_source.max()),
            'target_mean': float(scores_target.mean()),
            'target_std': float(scores_target.std()),
            'target_max': float(scores_target.max()),
        }
    }

    for alpha in alpha_values:
        # Mode A: source-calibrated, evaluated on full test set
        q_hat_source = compute_quantile_threshold(scores_source, alpha)
        pred_sets_source = predict_sets(model, X_test, q_hat_source)
        metrics_source = evaluate_conformal(pred_sets_source, y_test, alpha)
        metrics_source['q_hat'] = float(q_hat_source)
        results['source_calibrated'][f'alpha_{alpha}'] = metrics_source

        # Mode B: target-calibrated, evaluated on held-out test portion
        q_hat_target = compute_quantile_threshold(scores_target, alpha)
        pred_sets_target = predict_sets(model, X_test_eval, q_hat_target)
        metrics_target = evaluate_conformal(pred_sets_target, y_test_eval, alpha)
        metrics_target['q_hat'] = float(q_hat_target)
        results['target_calibrated'][f'alpha_{alpha}'] = metrics_target

    return results
