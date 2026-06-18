"""
Model explainability using SHAP (SHapley Additive exPlanations).

For each prediction, SHAP computes the contribution of every feature
to that specific decision. This allows us to answer: "Why did the model
flag this connection as an attack?" rather than just "the model says attack."

For tree-based models (RF), we use TreeExplainer which computes exact
SHAP values in polynomial time. For neural networks, we use KernelSHAP.

Key concepts:
    - SHAP value phi_j(x): the contribution of feature j to the prediction
      for input x, relative to the average prediction.
    - Sum of all SHAP values + base value = model's prediction
    - Global importance: mean(|phi_j|) across all samples

References:
    - Lundberg & Lee. "A Unified Approach to Interpreting Model
      Predictions" (NeurIPS 2017). Original SHAP paper.
    - Lundberg et al. "From local explanations to global understanding
      with explainable AI for trees" (Nature Machine Intelligence, 2020).
      TreeExplainer with polynomial-time exactness.

Author: Uvesh Patel
"""

import numpy as np


def compute_shap_values(model, X, feature_names=None, max_samples=500):
    """
    Compute SHAP values for a tree-based model.

    Uses TreeExplainer for exact computation. Limits samples to avoid
    excessive computation time on large datasets.

    Args:
        model: trained sklearn tree-based model (RF, GBT)
        X: feature matrix (will be subsampled if > max_samples)
        feature_names: list of feature names
        max_samples: max samples to explain (for speed)

    Returns:
        dict with:
            shap_values: array of shape (n_samples, n_features) or
                         (n_samples, n_features, n_classes) for multi-class
            base_value: expected model output
            feature_names: feature names used
            X_sample: the actual samples explained
    """
    import shap

    # Subsample if needed
    if len(X) > max_samples:
        rng = np.random.RandomState(42)
        idx = rng.choice(len(X), max_samples, replace=False)
        X_sample = X[idx]
    else:
        X_sample = X

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    return {
        'shap_values': shap_values,
        'base_value': explainer.expected_value,
        'feature_names': feature_names,
        'X_sample': X_sample,
    }


def global_feature_importance(shap_result, top_k=15):
    """
    Compute global feature importance from SHAP values.

    Global importance = mean(|SHAP value|) for each feature across all samples.
    This shows which features the model relies on most overall.

    Args:
        shap_result: output from compute_shap_values
        top_k: number of top features to return

    Returns:
        list of (feature_name, importance) tuples, sorted descending
    """
    shap_values = shap_result['shap_values']
    feature_names = shap_result['feature_names']

    # Handle different SHAP output formats:
    # - Old format: list of 2D arrays (one per class)
    # - New format: 3D numpy array (samples x features x classes)
    # - Binary: 2D array (samples x features)
    if isinstance(shap_values, list):
        # Old multi-class format: list of (n_samples, n_features) arrays
        abs_vals = np.mean([np.abs(sv) for sv in shap_values], axis=0)
    elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        # New format: (n_samples, n_features, n_classes)
        abs_vals = np.mean(np.abs(shap_values), axis=2)
    else:
        # Binary 2D: (n_samples, n_features)
        abs_vals = np.abs(shap_values)

    # Mean across samples to get per-feature importance (1D)
    mean_importance = np.mean(abs_vals, axis=0)

    # Pair with feature names
    n_features = len(mean_importance)
    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(n_features)]

    importance_pairs = [(feature_names[i], float(mean_importance[i]))
                        for i in range(n_features)]
    importance_pairs.sort(key=lambda x: x[1], reverse=True)

    return importance_pairs[:top_k]


def explain_misclassified(model, X_test, y_test, y_pred, feature_names=None,
                          max_samples=50):
    """
    Explain why the model got specific samples wrong.

    For misclassified samples, compute SHAP values to understand which
    features led the model astray. Useful for diagnosing systematic errors.

    Args:
        model: trained model
        X_test: test features
        y_test: true labels
        y_pred: predicted labels
        feature_names: feature names
        max_samples: max misclassified samples to explain

    Returns:
        dict with SHAP analysis of misclassified samples
    """
    import shap

    # Find misclassified indices
    misclassified_mask = y_test != y_pred
    misclassified_idx = np.where(misclassified_mask)[0]

    if len(misclassified_idx) == 0:
        return {'n_misclassified': 0, 'message': 'No misclassified samples'}

    # Subsample if too many
    if len(misclassified_idx) > max_samples:
        rng = np.random.RandomState(42)
        misclassified_idx = rng.choice(misclassified_idx, max_samples, replace=False)

    X_mis = X_test[misclassified_idx]

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_mis)

    # Find which features most often contribute to errors
    if isinstance(shap_values, list):
        abs_vals = np.mean([np.abs(sv) for sv in shap_values], axis=0)
    elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        abs_vals = np.mean(np.abs(shap_values), axis=2)
    else:
        abs_vals = np.abs(shap_values)

    mean_importance_errors = np.mean(abs_vals, axis=0)

    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(len(mean_importance_errors))]

    error_drivers = [(feature_names[i], float(mean_importance_errors[i]))
                     for i in range(len(mean_importance_errors))]
    error_drivers.sort(key=lambda x: x[1], reverse=True)

    return {
        'n_misclassified': int(misclassified_mask.sum()),
        'n_explained': len(misclassified_idx),
        'top_error_drivers': error_drivers[:10],
        'true_labels': y_test[misclassified_idx].tolist(),
        'predicted_labels': y_pred[misclassified_idx].tolist(),
    }
