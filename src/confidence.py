"""
Confidence estimation and selective prediction for the pipeline.

In a deployed IDS, blindly trusting every prediction is risky.
This module extracts prediction confidence from models, computes
accuracy at various coverage levels, and lets the pipeline abstain
on uncertain samples instead of guessing.

Works with any sklearn model that supports predict_proba, or with
the DAN/DANN models via softmax outputs.
"""

import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import accuracy_score, f1_score


def get_confidence(model, X):
    """
    Extract prediction confidence from a model.
    Returns (predictions, max_probability_per_sample).
    """
    proba = model.predict_proba(X)
    predictions = np.argmax(proba, axis=1)
    confidence = np.max(proba, axis=1)
    return predictions, confidence


def get_confidence_neural(feat_ext, classifier, X):
    """
    Extract confidence from a DAN/DANN model using softmax.
    """
    import torch
    import torch.nn.functional as F

    device = next(feat_ext.parameters()).device
    X_tensor = torch.FloatTensor(X).to(device)
    with torch.no_grad():
        features = feat_ext(X_tensor)
        logits = classifier(features)
        proba = F.softmax(logits, dim=1)
        confidence, predictions = torch.max(proba, dim=1)
    return predictions.cpu().numpy(), confidence.cpu().numpy()


def accuracy_at_coverage(y_true, y_pred, confidence, coverages=None):
    """
    Compute accuracy when only classifying the most confident samples.

    Coverage = fraction of samples the model dares to classify.
    At coverage=1.0, all samples are classified (normal accuracy).
    At coverage=0.5, only the top 50% most confident are classified.

    Returns dict with coverage levels and corresponding accuracies.
    """
    if coverages is None:
        coverages = np.arange(0.1, 1.05, 0.05)

    n = len(y_true)
    sorted_indices = np.argsort(-confidence)  # highest confidence first

    results = []
    for cov in coverages:
        k = max(1, int(cov * n))
        idx = sorted_indices[:k]
        acc = accuracy_score(y_true[idx], y_pred[idx])
        f1 = f1_score(y_true[idx], y_pred[idx], average='weighted', zero_division=0)
        results.append({
            'coverage': float(cov),
            'n_samples': k,
            'accuracy': float(acc),
            'f1': float(f1),
        })

    return results


def confidence_gated_pipeline(stage1_model, stage2_model, X, y_binary, y_multi,
                               threshold=0.8):
    """
    Run the two-stage pipeline with confidence gating.

    Stage 1 classifies all samples. Only samples where Stage 1 confidence
    exceeds the threshold AND prediction is 'attack' proceed to Stage 2.
    Remaining uncertain samples are flagged for human review.

    Returns metrics and counts for each category.
    """
    preds_s1, conf_s1 = get_confidence(stage1_model, X)

    # Split into confident and uncertain at Stage 1
    confident_mask = conf_s1 >= threshold
    attack_mask = preds_s1 == 1

    # Samples that proceed to Stage 2: confident AND predicted attack
    proceed_to_s2 = confident_mask & attack_mask
    # Samples flagged for review: uncertain
    flagged = ~confident_mask

    # Stage 1 metrics on confident samples only
    if confident_mask.sum() > 0:
        s1_acc_confident = accuracy_score(y_binary[confident_mask], preds_s1[confident_mask])
    else:
        s1_acc_confident = 0.0

    s1_acc_all = accuracy_score(y_binary, preds_s1)

    # Stage 2 on samples that passed through
    s2_acc = None
    if proceed_to_s2.sum() > 0:
        X_s2 = X[proceed_to_s2]
        y_s2_true = y_multi[proceed_to_s2]
        preds_s2, conf_s2 = get_confidence(stage2_model, X_s2)
        s2_acc = accuracy_score(y_s2_true, preds_s2)

    return {
        'threshold': threshold,
        'total_samples': len(X),
        'confident_count': int(confident_mask.sum()),
        'flagged_for_review': int(flagged.sum()),
        'flagged_percent': float(flagged.sum() / len(X) * 100),
        'proceeded_to_s2': int(proceed_to_s2.sum()),
        's1_accuracy_all': float(s1_acc_all),
        's1_accuracy_confident_only': float(s1_acc_confident),
        's2_accuracy_on_passed': float(s2_acc) if s2_acc is not None else None,
    }


def calibration_analysis(y_true, confidence, predictions, n_bins=10):
    """
    Check if model confidence scores are well-calibrated.

    A well-calibrated model: when it says "90% confident", it should
    be correct 90% of the time. Returns bin-level calibration data.
    """
    correct = (predictions == y_true).astype(int)

    # Bin samples by confidence level
    bins = np.linspace(0, 1, n_bins + 1)
    bin_data = []

    for i in range(n_bins):
        mask = (confidence >= bins[i]) & (confidence < bins[i+1])
        if mask.sum() == 0:
            continue
        avg_confidence = confidence[mask].mean()
        avg_accuracy = correct[mask].mean()
        bin_data.append({
            'bin_lower': float(bins[i]),
            'bin_upper': float(bins[i+1]),
            'avg_confidence': float(avg_confidence),
            'avg_accuracy': float(avg_accuracy),
            'n_samples': int(mask.sum()),
            'calibration_error': float(abs(avg_confidence - avg_accuracy)),
        })

    # Expected Calibration Error (ECE)
    total = sum(b['n_samples'] for b in bin_data)
    ece = sum(b['calibration_error'] * b['n_samples'] / total for b in bin_data)

    return {
        'bins': bin_data,
        'ece': float(ece),
    }


def sweep_thresholds(stage1_model, X, y_binary, thresholds=None):
    """
    Sweep over confidence thresholds to find the best operating point.
    Returns tradeoff data: threshold vs accuracy vs flagged percentage.
    """
    if thresholds is None:
        thresholds = np.arange(0.5, 0.99, 0.05)

    preds, confidence = get_confidence(stage1_model, X)
    results = []

    for t in thresholds:
        confident_mask = confidence >= t
        if confident_mask.sum() == 0:
            continue
        acc = accuracy_score(y_binary[confident_mask], preds[confident_mask])
        flagged_pct = (1 - confident_mask.mean()) * 100
        results.append({
            'threshold': float(t),
            'accuracy': float(acc),
            'flagged_percent': float(flagged_pct),
            'n_classified': int(confident_mask.sum()),
        })

    return results
