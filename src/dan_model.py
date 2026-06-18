"""
Deep Adaptation Network (DAN) for intrusion detection.

Based on Long et al. (2015) "Learning Transferable Features with Deep
Adaptation Networks." The key difference from our earlier approach: the
feature extractor, classifier, and MMD alignment are trained jointly in
a single end-to-end network.

The feature extractor learns representations that are:
1. Discriminative (good at classification, driven by cross-entropy loss)
2. Domain-invariant (similar across source/target, driven by MMD loss)

Also includes a DANN variant using gradient reversal (Ganin et al. 2016).
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.autograd import Function


# ---- Gradient Reversal Layer for DANN ----

class GradientReversalFunction(Function):
    """Flips the gradient sign during backpropagation."""
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.alpha * grad_output, None


class GradientReversalLayer(nn.Module):
    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.alpha)


# ---- Network Architecture ----

class FeatureExtractor(nn.Module):
    """Shared encoder that produces domain-invariant features."""
    def __init__(self, input_dim=41, hidden_dim=64, feature_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, feature_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


class TaskClassifier(nn.Module):
    """Predicts class labels from extracted features."""
    def __init__(self, feature_dim=32, num_classes=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, 16),
            nn.ReLU(),
            nn.Linear(16, num_classes),
        )

    def forward(self, x):
        return self.net(x)


class DomainClassifier(nn.Module):
    """For DANN: predicts whether input is from source or target domain."""
    def __init__(self, feature_dim=32):
        super().__init__()
        self.grl = GradientReversalLayer()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 2),
        )

    def forward(self, x, alpha=1.0):
        self.grl.alpha = alpha
        x = self.grl(x)
        return self.net(x)


# ---- MMD computation ----

def compute_mmd(source, target, bandwidths=[0.1, 0.5, 1.0, 2.0, 5.0]):
    """Multi-scale MMD between source and target feature tensors."""
    total = 0
    for bw in bandwidths:
        ss = _rbf(source, source, bw).mean()
        tt = _rbf(target, target, bw).mean()
        st = _rbf(source, target, bw).mean()
        total += ss + tt - 2 * st
    return total / len(bandwidths)


def _rbf(x, y, bw):
    dist = ((x.unsqueeze(1) - y.unsqueeze(0)) ** 2).sum(dim=2)
    return torch.exp(-dist / (2 * bw ** 2))


# ---- DAN Training ----

def train_dan(X_source, y_source, X_target, num_classes=2,
              epochs=80, batch_size=256, lr=1e-3, mmd_weight=0.5):
    """
    Train DAN: feature extractor + classifier + MMD.
    Classification loss on source (labeled), MMD on both (unlabeled target).
    """
    print(f"\n  Training DAN (end-to-end, lambda={mmd_weight})...")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_dim = X_source.shape[1]

    feat_ext = FeatureExtractor(input_dim=input_dim).to(device)
    classifier = TaskClassifier(num_classes=num_classes).to(device)

    params = list(feat_ext.parameters()) + list(classifier.parameters())
    optimizer = optim.Adam(params, lr=lr)
    cls_criterion = nn.CrossEntropyLoss()

    source_ds = TensorDataset(
        torch.FloatTensor(X_source),
        torch.LongTensor(y_source)
    )
    target_ds = TensorDataset(torch.FloatTensor(X_target))

    source_loader = DataLoader(source_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    target_loader = DataLoader(target_ds, batch_size=batch_size, shuffle=True, drop_last=True)

    for epoch in range(epochs):
        feat_ext.train()
        classifier.train()
        total_cls_loss = 0
        total_mmd_loss = 0
        n_batches = 0

        target_iter = iter(target_loader)
        for source_batch, source_labels in source_loader:
            try:
                (target_batch,) = next(target_iter)
            except StopIteration:
                target_iter = iter(target_loader)
                (target_batch,) = next(target_iter)

            source_batch = source_batch.to(device)
            source_labels = source_labels.to(device)
            target_batch = target_batch.to(device)

            # Forward
            source_features = feat_ext(source_batch)
            target_features = feat_ext(target_batch)
            source_preds = classifier(source_features)

            # Losses
            cls_loss = cls_criterion(source_preds, source_labels)
            mmd_loss = compute_mmd(source_features, target_features)
            loss = cls_loss + mmd_weight * mmd_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_cls_loss += cls_loss.item()
            total_mmd_loss += mmd_loss.item()
            n_batches += 1

        if (epoch + 1) % 20 == 0:
            avg_cls = total_cls_loss / n_batches
            avg_mmd = total_mmd_loss / n_batches
            print(f"    Epoch {epoch+1}/{epochs} — cls: {avg_cls:.4f}, MMD: {avg_mmd:.6f}")

    feat_ext.eval()
    classifier.eval()
    print(f"  DAN trained.")
    return feat_ext, classifier


# ---- DANN Training ----

def train_dann(X_source, y_source, X_target, num_classes=2,
               epochs=80, batch_size=256, lr=1e-3):
    """
    Train DANN: feature extractor + task classifier + domain classifier.
    Uses gradient reversal layer so the feature extractor learns to
    fool the domain classifier while being good at label prediction.
    """
    print(f"\n  Training DANN (gradient reversal)...")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_dim = X_source.shape[1]

    feat_ext = FeatureExtractor(input_dim=input_dim).to(device)
    task_clf = TaskClassifier(num_classes=num_classes).to(device)
    domain_clf = DomainClassifier().to(device)

    params = (list(feat_ext.parameters()) +
              list(task_clf.parameters()) +
              list(domain_clf.parameters()))
    optimizer = optim.Adam(params, lr=lr)
    cls_criterion = nn.CrossEntropyLoss()
    domain_criterion = nn.CrossEntropyLoss()

    source_ds = TensorDataset(
        torch.FloatTensor(X_source),
        torch.LongTensor(y_source)
    )
    target_ds = TensorDataset(torch.FloatTensor(X_target))

    source_loader = DataLoader(source_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    target_loader = DataLoader(target_ds, batch_size=batch_size, shuffle=True, drop_last=True)

    for epoch in range(epochs):
        feat_ext.train()
        task_clf.train()
        domain_clf.train()

        # Schedule alpha (ramp up domain adversarial strength)
        p = epoch / epochs
        alpha = 2.0 / (1.0 + np.exp(-10.0 * p)) - 1.0

        total_task_loss = 0
        total_domain_loss = 0
        n_batches = 0

        target_iter = iter(target_loader)
        for source_batch, source_labels in source_loader:
            try:
                (target_batch,) = next(target_iter)
            except StopIteration:
                target_iter = iter(target_loader)
                (target_batch,) = next(target_iter)

            source_batch = source_batch.to(device)
            source_labels = source_labels.to(device)
            target_batch = target_batch.to(device)

            bs = source_batch.size(0)
            bt = target_batch.size(0)

            # Domain labels: 0 = source, 1 = target
            domain_labels = torch.cat([
                torch.zeros(bs, dtype=torch.long),
                torch.ones(bt, dtype=torch.long)
            ]).to(device)

            # Forward
            source_features = feat_ext(source_batch)
            target_features = feat_ext(target_batch)

            # Task loss (only on labeled source)
            task_preds = task_clf(source_features)
            task_loss = cls_criterion(task_preds, source_labels)

            # Domain loss (on both, with gradient reversal)
            all_features = torch.cat([source_features, target_features], dim=0)
            domain_preds = domain_clf(all_features, alpha=alpha)
            domain_loss = domain_criterion(domain_preds, domain_labels)

            loss = task_loss + domain_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_task_loss += task_loss.item()
            total_domain_loss += domain_loss.item()
            n_batches += 1

        if (epoch + 1) % 20 == 0:
            avg_task = total_task_loss / n_batches
            avg_domain = total_domain_loss / n_batches
            print(f"    Epoch {epoch+1}/{epochs} — task: {avg_task:.4f}, domain: {avg_domain:.4f}, alpha: {alpha:.3f}")

    feat_ext.eval()
    task_clf.eval()
    print(f"  DANN trained.")
    return feat_ext, task_clf


# ---- Prediction ----

def predict(feat_ext, classifier, X):
    """Run inference through feature extractor + classifier."""
    device = next(feat_ext.parameters()).device
    X_tensor = torch.FloatTensor(X).to(device)
    with torch.no_grad():
        features = feat_ext(X_tensor)
        logits = classifier(features)
        preds = torch.argmax(logits, dim=1)
    return preds.cpu().numpy()
