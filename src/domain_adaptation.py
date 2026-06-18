"""
MMD-based domain adaptation.

Maximum Mean Discrepancy (MMD) measures how different two distributions are
in a reproducing kernel Hilbert space. If MMD is zero, the distributions
are identical. We use it here to align source (training) and target (test)
embeddings so the classifier generalizes better across environments.

The training loop jointly minimizes:
  1. Reconstruction loss (autoencoder should still produce good embeddings)
  2. MMD loss (source and target embeddings should look similar)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from src.embedding import TrafficAutoencoder


def compute_mmd(source, target, kernel='rbf', bandwidth=1.0):
    """
    Compute MMD between two sets of samples using RBF (Gaussian) kernel.

    MMD^2 = E[k(xs, xs')] + E[k(xt, xt')] - 2*E[k(xs, xt)]

    where k is the RBF kernel: k(x, y) = exp(-||x-y||^2 / (2 * bandwidth^2))
    """
    n_source = source.size(0)
    n_target = target.size(0)

    # Kernel computations
    ss_kernel = _rbf_kernel(source, source, bandwidth)
    tt_kernel = _rbf_kernel(target, target, bandwidth)
    st_kernel = _rbf_kernel(source, target, bandwidth)

    # MMD^2 estimate
    mmd = (ss_kernel.sum() / (n_source * n_source)
           + tt_kernel.sum() / (n_target * n_target)
           - 2 * st_kernel.sum() / (n_source * n_target))

    return mmd


def _rbf_kernel(x, y, bandwidth):
    """RBF kernel matrix between x and y."""
    x_size = x.size(0)
    y_size = y.size(0)
    dim = x.size(1)

    # Expand for pairwise distance computation
    x = x.unsqueeze(1)  # (n, 1, d)
    y = y.unsqueeze(0)  # (1, m, d)

    # Pairwise squared distances
    dist = ((x - y) ** 2).sum(dim=2)  # (n, m)

    return torch.exp(-dist / (2 * bandwidth ** 2))


def compute_multi_scale_mmd(source, target, bandwidths=[0.1, 0.5, 1.0, 2.0, 5.0]):
    """
    Multi-scale MMD uses multiple bandwidth values and sums them.
    More robust than picking a single bandwidth.
    """
    total_mmd = 0
    for bw in bandwidths:
        total_mmd += compute_mmd(source, target, bandwidth=bw)
    return total_mmd / len(bandwidths)


def train_with_mmd(X_source, X_target, epochs=50, batch_size=256,
                   lr=1e-3, embedding_dim=16, mmd_weight=1.0):
    """
    Train autoencoder with joint reconstruction + MMD loss.

    The autoencoder learns embeddings that:
    1. Can reconstruct the input well (preserves information)
    2. Have similar distributions for source and target (domain-invariant)

    X_source: training data (source domain)
    X_target: test data features only, no labels used (target domain)
    mmd_weight: how much to penalize distribution mismatch (lambda)
    """
    print(f"\n  Training autoencoder with MMD alignment (lambda={mmd_weight})...")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_dim = X_source.shape[1]

    model = TrafficAutoencoder(input_dim=input_dim, embedding_dim=embedding_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    recon_criterion = nn.MSELoss()

    source_dataset = TensorDataset(torch.FloatTensor(X_source))
    target_dataset = TensorDataset(torch.FloatTensor(X_target))

    source_loader = DataLoader(source_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    target_loader = DataLoader(target_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    model.train()
    for epoch in range(epochs):
        total_recon_loss = 0
        total_mmd_loss = 0
        n_batches = 0

        target_iter = iter(target_loader)
        for (source_batch,) in source_loader:
            # Get a target batch (cycle if target is smaller)
            try:
                (target_batch,) = next(target_iter)
            except StopIteration:
                target_iter = iter(target_loader)
                (target_batch,) = next(target_iter)

            source_batch = source_batch.to(device)
            target_batch = target_batch.to(device)

            # Forward pass
            source_recon, source_emb = model(source_batch)
            target_recon, target_emb = model(target_batch)

            # Reconstruction loss (both domains should reconstruct well)
            recon_loss = (recon_criterion(source_recon, source_batch)
                         + recon_criterion(target_recon, target_batch))

            # MMD loss (embeddings should be aligned)
            mmd_loss = compute_multi_scale_mmd(source_emb, target_emb)

            # Combined loss
            loss = recon_loss + mmd_weight * mmd_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_recon_loss += recon_loss.item()
            total_mmd_loss += mmd_loss.item()
            n_batches += 1

        if (epoch + 1) % 10 == 0:
            avg_recon = total_recon_loss / n_batches
            avg_mmd = total_mmd_loss / n_batches
            print(f"    Epoch {epoch+1}/{epochs} — recon: {avg_recon:.6f}, MMD: {avg_mmd:.6f}")

    model.eval()
    print(f"  Domain-adapted autoencoder trained. Embedding dim: {embedding_dim}")
    return model


def measure_domain_shift(model, X_source, X_target):
    """
    Compute MMD between source and target in embedding space.
    Lower is better (distributions are more aligned).
    """
    device = next(model.parameters()).device

    with torch.no_grad():
        source_emb = model.encoder(torch.FloatTensor(X_source).to(device))
        target_emb = model.encoder(torch.FloatTensor(X_target).to(device))

        # Use a subsample for efficiency
        n = min(2000, source_emb.size(0), target_emb.size(0))
        idx_s = torch.randperm(source_emb.size(0))[:n]
        idx_t = torch.randperm(target_emb.size(0))[:n]

        mmd = compute_multi_scale_mmd(source_emb[idx_s], target_emb[idx_t])

    return mmd.item()
