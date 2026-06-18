"""
Alternative embedding architectures for traffic features: 1D-CNN and Transformer.

Flow-level features (41 for NSL-KDD, 56 for CICFlowMeter) are aggregate
statistics per connection — not time-ordered sequences. So LSTM doesn't
apply here. Instead we use:

- 1D-CNN: captures local feature group dependencies. NSL-KDD features are
  grouped (basic, content, traffic, host-level) and adjacent features are
  semantically related (e.g., serror_rate next to srv_serror_rate).
- Transformer: captures global feature interactions via self-attention.
  Any feature can attend to any other regardless of position — useful when
  important interactions span distant feature groups.

Both produce fixed-size embeddings comparable to the autoencoder in
embedding.py. The question is whether the inductive bias (local vs global)
affects robustness under distribution shift.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


# ---- 1D-CNN Embedding Model ----

class CNNEmbedding(nn.Module):
    """
    1D Convolutional embedding for flow-level features.

    Treats the feature vector as a 1D signal where adjacent features
    have related semantics. Multiple kernel sizes (3, 5, 7) capture
    different receptive fields over the feature vector.
    """
    def __init__(self, input_dim=41, embedding_dim=16, channels=32):
        super().__init__()
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim

        # Multi-scale 1D convolutions (different kernel sizes)
        self.conv_k3 = nn.Sequential(
            nn.Conv1d(1, channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(channels),
            nn.ReLU(),
        )
        self.conv_k5 = nn.Sequential(
            nn.Conv1d(1, channels, kernel_size=5, padding=2),
            nn.BatchNorm1d(channels),
            nn.ReLU(),
        )
        self.conv_k7 = nn.Sequential(
            nn.Conv1d(1, channels, kernel_size=7, padding=3),
            nn.BatchNorm1d(channels),
            nn.ReLU(),
        )
        # Second layer processes concatenated multi-scale features
        self.conv2 = nn.Sequential(
            nn.Conv1d(channels * 3, channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(channels),
            nn.ReLU(),
        )
        # Global average pooling → projection to embedding
        self.projection = nn.Sequential(
            nn.Linear(channels, embedding_dim),
            nn.ReLU(),
        )
        # Decoder for reconstruction
        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim, channels),
            nn.ReLU(),
            nn.Linear(channels, input_dim),
            nn.Sigmoid(),
        )

    def forward(self, x):
        """Returns (reconstruction, embedding)."""
        embedding = self.encode_forward(x)
        reconstruction = self.decoder(embedding)
        return reconstruction, embedding

    def encode_forward(self, x):
        """Encode input to embedding (with gradient)."""
        # Reshape: (batch, features) → (batch, 1, features) for Conv1d
        x_1d = x.unsqueeze(1)

        # Multi-scale convolutions
        c3 = self.conv_k3(x_1d)
        c5 = self.conv_k5(x_1d)
        c7 = self.conv_k7(x_1d)

        # Concatenate along channel dimension: (batch, channels*3, features)
        multi_scale = torch.cat([c3, c5, c7], dim=1)

        # Second convolution
        features = self.conv2(multi_scale)

        # Global average pooling: (batch, channels, features) → (batch, channels)
        pooled = features.mean(dim=2)

        embedding = self.projection(pooled)
        return embedding

    def encode(self, x):
        """Get embedding without gradient (for inference)."""
        with torch.no_grad():
            return self.encode_forward(x)


# ---- Transformer Embedding Model ----

class TransformerEmbedding(nn.Module):
    """
    Transformer encoder for feature interaction modeling.

    Each feature (or group of features) is treated as a token with
    learnable positional encoding. Self-attention captures pairwise
    interactions between any features regardless of position — useful
    since flow feature ordering is somewhat arbitrary.
    """
    def __init__(self, input_dim=41, embedding_dim=16, d_model=32,
                 nhead=4, num_layers=2, group_size=1, dropout=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.group_size = group_size
        self.seq_len = input_dim // group_size + (1 if input_dim % group_size else 0)
        self.padded_dim = self.seq_len * group_size
        self.d_model = d_model

        # Project each feature group to d_model dimensions
        self.input_projection = nn.Linear(group_size, d_model)

        # Learnable positional encoding
        self.pos_encoding = nn.Parameter(torch.randn(1, self.seq_len, d_model) * 0.02)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            activation='gelu',
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Pool + project to embedding
        self.projection = nn.Sequential(
            nn.Linear(d_model, embedding_dim),
            nn.ReLU(),
        )
        # Decoder for reconstruction
        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim, d_model),
            nn.ReLU(),
            nn.Linear(d_model, input_dim),
            nn.Sigmoid(),
        )

    def forward(self, x):
        """Returns (reconstruction, embedding)."""
        embedding = self.encode_forward(x)
        reconstruction = self.decoder(embedding)
        return reconstruction, embedding

    def encode_forward(self, x):
        """Encode input to embedding (with gradient)."""
        batch_size = x.size(0)
        # Pad and reshape
        if self.padded_dim > self.input_dim:
            padding = torch.zeros(batch_size, self.padded_dim - self.input_dim,
                                  device=x.device)
            x_padded = torch.cat([x, padding], dim=1)
        else:
            x_padded = x
        x_seq = x_padded.view(batch_size, self.seq_len, self.group_size)

        # Project to d_model and add positional encoding
        x_proj = self.input_projection(x_seq) + self.pos_encoding

        # Transformer encoding
        transformer_out = self.transformer(x_proj)

        # Mean pooling over sequence
        pooled = transformer_out.mean(dim=1)

        embedding = self.projection(pooled)
        return embedding

    def encode(self, x):
        """Get embedding without gradient (for inference)."""
        with torch.no_grad():
            return self.encode_forward(x)


# ---- Training Function (shared for both models) ----

def train_sequential_embedding(X_train, model_type='cnn', epochs=50,
                                batch_size=256, lr=1e-3, embedding_dim=16,
                                X_target=None, mmd_lambda=0.0):
    """
    Train a CNN or Transformer embedding model.

    If X_target is provided and mmd_lambda > 0, also minimizes MMD
    between source and target embeddings (joint domain adaptation).

    Args:
        X_train: source domain training data
        model_type: 'cnn' or 'transformer'
        epochs: training epochs
        batch_size: batch size
        lr: learning rate
        embedding_dim: output embedding dimension
        X_target: optional target domain data for MMD alignment
        mmd_lambda: weight for MMD loss (0 = no adaptation)

    Returns:
        trained model, training history dict
    """
    from src.domain_adaptation import compute_mmd

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_dim = X_train.shape[1]

    print(f"\n  Training {model_type.upper()} embedding (dim={embedding_dim})...")

    if model_type == 'cnn':
        model = CNNEmbedding(input_dim=input_dim, embedding_dim=embedding_dim).to(device)
    elif model_type == 'transformer':
        model = TransformerEmbedding(input_dim=input_dim, embedding_dim=embedding_dim).to(device)
    else:
        raise ValueError(f"Unknown model_type: {model_type}. Use 'cnn' or 'transformer'.")

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    # Data loaders
    source_dataset = TensorDataset(torch.FloatTensor(X_train))
    source_loader = DataLoader(source_dataset, batch_size=batch_size, shuffle=True)

    target_loader = None
    if X_target is not None and mmd_lambda > 0:
        target_dataset = TensorDataset(torch.FloatTensor(X_target))
        target_loader = DataLoader(target_dataset, batch_size=batch_size, shuffle=True)

    history = {'recon_loss': [], 'mmd_loss': [], 'total_loss': []}

    model.train()
    for epoch in range(epochs):
        epoch_recon = 0.0
        epoch_mmd = 0.0
        n_batches = 0

        target_iter = iter(target_loader) if target_loader else None

        for (source_batch,) in source_loader:
            source_batch = source_batch.to(device)
            reconstruction, embedding_s = model(source_batch)

            # Reconstruction loss
            recon_loss = criterion(reconstruction, source_batch)

            # MMD loss (if target provided)
            mmd_loss = torch.tensor(0.0, device=device)
            if target_iter is not None:
                try:
                    (target_batch,) = next(target_iter)
                except StopIteration:
                    target_iter = iter(target_loader)
                    (target_batch,) = next(target_iter)
                target_batch = target_batch.to(device)
                _, embedding_t = model(target_batch)
                mmd_loss = compute_mmd(embedding_s, embedding_t)

            total_loss = recon_loss + mmd_lambda * mmd_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            epoch_recon += recon_loss.item()
            epoch_mmd += mmd_loss.item()
            n_batches += 1

        avg_recon = epoch_recon / n_batches
        avg_mmd = epoch_mmd / n_batches
        history['recon_loss'].append(avg_recon)
        history['mmd_loss'].append(avg_mmd)
        history['total_loss'].append(avg_recon + mmd_lambda * avg_mmd)

        if (epoch + 1) % 10 == 0:
            msg = f"    Epoch {epoch+1}/{epochs} — recon: {avg_recon:.6f}"
            if mmd_lambda > 0:
                msg += f", MMD: {avg_mmd:.6f}"
            print(msg)

    model.eval()
    param_count = sum(p.numel() for p in model.parameters())
    print(f"  {model_type.upper()} trained. Parameters: {param_count:,}")
    return model, history


def get_embeddings_sequential(model, X, batch_size=4096):
    """Extract embeddings from a trained CNN or Transformer model."""
    device = next(model.parameters()).device
    n_samples = X.shape[0]
    embeddings = []

    with torch.no_grad():
        for i in range(0, n_samples, batch_size):
            batch = torch.FloatTensor(X[i:i+batch_size]).to(device)
            emb = model.encode_forward(batch)
            embeddings.append(emb.cpu())

    return torch.cat(embeddings, dim=0).numpy()
