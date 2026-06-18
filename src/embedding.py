"""
Autoencoder for learning traffic embeddings.

Takes the 41 NSL-KDD features and compresses them into a lower-dimensional
representation. The idea is that this learned embedding captures the
structure of the data better than raw features, and gives us a space
where we can apply domain adaptation (MMD alignment).
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


class TrafficAutoencoder(nn.Module):
    """
    Symmetric autoencoder: 41 -> 32 -> 16 -> 32 -> 41
    The 16-dimensional bottleneck is the embedding.
    """
    def __init__(self, input_dim=41, embedding_dim=16):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, embedding_dim),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim, 32),
            nn.ReLU(),
            nn.Linear(32, input_dim),
            nn.Sigmoid(),  # output in [0, 1] since features are min-max scaled
        )

    def forward(self, x):
        embedding = self.encoder(x)
        reconstruction = self.decoder(embedding)
        return reconstruction, embedding

    def encode(self, x):
        """Get just the embedding without reconstruction."""
        with torch.no_grad():
            return self.encoder(x)


def train_autoencoder(X_train, epochs=50, batch_size=256, lr=1e-3, embedding_dim=16):
    """
    Train the autoencoder on reconstruction loss.
    Returns the trained model.
    """
    print("\n  Training autoencoder...")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_dim = X_train.shape[1]

    model = TrafficAutoencoder(input_dim=input_dim, embedding_dim=embedding_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    dataset = TensorDataset(torch.FloatTensor(X_train))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for (batch,) in loader:
            batch = batch.to(device)
            reconstruction, _ = model(batch)
            loss = criterion(reconstruction, batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch.size(0)

        avg_loss = total_loss / len(dataset)
        if (epoch + 1) % 10 == 0:
            print(f"    Epoch {epoch+1}/{epochs} — reconstruction loss: {avg_loss:.6f}")

    model.eval()
    print(f"  Autoencoder trained. Embedding dimension: {embedding_dim}")
    return model


def get_embeddings(model, X, batch_size=4096):
    """Extract embeddings from trained autoencoder."""
    device = next(model.parameters()).device
    n_samples = X.shape[0]
    embeddings = []

    with torch.no_grad():
        for i in range(0, n_samples, batch_size):
            batch = torch.FloatTensor(X[i:i+batch_size]).to(device)
            emb = model.encoder(batch)
            embeddings.append(emb.cpu())

    return torch.cat(embeddings, dim=0).numpy()
