import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

class VAE(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int = 16, hidden_dims=(128, 64)):
        super().__init__()
        h1, h2 = hidden_dims
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, h1),
            nn.ReLU(),
            nn.Linear(h1, h2),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(h2, latent_dim)
        self.fc_logvar = nn.Linear(h2, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, h2),
            nn.ReLU(),
            nn.Linear(h2, h1),
            nn.ReLU(),
            nn.Linear(h1, input_dim),
        )

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar


def loss_fn(recon_x: torch.Tensor, x: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    # Reconstruction loss (MSE) + KL divergence
    recon_loss = F.mse_loss(recon_x, x, reduction='mean')
    kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kld


def train_vae(model: VAE, dataloader, epochs: int = 50, lr: float = 1e-3, device: str = 'cpu'):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    skipped = 0
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        for batch in dataloader:
            x = batch.to(device)
            # sanitize batch
            x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
            if torch.isnan(x).any() or torch.isinf(x).any():
                skipped += 1
                continue
            optimizer.zero_grad()
            recon, mu, logvar = model(x)
            loss = loss_fn(recon, x, mu, logvar)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x.size(0)
        avg_loss = total_loss / len(dataloader.dataset)
        if epoch % max(1, epochs // 10) == 0 or epoch == 1:
            print(f"Epoch {epoch}/{epochs} | loss: {avg_loss:.4f}")
    if skipped:
        print(f"Warning: skipped {skipped} batches due to NaN/Inf inputs.")
    return model


def get_latents(model: VAE, dataloader, device: str = 'cpu') -> torch.Tensor:
    model.to(device)
    model.eval()
    latents = []
    with torch.no_grad():
        for batch in dataloader:
            x = batch.to(device)
            mu, logvar = model.encode(x)
            z = mu  # use mean as deterministic latent
            latents.append(z.cpu())
    return torch.cat(latents, dim=0)
