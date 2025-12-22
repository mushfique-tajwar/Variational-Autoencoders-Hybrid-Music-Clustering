from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn


class MLPVAE(nn.Module):
    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 16,
        hidden_dims: tuple[int, ...] = (512, 256),
    ) -> None:
        super().__init__()

        enc_layers: list[nn.Module] = []
        prev = input_dim
        for h in hidden_dims:
            enc_layers.extend([nn.Linear(prev, h), nn.ReLU()])
            prev = h
        self.encoder = nn.Sequential(*enc_layers)
        self.fc_mu = nn.Linear(prev, latent_dim)
        self.fc_logvar = nn.Linear(prev, latent_dim)

        dec_layers: list[nn.Module] = []
        prev = latent_dim
        for h in reversed(hidden_dims):
            dec_layers.extend([nn.Linear(prev, h), nn.ReLU()])
            prev = h
        dec_layers.append(nn.Linear(prev, input_dim))
        self.decoder = nn.Sequential(*dec_layers)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar


class BetaVAE(MLPVAE):
    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 16,
        hidden_dims: tuple[int, ...] = (512, 256),
        beta: float = 4.0,
    ) -> None:
        super().__init__(input_dim=input_dim, latent_dim=latent_dim, hidden_dims=hidden_dims)
        self.beta = beta


@dataclass
class TrainConfig:
    epochs: int = 30
    batch_size: int = 32
    lr: float = 1e-3
    device: str = "cpu"
    beta: float = 1.0


def vae_loss(
    x: torch.Tensor,
    recon: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    *,
    beta: float = 1.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    recon_loss = nn.functional.mse_loss(recon, x, reduction="mean")
    kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    loss = recon_loss + beta * kld
    return loss, {"recon": float(recon_loss.detach()), "kld": float(kld.detach()), "loss": float(loss.detach())}


@torch.no_grad()
def encode_dataset(model: nn.Module, X: torch.Tensor, batch_size: int = 256) -> torch.Tensor:
    model.eval()
    mus: list[torch.Tensor] = []
    for i in range(0, X.shape[0], batch_size):
        xb = X[i : i + batch_size]
        mu, _ = model.encode(xb)
        mus.append(mu)
    return torch.cat(mus, dim=0)
