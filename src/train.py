from __future__ import annotations

from dataclasses import asdict

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from .models import TrainConfig, vae_loss


def train_vae(
    model: torch.nn.Module,
    X: np.ndarray,
    cfg: TrainConfig,
) -> dict:
    device = torch.device(cfg.device)
    model = model.to(device)

    ds = TensorDataset(torch.from_numpy(X))
    dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, drop_last=False)

    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    history: list[dict[str, float]] = []
    for epoch in range(cfg.epochs):
        model.train()
        acc = {"recon": 0.0, "kld": 0.0, "loss": 0.0}
        n = 0
        for (xb,) in dl:
            xb = xb.to(device)
            opt.zero_grad(set_to_none=True)
            recon, mu, logvar = model(xb)
            loss, parts = vae_loss(xb, recon, mu, logvar, beta=cfg.beta)
            loss.backward()
            opt.step()

            bs = xb.shape[0]
            n += bs
            for k in acc:
                acc[k] += parts[k] * bs

        for k in acc:
            acc[k] /= max(1, n)
        acc["epoch"] = float(epoch + 1)
        history.append(acc)

    return {"config": asdict(cfg), "history": history}
