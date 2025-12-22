from __future__ import annotations

import argparse
from pathlib import Path

import torch

from src.clustering import kmeans_cluster
from src.data import build_items, extract_audio_features
from src.evaluation import clustering_metrics
from src.models import BetaVAE, TrainConfig, encode_dataset
from src.train import train_vae
from src.utils import seed_everything
from src.visualize import embed_2d, save_scatter


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Hard task: Beta-VAE + metrics + visualizations")
    p.add_argument("--audio_dir", type=str, required=True)
    p.add_argument("--out_dir", type=str, default="results/hard")

    p.add_argument("--latent_dim", type=int, default=16)
    p.add_argument("--n_clusters", type=int, default=10)

    p.add_argument("--beta", type=float, default=4.0)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--device", type=str, default="cpu")

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_items", type=int, default=200)

    p.add_argument("--embed", choices=["tsne", "umap"], default="tsne")
    p.add_argument("--use_labels", action="store_true")
    return p


def run(args: argparse.Namespace) -> None:
    seed_everything(args.seed)
    items = build_items(Path(args.audio_dir))
    if args.max_items:
        items = items[: args.max_items]

    batch = extract_audio_features(items)

    model = BetaVAE(input_dim=batch.X.shape[1], latent_dim=args.latent_dim, beta=args.beta)
    cfg = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
        beta=args.beta,
    )
    train_vae(model, batch.X, cfg)

    Z = encode_dataset(model, torch.from_numpy(batch.X).to(cfg.device), batch_size=256).cpu().numpy()
    cl = kmeans_cluster(Z, n_clusters=args.n_clusters, seed=args.seed)

    m = clustering_metrics(Z, cl.labels, true_labels=batch.labels if args.use_labels else None)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    import pandas as pd

    pd.DataFrame([{"method": "beta-vae+kmeans", **m.__dict__}]).to_csv(out_dir / "metrics_hard.csv", index=False)

    E = embed_2d(Z, method=args.embed, seed=args.seed)
    save_scatter(E, cl.labels, out_dir / f"beta_vae_{args.embed}.png", title=f"Beta-VAE latent ({args.embed})")

    print("Wrote results to", out_dir)


if __name__ == "__main__":
    run(build_argparser().parse_args())
