from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from src.clustering import kmeans_cluster
from src.data import build_items, extract_audio_features
from src.evaluation import clustering_metrics
from src.models import MLPVAE, TrainConfig, encode_dataset
from src.train import train_vae
from src.utils import seed_everything
from src.visualize import embed_2d, save_scatter


def run(args: argparse.Namespace) -> dict:
    seed_everything(args.seed)

    items = build_items(Path(args.audio_dir), Path(args.lyrics_dir) if args.lyrics_dir else None)
    if args.max_items:
        items = items[: args.max_items]

    batch = extract_audio_features(
        items,
        sample_rate=args.sample_rate,
        n_mels=args.n_mels,
        target_frames=args.target_frames,
    )

    model = MLPVAE(input_dim=batch.X.shape[1], latent_dim=args.latent_dim)
    cfg = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
        beta=1.0,
    )
    train_info = train_vae(model, batch.X, cfg)

    Z = encode_dataset(model, torch.from_numpy(batch.X).to(cfg.device), batch_size=256).cpu().numpy()

    # VAE + KMeans
    cl = kmeans_cluster(Z, n_clusters=args.n_clusters, seed=args.seed)
    m = clustering_metrics(Z, cl.labels, true_labels=batch.labels if args.use_labels else None)

    # PCA baseline + KMeans
    from sklearn.decomposition import PCA

    Zp = PCA(n_components=min(args.latent_dim, batch.X.shape[1]), random_state=args.seed).fit_transform(batch.X)
    cl_pca = kmeans_cluster(Zp, n_clusters=args.n_clusters, seed=args.seed)
    m_pca = clustering_metrics(Zp, cl_pca.labels, true_labels=batch.labels if args.use_labels else None)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # visualizations
    E = embed_2d(Z, method=args.embed, seed=args.seed)
    save_scatter(E, cl.labels, out_dir / f"vae_{args.embed}.png", title=f"VAE latent ({args.embed}) - {cl.method}")

    Ep = embed_2d(Zp, method=args.embed, seed=args.seed)
    save_scatter(Ep, cl_pca.labels, out_dir / f"pca_{args.embed}.png", title=f"PCA baseline ({args.embed}) - {cl_pca.method}")

    # metrics CSV
    import pandas as pd

    rows = [
        {
            "method": "vae+kmeans",
            **m.__dict__,
        },
        {
            "method": "pca+kmeans",
            **m_pca.__dict__,
        },
    ]
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "metrics_easy.csv", index=False)

    return {"train": train_info, "metrics": rows}


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Easy task: MLP-VAE + KMeans + t-SNE/UMAP + PCA baseline")
    p.add_argument("--audio_dir", type=str, required=True)
    p.add_argument("--lyrics_dir", type=str, default="")
    p.add_argument("--out_dir", type=str, default="results/easy")

    p.add_argument("--latent_dim", type=int, default=16)
    p.add_argument("--n_clusters", type=int, default=5)

    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_items", type=int, default=200)

    p.add_argument("--sample_rate", type=int, default=22050)
    p.add_argument("--n_mels", type=int, default=64)
    p.add_argument("--target_frames", type=int, default=256)

    p.add_argument("--embed", choices=["tsne", "umap"], default="tsne")
    p.add_argument("--use_labels", action="store_true", help="Use inferred labels (genre from filename) for ARI/NMI/Purity")
    return p


if __name__ == "__main__":
    args = build_argparser().parse_args()
    out = run(args)
    print("Wrote results to", args.out_dir)
    print(out["metrics"])
