from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from src.clustering import agglomerative_cluster, dbscan_cluster, kmeans_cluster
from src.data import build_items, extract_audio_features, extract_lyrics_features, fuse_features
from src.evaluation import clustering_metrics
from src.models import MLPVAE, TrainConfig, encode_dataset
from src.train import train_vae
from src.utils import seed_everything
from src.visualize import embed_2d, save_scatter


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Medium task: audio+lyrics fusion, extra clustering methods")
    p.add_argument("--audio_dir", type=str, required=True)
    p.add_argument("--lyrics_dir", type=str, default="")
    p.add_argument("--use_lyrics", action="store_true")

    p.add_argument("--out_dir", type=str, default="results/medium")

    p.add_argument("--latent_dim", type=int, default=16)
    p.add_argument("--n_clusters", type=int, default=5)

    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_items", type=int, default=200)

    p.add_argument("--embed", choices=["tsne", "umap"], default="tsne")
    p.add_argument("--use_labels", action="store_true")

    # DBSCAN params
    p.add_argument("--dbscan_eps", type=float, default=0.8)
    p.add_argument("--dbscan_min_samples", type=int, default=5)

    return p


def run(args: argparse.Namespace) -> None:
    seed_everything(args.seed)
    items = build_items(Path(args.audio_dir), Path(args.lyrics_dir) if args.lyrics_dir else None)
    if args.max_items:
        items = items[: args.max_items]

    audio = extract_audio_features(items)
    if args.use_lyrics:
        lyr = extract_lyrics_features(items)
        features = fuse_features(audio, lyr)
    else:
        features = audio

    model = MLPVAE(input_dim=features.X.shape[1], latent_dim=args.latent_dim, hidden_dims=(1024, 512, 256))
    cfg = TrainConfig(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, device=args.device, beta=1.0)
    train_vae(model, features.X, cfg)

    Z = encode_dataset(model, torch.from_numpy(features.X).to(cfg.device), batch_size=256).cpu().numpy()

    outs = []

    cl_km = kmeans_cluster(Z, n_clusters=args.n_clusters, seed=args.seed)
    outs.append(("vae+kmeans", cl_km))

    cl_ag = agglomerative_cluster(Z, n_clusters=args.n_clusters)
    outs.append(("vae+agglomerative", cl_ag))

    cl_db = dbscan_cluster(Z, eps=args.dbscan_eps, min_samples=args.dbscan_min_samples)
    outs.append(("vae+dbscan", cl_db))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    import pandas as pd

    rows = []
    for name, cl in outs:
        m = clustering_metrics(Z, cl.labels, true_labels=features.labels if args.use_labels else None)
        rows.append({"method": name, **m.__dict__})

        E = embed_2d(Z, method=args.embed, seed=args.seed)
        save_scatter(E, cl.labels, out_dir / f"{name.replace('+','_')}_{args.embed}.png", title=f"{name} ({args.embed})")

    pd.DataFrame(rows).to_csv(out_dir / "metrics_medium.csv", index=False)
    print("Wrote results to", out_dir)


if __name__ == "__main__":
    run(build_argparser().parse_args())
