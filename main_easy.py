import os
import argparse
import numpy as np
import torch

from src.dataset import make_dataloader
from src.vae import VAE, train_vae, get_latents
from src.clustering import kmeans_clusters, visualize_embedding, pca_baseline
from src.evaluation import compute_metrics, save_metrics_csv


def run(args):
    device = 'cuda' if torch.cuda.is_available() and not args.cpu else 'cpu'
    print(f"Using device: {device}")

    # Load dataset
    dl, ds = make_dataloader(args.train_csv, batch_size=args.batch_size, shuffle=True, label_column=args.label_column)
    input_dim = ds.X.shape[1]

    # Train VAE
    vae = VAE(input_dim=input_dim, latent_dim=args.latent_dim)
    vae = train_vae(vae, dl, epochs=args.epochs, lr=args.lr, device=device)

    # Get latents
    Z = get_latents(vae, dl, device=device).numpy()

    # KMeans on latents
    labels, km = kmeans_clusters(Z, n_clusters=args.n_clusters)

    # Visualizations
    os.makedirs('results/latent_visualization', exist_ok=True)
    visualize_embedding(Z, labels, method='tsne', out_path='results/latent_visualization/tsne_latent.png')
    visualize_embedding(Z, labels, method='umap', out_path='results/latent_visualization/umap_latent.png')

    # Baseline: PCA + KMeans
    X = ds.X.numpy()
    Z_pca = pca_baseline(X, n_components=args.latent_dim)
    labels_pca, _ = kmeans_clusters(Z_pca, n_clusters=args.n_clusters)

    # Metrics
    metrics_vae = compute_metrics(Z, labels)
    metrics_pca = compute_metrics(Z_pca, labels_pca)
    results = {
        'method': ['vae+kmeans', 'pca+kmeans'],
        'silhouette': [metrics_vae['silhouette'], metrics_pca['silhouette']],
        'calinski_harabasz': [metrics_vae['calinski_harabasz'], metrics_pca['calinski_harabasz']],
    }
    import pandas as pd
    pd.DataFrame(results).to_csv('results/clustering_metrics.csv', index=False)

    print("Saved metrics to results/clustering_metrics.csv")
    print("Saved visualizations to results/latent_visualization/")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_csv', type=str, default='dataset/train.csv')
    parser.add_argument('--label_column', type=str, default=None, help='Optional label column to drop from features')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--latent_dim', type=int, default=16)
    parser.add_argument('--n_clusters', type=int, default=5)
    parser.add_argument('--cpu', action='store_true')
    args = parser.parse_args()
    run(args)
