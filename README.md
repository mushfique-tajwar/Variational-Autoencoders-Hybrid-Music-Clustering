# VAE Hybrid Music Clustering (Easy Task)

This repo implements the Easy task: a basic VAE to extract latent features from numeric CSV data, clustering with K-Means, visualizing via t-SNE/UMAP, and comparing against a PCA+KMeans baseline using Silhouette and Calinski-Harabasz metrics.

## Setup

Install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Data

Place your numeric features in `dataset/train.csv`. Optionally, specify a label column name via `--label_column` to drop it from features.

## Run (Easy Task)

```bash
python main_easy.py --train_csv dataset/train.csv --latent_dim 16 --n_clusters 5 --epochs 50
```

Outputs:
- `results/latent_visualization/tsne_latent.png`
- `results/latent_visualization/umap_latent.png`
- `results/clustering_metrics.csv`

## Notes
- The VAE is a simple MLP suitable for tabular features (e.g., MFCCs or precomputed embeddings).
- For very small datasets, t-SNE/UMAP may show limited separation and metrics may be undefined; they’re guarded to return NaN.
- GPU is used if available; pass `--cpu` to force CPU.
