# VAE Hybrid Music Clustering (Easy Task)

This repo implements the Easy task: a basic VAE to extract latent features from numeric CSV data, clustering with K-Means, visualizing via t-SNE/UMAP, and comparing against a PCA+KMeans baseline using Silhouette and Calinski-Harabasz metrics.

## Setup

Install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Data

Place your features in `dataset/train.csv`.

- If your CSV is numeric-only (e.g., MFCCs), it will be used directly.
- If it contains text/categorical columns, the pipeline can automatically build features:
	- Text via TF-IDF (e.g., `Lyrics`)
	- Categorical via one-hot (e.g., `Genre`, `Language`)

Optionally, specify a label column name via `--label_column` to drop it from features.

## Run (Easy Task)

Basic numeric run:

```bash
python main_easy.py --train_csv dataset/train.csv --latent_dim 16 --n_clusters 5 --epochs 50
```

Mixed-type (text + categorical) run:

```bash
python main_easy.py \
	--train_csv dataset/train.csv \
	--drop_columns Artist Song \
	--text_cols Lyrics \
	--cat_cols Genre Language \
	--tfidf_max_features 2000 --tfidf_ngram_min 1 --tfidf_ngram_max 2 \
	--latent_dim 16 --n_clusters 5 --epochs 50
```

Fast dev run (skip visuals, fewer epochs):

```bash
python main_easy.py --train_csv dataset/train.csv --epochs 5 --n_clusters 2 --skip_visuals
```

Inspect which columns are kept/dropped:

```bash
python main_easy.py --train_csv dataset/train.csv --inspect_columns \
	--drop_columns Artist Song --text_cols Lyrics --cat_cols Genre Language
```

Outputs:

- `results/latent_visualization/tsne_latent.png` (if not skipped)
- `results/latent_visualization/umap_latent.png` (if not skipped)
- `results/clustering_metrics.csv`

## Notes

- The VAE is a simple MLP suitable for tabular features (e.g., MFCCs or TF-IDF embeddings of lyrics).
- If your data is tiny or near-identical, K-Means may collapse to < n_clusters; consider reducing `--n_clusters` or enriching features.
- For faster runs, use `--skip_visuals`, or selectively `--skip_tsne`/`--skip_umap`.
- GPU is used if available; pass `--cpu` to force CPU.
