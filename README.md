# Variational-Autoencoders-Hybrid-Music-Clustering

This repo contains a runnable end-to-end implementation of the **Easy / Medium / Hard** tasks from `Guidelines.md`.

It uses the provided `dataset/audio` folder (GTZAN-style files like `blues.00000.au`) and optionally `dataset/lyrics` if you add matching lyric files.

## Setup

On many Linux distros, `pip` is blocked from installing system-wide packages (PEP 668: `externally-managed-environment`).
Use a **virtual environment** instead.

### Option A: Use the existing repo venv (recommended)

This repo already supports a local venv at `.venv/`.

```bash
./.venv/bin/python -m pip install -r requirements.txt
```

### Option B: Create a new venv

```bash
python3 -m venv .venv
./.venv/bin/python -m pip install --upgrade pip
./.venv/bin/python -m pip install -r requirements.txt
```

```bash
pip install -r requirements.txt
```

## Easy task (VAE + KMeans + t-SNE/UMAP + PCA baseline)

```bash
./.venv/bin/python -m scripts.main_easy --audio_dir dataset/audio --latent_dim 16 --n_clusters 5 --epochs 30 --max_items 200 --embed tsne
```


Outputs in `results/easy/`:

- `metrics_easy.csv`
- `vae_tsne.png`, `pca_tsne.png` (or UMAP)

If you want label-based metrics (ARI/NMI/Purity), enable:

```bash
./.venv/bin/python -m scripts.main_easy --audio_dir dataset/audio --use_labels
```

Labels are inferred from filename prefixes (e.g. `blues`, `classical`).

```bash
./.venv/bin/python -m scripts.main_easy --audio_dir dataset/audio
```

## Medium task (audio+lyrics fusion, more clustering)

```bash
./.venv/bin/python -m scripts.main_medium --audio_dir dataset/audio --epochs 30 --embed tsne
```

If you have lyrics:

```bash
./.venv/bin/python -m scripts.main_medium --audio_dir dataset/audio --lyrics_dir dataset/lyrics --use_lyrics
```


```bash
./.venv/bin/python -m scripts.main_medium --audio_dir dataset/audio
```


## Hard task (Beta-VAE)

```bash
./.venv/bin/python -m scripts.main_hard --audio_dir dataset/audio --beta 4.0 --epochs 50 --n_clusters 10 --use_labels
```

```bash
./.venv/bin/python -m scripts.main_hard --audio_dir dataset/audio --use_labels
```

## Notes

- This is designed to be **lightweight and reproducible** rather than state-of-the-art.
- Audio features are log-mel spectrograms padded/truncated to a fixed length.
