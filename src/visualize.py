from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def embed_2d(Z: np.ndarray, method: str = "tsne", seed: int = 42) -> np.ndarray:
    method = method.lower()
    if method == "tsne":
        from sklearn.manifold import TSNE

        # t-SNE requires perplexity < n_samples. Use a conservative default that still works for
        # tiny smoke tests.
        n = int(Z.shape[0])
        perplexity = float(min(30.0, max(2.0, (n - 1) / 3)))
        return TSNE(
            n_components=2,
            init="pca",
            learning_rate="auto",
            random_state=seed,
            perplexity=perplexity,
        ).fit_transform(Z)
    if method == "umap":
        import umap

        return umap.UMAP(n_components=2, random_state=seed).fit_transform(Z)
    raise ValueError(f"Unknown embedding method: {method}")


def save_scatter(
    E: np.ndarray,
    cluster_labels: np.ndarray,
    out_path: Path,
    *,
    title: str,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(7, 6))
    uniq = np.unique(cluster_labels)
    for u in uniq:
        idx = cluster_labels == u
        plt.scatter(E[idx, 0], E[idx, 1], s=18, alpha=0.8, label=str(u))
    plt.title(title)
    plt.xlabel("dim-1")
    plt.ylabel("dim-2")
    if uniq.size <= 20:
        plt.legend(markerscale=1.2, bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(out_path.as_posix(), dpi=160)
    plt.close()
