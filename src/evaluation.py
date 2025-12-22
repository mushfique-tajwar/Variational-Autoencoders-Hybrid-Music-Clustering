from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Metrics:
    silhouette: float | None
    calinski_harabasz: float | None
    davies_bouldin: float | None
    ari: float | None
    nmi: float | None
    purity: float | None


def _has_enough_clusters(pred: np.ndarray) -> bool:
    uniq = np.unique(pred)
    # DBSCAN may have -1 noise; ignore noise for cluster count
    uniq = uniq[uniq != -1]
    return uniq.size >= 2


def clustering_metrics(
    X: np.ndarray,
    pred: np.ndarray,
    true_labels: list[str] | None = None,
) -> Metrics:
    from sklearn.metrics import (
        adjusted_rand_score,
        calinski_harabasz_score,
        davies_bouldin_score,
        normalized_mutual_info_score,
        silhouette_score,
    )

    sil = ch = db = None
    if _has_enough_clusters(pred):
        # For silhouette we need at least 2 clusters and no all-noise
        try:
            sil = float(silhouette_score(X, pred))
        except Exception:
            sil = None
        try:
            ch = float(calinski_harabasz_score(X, pred))
        except Exception:
            ch = None
        try:
            db = float(davies_bouldin_score(X, pred))
        except Exception:
            db = None

    ari = nmi = purity = None
    if true_labels is not None:
        # encode labels
        from sklearn.preprocessing import LabelEncoder

        le = LabelEncoder()
        y_true = le.fit_transform(true_labels)
        try:
            ari = float(adjusted_rand_score(y_true, pred))
        except Exception:
            ari = None
        try:
            nmi = float(normalized_mutual_info_score(y_true, pred))
        except Exception:
            nmi = None
        try:
            purity = float(cluster_purity(y_true, pred))
        except Exception:
            purity = None

    return Metrics(
        silhouette=sil,
        calinski_harabasz=ch,
        davies_bouldin=db,
        ari=ari,
        nmi=nmi,
        purity=purity,
    )


def cluster_purity(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    # ignore noise label -1 in predicted clusters
    mask = y_pred != -1
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    if y_true.size == 0:
        return 0.0

    total = y_true.size
    purity_sum = 0
    for c in np.unique(y_pred):
        idx = y_pred == c
        if not np.any(idx):
            continue
        vals, counts = np.unique(y_true[idx], return_counts=True)
        purity_sum += int(counts.max())
    return purity_sum / total
