import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score, calinski_harabasz_score


def compute_metrics(X: np.ndarray, labels: np.ndarray) -> dict:
    metrics = {}
    # Silhouette requires >1 cluster and samples > n_clusters
    try:
        metrics['silhouette'] = float(silhouette_score(X, labels))
    except Exception:
        metrics['silhouette'] = np.nan
    try:
        metrics['calinski_harabasz'] = float(calinski_harabasz_score(X, labels))
    except Exception:
        metrics['calinski_harabasz'] = np.nan
    return metrics


def save_metrics_csv(results: dict, out_csv: str):
    df = pd.DataFrame([results])
    df.to_csv(out_csv, index=False)
