import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Tuple, List, Dict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder

class CSVDataset(Dataset):
    def __init__(
        self,
        csv_path: str,
        label_column: Optional[str] = None,
        drop_columns: Optional[List[str]] = None,
        normalize: bool = True,
        enable_text_features: bool = True,
        text_cols: Optional[List[str]] = None,
        cat_cols: Optional[List[str]] = None,
        tfidf_max_features: int = 1000,
        tfidf_ngram_min: int = 1,
        tfidf_ngram_max: int = 1,
    ):
        df = pd.read_csv(csv_path)
        # Optionally separate labels
        if label_column and label_column in df.columns:
            self.labels = df[label_column].values
            df = df.drop(columns=[label_column])
        else:
            self.labels = None

        # Drop any user-specified text columns (e.g., titles, ids)
        if drop_columns:
            drop_cols = [c for c in drop_columns if c in df.columns]
            if drop_cols:
                df = df.drop(columns=drop_cols)

        # Attempt to coerce all columns to numeric where possible
        df_numeric = df.apply(pd.to_numeric, errors='coerce')
        # Keep columns that became numeric and drop all-NaN columns
        features = df_numeric.dropna(axis=1, how='all')
        X_parts: List[np.ndarray] = []
        feature_names: List[str] = []

        # Numeric block
        if features.shape[1] > 0:
            num_block = features.fillna(features.mean(axis=0)).fillna(0.0)
            X_parts.append(num_block.values)
            feature_names.extend(list(num_block.columns))

        # If no numeric features and text features enabled, build text/categorical features
        if features.shape[1] == 0 and enable_text_features:
            obj_cols = [c for c in df.columns if df[c].dtype == 'object']
            # Determine text vs categorical columns
            if text_cols is not None:
                text_candidates = [c for c in text_cols if c in df.columns]
            else:
                text_candidates = []
                for c in obj_cols:
                    cname = c.lower()
                    if cname in {'lyrics','text','content','description','transcript'}:
                        text_candidates.append(c)
                        continue
                    # Heuristic: long average string length implies text
                    vals = df[c].astype(str).fillna('')
                    avg_len = vals.str.len().mean()
                    if avg_len >= 20:
                        text_candidates.append(c)
            cat_candidates = [c for c in obj_cols if c not in text_candidates]

            # TF-IDF for each text column
            for c in text_candidates:
                vec = TfidfVectorizer(max_features=tfidf_max_features, ngram_range=(tfidf_ngram_min, tfidf_ngram_max))
                mat = vec.fit_transform(df[c].fillna("").astype(str))
                X_parts.append(mat.toarray())
                # name features as c:term; store only count for brevity
                feature_names.extend([f"tfidf_{c}_{i}" for i in range(mat.shape[1])])

            # One-hot for categorical columns
            if cat_cols is not None:
                cat_candidates = [c for c in cat_cols if c in df.columns]
            if cat_candidates:
                try:
                    ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
                except TypeError:
                    # For older scikit-learn versions
                    ohe = OneHotEncoder(handle_unknown='ignore', sparse=False)
                mat = ohe.fit_transform(df[cat_candidates].astype(str))
                X_parts.append(mat)
                feature_names.extend([f"ohe_{i}" for i in range(mat.shape[1])])

        if not X_parts:
            raise ValueError(
                "No numeric/text features could be constructed. "
                "Provide numeric features or enable text features / specify --text_cols/--cat_cols."
            )

        X = np.hstack(X_parts).astype(np.float32)
        self.feature_cols = feature_names if feature_names else [f"f{i}" for i in range(X.shape[1])]
        # Ensure no NaN/Inf before normalization
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        if normalize:
            # Standardize per column: (x - mean) / std (avoid div by zero)
            mean = X.mean(axis=0)
            std = X.std(axis=0)
            std[std == 0] = 1.0
            X = (X - mean) / std
            # Final sanitize
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        self.X = torch.from_numpy(X)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx]


def make_dataloader(
    csv_path: str,
    batch_size: int = 64,
    shuffle: bool = True,
    label_column: Optional[str] = None,
    drop_columns: Optional[List[str]] = None,
    enable_text_features: bool = True,
    text_cols: Optional[List[str]] = None,
    cat_cols: Optional[List[str]] = None,
    tfidf_max_features: int = 1000,
    tfidf_ngram_min: int = 1,
    tfidf_ngram_max: int = 1,
) -> Tuple[DataLoader, CSVDataset]:
    ds = CSVDataset(
        csv_path,
        label_column=label_column,
        drop_columns=drop_columns,
        normalize=True,
        enable_text_features=enable_text_features,
        text_cols=text_cols,
        cat_cols=cat_cols,
        tfidf_max_features=tfidf_max_features,
        tfidf_ngram_min=tfidf_ngram_min,
        tfidf_ngram_max=tfidf_ngram_max,
    )
    dl = DataLoader(ds, batch_size=batch_size, shuffle=shuffle)
    return dl, ds


def summarize_columns(csv_path: str, label_column: Optional[str] = None, drop_columns: Optional[List[str]] = None) -> Dict[str, object]:
    """Return a summary of columns kept/dropped after numeric preprocessing rules."""
    df = pd.read_csv(csv_path)
    label_present = label_column in df.columns if label_column else False
    if label_present:
        df = df.drop(columns=[label_column])
    drop_cols_used = []
    if drop_columns:
        drop_cols_used = [c for c in drop_columns if c in df.columns]
        if drop_cols_used:
            df = df.drop(columns=drop_cols_used)
    df_numeric = df.apply(pd.to_numeric, errors='coerce')
    kept_cols = df_numeric.dropna(axis=1, how='all').columns.tolist()
    dropped_non_numeric = [c for c in df.columns if c not in kept_cols]
    return {
        'label_removed': label_present,
        'drop_columns_applied': drop_cols_used,
        'all_columns_after_label_drop': list(df.columns),
        'kept_numeric_columns': kept_cols,
        'dropped_due_to_non_numeric': dropped_non_numeric,
        'num_kept': len(kept_cols),
        'num_dropped_non_numeric': len(dropped_non_numeric),
    }
