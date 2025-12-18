import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Tuple

class CSVDataset(Dataset):
    def __init__(self, csv_path: str, label_column: Optional[str] = None, normalize: bool = True):
        df = pd.read_csv(csv_path)
        if label_column and label_column in df.columns:
            self.labels = df[label_column].values
            features = df.drop(columns=[label_column])
        else:
            self.labels = None
            features = df
        self.feature_cols = list(features.columns)
        X = features.values.astype(np.float32)
        if normalize:
            # Standardize per column: (x - mean) / std (avoid div by zero)
            mean = X.mean(axis=0)
            std = X.std(axis=0)
            std[std == 0] = 1.0
            X = (X - mean) / std
        self.X = torch.from_numpy(X)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx]


def make_dataloader(csv_path: str, batch_size: int = 64, shuffle: bool = True, label_column: Optional[str] = None) -> Tuple[DataLoader, CSVDataset]:
    ds = CSVDataset(csv_path, label_column=label_column)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=shuffle)
    return dl, ds
