import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from typing import Tuple


def make_tensors(df: pd.DataFrame) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convert a DataFrame split into PyTorch tensors.
    X shape: (N, 1, 2, n_bins) — N loci, channel (K9 only), strains (A4/A7), bins with K9 count
    y shape: (N, 2)            — label_A4, label_A7
    """
    A4 = np.stack(df["A4_K9"].values)
    A7 = np.stack(df["A7_K9"].values)
    X  = torch.FloatTensor(np.stack([A4, A7], axis=1)[:, np.newaxis, :, :])
    y  = torch.FloatTensor(df[["label_A4", "label_A7"]].values.astype(np.float32))
    return X, y

def make_splits(
    full_df:       pd.DataFrame,
    val_fraction:  float = 0.15,
    test_fraction: float = 0.15,
    seed:          int   = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split full_df into train, val, test with stratification by label pair.
    Returns train_df, val_df, test_df.
    """
    strat = [f"{int(a)}{int(b)}" for a, b in full_df[["label_A4", "label_A7"]].values]

    trainval_df, test_df = train_test_split(
        full_df, test_size=test_fraction, stratify=strat, random_state=seed
    )
    strat_tv = [f"{int(a)}{int(b)}" for a, b in trainval_df[["label_A4", "label_A7"]].values]

    train_df, val_df = train_test_split(
        trainval_df, test_size=val_fraction / (1 - test_fraction),
        stratify=strat_tv, random_state=seed
    )
    return train_df, val_df, test_df
