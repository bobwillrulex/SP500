from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

from .config import REQUIRED_COLUMNS
from .features import build_features


@dataclass
class PreparedData:
    x_train: torch.Tensor
    y_train: torch.Tensor
    x_val: torch.Tensor
    y_val: torch.Tensor
    latest_window: torch.Tensor
    scaler: StandardScaler
    feature_columns: list[str]


def load_ohlcv_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df


def _to_sequences(x: np.ndarray, y: np.ndarray, seq_len: int):
    xs, ys = [], []
    for i in range(seq_len, len(x)):
        xs.append(x[i - seq_len : i])
        ys.append(y[i])
    return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.float32)


def prepare_data(df: pd.DataFrame, seq_len: int, val_ratio: float) -> PreparedData:
    feats = build_features(df)
    aligned = df.loc[feats.index].copy()

    target = aligned["close"].shift(-1)
    feats = feats.iloc[:-1]
    target = target.iloc[:-1]

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(feats.values)
    y = target.values.astype(np.float32)

    x_seq, y_seq = _to_sequences(x_scaled, y, seq_len)
    if len(x_seq) < 10:
        raise ValueError("Not enough data after feature engineering to create training sequences.")

    split_idx = int(len(x_seq) * (1 - val_ratio))
    x_train, y_train = x_seq[:split_idx], y_seq[:split_idx]
    x_val, y_val = x_seq[split_idx:], y_seq[split_idx:]

    latest_window = torch.tensor(x_seq[-1], dtype=torch.float32).unsqueeze(0)

    return PreparedData(
        x_train=torch.tensor(x_train),
        y_train=torch.tensor(y_train).unsqueeze(-1),
        x_val=torch.tensor(x_val),
        y_val=torch.tensor(y_val).unsqueeze(-1),
        latest_window=latest_window,
        scaler=scaler,
        feature_columns=list(feats.columns),
    )
