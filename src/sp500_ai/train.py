from __future__ import annotations

import argparse
import json
import os
import random

import joblib
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .config import TrainConfig
from .data import load_ohlcv_csv, prepare_data
from .model import PriceForecaster


def set_deterministic(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def evaluate(model, loader, criterion, device):
    model.eval()
    losses = []
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            losses.append(criterion(pred, yb).item())
    return float(np.mean(losses))


def train_once(data_path: str, output_dir: str, cfg: TrainConfig) -> None:
    os.makedirs(output_dir, exist_ok=True)
    set_deterministic(cfg.seed)

    df = load_ohlcv_csv(data_path)
    prepared = prepare_data(df, cfg.seq_len, cfg.val_ratio)

    train_ds = TensorDataset(prepared.x_train, prepared.y_train)
    val_ds = TensorDataset(prepared.x_val, prepared.y_val)

    generator = torch.Generator().manual_seed(cfg.seed)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, generator=generator)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PriceForecaster(
        n_features=prepared.x_train.shape[-1],
        hidden_dim=cfg.hidden_dim,
        n_heads=cfg.n_heads,
        n_layers=cfg.n_layers,
        dropout=cfg.dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=4, factor=0.5)
    criterion = nn.HuberLoss(delta=2.0)

    best_val = float("inf")
    patience = 10
    stale = 0

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        epoch_losses = []

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            epoch_losses.append(loss.item())

        val_loss = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_loss)
        train_loss = float(np.mean(epoch_losses))

        print(f"epoch={epoch:03d} train={train_loss:.6f} val={val_loss:.6f}")

        if val_loss < best_val:
            best_val = val_loss
            stale = 0
            torch.save(model.state_dict(), os.path.join(output_dir, "best_model.pt"))
        else:
            stale += 1
            if stale >= patience:
                print("Early stopping triggered.")
                break

    joblib.dump(prepared.scaler, os.path.join(output_dir, "scaler.pkl"))

    meta = {
        "feature_columns": prepared.feature_columns,
        "config": cfg.__dict__,
        "best_val_huber": best_val,
        "device": str(device),
    }
    with open(os.path.join(output_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to OHLCV CSV")
    parser.add_argument("--output", required=True, help="Output artifact directory")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_once(args.data, args.output, TrainConfig())
