from __future__ import annotations

import argparse
import json

import joblib
import torch

from .data import load_ohlcv_csv, prepare_data
from .model import PriceForecaster


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--scaler", required=True)
    p.add_argument("--meta", default=None)
    return p.parse_args()


def main():
    args = parse_args()

    meta_path = args.meta or args.model.replace("best_model.pt", "meta.json")
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    cfg = meta["config"]

    df = load_ohlcv_csv(args.data)
    prepared = prepare_data(df, cfg["seq_len"], cfg["val_ratio"])

    _ = joblib.load(args.scaler)

    model = PriceForecaster(
        n_features=prepared.latest_window.shape[-1],
        hidden_dim=cfg["hidden_dim"],
        n_heads=cfg["n_heads"],
        n_layers=cfg["n_layers"],
        dropout=cfg["dropout"],
    )
    state = torch.load(args.model, map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    with torch.no_grad():
        pred = model(prepared.latest_window).item()

    print(f"Predicted next close: {pred:.4f}")


if __name__ == "__main__":
    main()
