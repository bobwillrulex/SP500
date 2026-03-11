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


def predict_next_close(data_path: str, model_path: str, scaler_path: str, meta_path: str | None = None) -> float:
    resolved_meta_path = meta_path or model_path.replace("best_model.pt", "meta.json")
    with open(resolved_meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    cfg = meta["config"]

    df = load_ohlcv_csv(data_path)
    prepared = prepare_data(df, cfg["seq_len"], cfg["val_ratio"])

    _ = joblib.load(scaler_path)

    model = PriceForecaster(
        n_features=prepared.latest_window.shape[-1],
        hidden_dim=cfg["hidden_dim"],
        n_heads=cfg["n_heads"],
        n_layers=cfg["n_layers"],
        dropout=cfg["dropout"],
    )
    state = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    with torch.no_grad():
        return model(prepared.latest_window).item()


def main():
    args = parse_args()
    pred = predict_next_close(args.data, args.model, args.scaler, args.meta)
    print(f"Predicted next close: {pred:.4f}")


if __name__ == "__main__":
    main()
