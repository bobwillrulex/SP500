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
    p.add_argument("--target-scaler", default=None)
    return p.parse_args()


def predict_next_close(
    data_path: str,
    model_path: str,
    scaler_path: str,
    meta_path: str | None = None,
    target_scaler_path: str | None = None,
) -> float:
    resolved_meta_path = meta_path or model_path.replace("best_model.pt", "meta.json")
    resolved_target_scaler_path = target_scaler_path or model_path.replace("best_model.pt", "target_scaler.pkl")
    with open(resolved_meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    cfg = meta["config"]

    df = load_ohlcv_csv(data_path)
    prepared = prepare_data(df, cfg["seq_len"], cfg["val_ratio"])

    _ = joblib.load(scaler_path)
    try:
        target_scaler = joblib.load(resolved_target_scaler_path)
    except FileNotFoundError:
        target_scaler = None

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
        normalized_pred = model(prepared.latest_window).item()

    if target_scaler is None:
        return float(normalized_pred)

    return float(target_scaler.inverse_transform([[normalized_pred]])[0, 0])


def main():
    args = parse_args()
    pred = predict_next_close(args.data, args.model, args.scaler, args.meta, args.target_scaler)
    print(f"Predicted next close: {pred:.4f}")


if __name__ == "__main__":
    main()
