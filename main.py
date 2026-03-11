from __future__ import annotations

import datetime as dt
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sp500_ai.config import TrainConfig
from sp500_ai.predict import predict_next_close
from sp500_ai.train import train_once
from sp500_ai.yahoo import fetch_sp500_history


def ask(prompt: str, default: str) -> str:
    raw = input(f"{prompt} [{default}]: ").strip()
    return raw or default


def run_continuous_training(data_path: str, output_dir: str, interval_seconds: int) -> None:
    os.makedirs(output_dir, exist_ok=True)

    while True:
        ts = dt.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join(output_dir, f"run_{ts}")
        print(f"[{dt.datetime.utcnow().isoformat()}] starting training -> {run_dir}")

        try:
            train_once(data_path, run_dir, TrainConfig())
            latest = os.path.join(output_dir, "latest")
            if os.path.islink(latest) or os.path.exists(latest):
                try:
                    os.remove(latest)
                except OSError:
                    pass
            os.symlink(run_dir, latest)
            print(f"[{dt.datetime.utcnow().isoformat()}] training finished")
        except Exception as exc:
            print(f"[{dt.datetime.utcnow().isoformat()}] training failed: {exc}")

        print(f"Sleeping {interval_seconds} seconds before next run...")
        time.sleep(interval_seconds)


def menu() -> None:
    print("\nS&P 500 AI Console")
    print("1) Download latest Yahoo historical data (^GSPC)")
    print("2) Train model")
    print("3) Predict next close")
    print("4) Start continuous training")
    print("5) Exit")


def main() -> None:
    while True:
        menu()
        choice = input("Choose an option (1-5): ").strip()

        if choice == "1":
            output_path = ask("Output CSV path", "data/sp500.csv")
            db_path = ask("SQLite DB path (leave blank to skip DB)", "data/sp500.db")
            period = ask("Yahoo period (e.g., 1y, 5y, max)", "max")
            fetch_sp500_history(output_path=output_path, period=period, db_path=db_path)
            print(f"Saved normalized OHLCV data to {output_path}")
        elif choice == "2":
            data_path = ask("Input CSV path", "data/sp500.csv")
            output_dir = ask("Output artifact directory", "artifacts/run1")
            train_once(data_path, output_dir, TrainConfig())
            print(f"Training artifacts saved in {output_dir}")
        elif choice == "3":
            data_path = ask("Input CSV path", "data/sp500.csv")
            model_path = ask("Model path", "artifacts/run1/best_model.pt")
            scaler_path = ask("Scaler path", "artifacts/run1/scaler.pkl")
            meta_path = ask("Meta path", "artifacts/run1/meta.json")
            pred = predict_next_close(data_path, model_path, scaler_path, meta_path)
            print(f"Predicted next close: {pred:.4f}")
        elif choice == "4":
            data_path = ask("Input CSV path", "data/sp500.csv")
            output_dir = ask("Output base directory", "artifacts/live")
            interval_seconds = int(ask("Interval seconds", "3600"))
            print("Starting continuous training (Ctrl+C to stop)...")
            run_continuous_training(data_path, output_dir, interval_seconds)
        elif choice == "5":
            print("Bye!")
            return
        else:
            print("Invalid option. Please choose 1-5.")


if __name__ == "__main__":
    main()
