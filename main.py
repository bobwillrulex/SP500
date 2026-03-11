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
from sp500_ai.dqn import DQNConfig, predict_dqn_action, train_dqn
from sp500_ai.predict import predict_next_close
from sp500_ai.train import train_once
from sp500_ai.yahoo import fetch_sp500_history


def ask(prompt: str, default: str) -> str:
    raw = input(f"{prompt} [{default}]: ").strip()
    return raw or default


def ask_int(prompt: str, default: int) -> int:
    return int(ask(prompt, str(default)))


def ask_float(prompt: str, default: float) -> float:
    return float(ask(prompt, str(default)))


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


def mode_menu() -> None:
    print("\nS&P 500 AI Console")
    print("1) Forecast pipeline")
    print("2) DQN trading bot")
    print("3) Exit")


def forecast_menu() -> None:
    print("\nForecast pipeline")
    print("1) Download latest Yahoo historical data (^GSPC)")
    print("2) Train model")
    print("3) Predict next close")
    print("4) Start continuous training")
    print("5) Back")


def dqn_menu() -> None:
    print("\nDQN trading bot (BUY / SELL / HOLD)")
    print("1) Download latest Yahoo historical data (^GSPC)")
    print("2) Train DQN")
    print("3) Predict latest action")
    print("4) Back")


def build_train_config() -> TrainConfig:
    cfg = TrainConfig()
    cfg.seq_len = ask_int("Sequence length", cfg.seq_len)
    cfg.batch_size = ask_int("Batch size", cfg.batch_size)
    cfg.epochs = ask_int("Epochs", cfg.epochs)
    cfg.hidden_dim = ask_int("Hidden dimension", cfg.hidden_dim)
    cfg.n_layers = ask_int("Transformer layers", cfg.n_layers)
    cfg.n_heads = ask_int("Attention heads", cfg.n_heads)
    cfg.dropout = ask_float("Dropout (overfit guard)", cfg.dropout)
    cfg.lr = ask_float("Learning rate", cfg.lr)
    cfg.weight_decay = ask_float("Weight decay (regularization)", cfg.weight_decay)
    cfg.grad_clip = ask_float("Gradient clip", cfg.grad_clip)
    return cfg


def build_dqn_config() -> DQNConfig:
    cfg = DQNConfig()
    cfg.seq_len = ask_int("State sequence length", cfg.seq_len)
    cfg.episodes = ask_int("Episodes", cfg.episodes)
    cfg.batch_size = ask_int("Batch size", cfg.batch_size)
    cfg.replay_size = ask_int("Replay buffer size", cfg.replay_size)
    cfg.warmup_steps = ask_int("Warmup steps before learning", cfg.warmup_steps)
    cfg.hidden_dim = ask_int("Network hidden dimension", cfg.hidden_dim)
    cfg.dropout = ask_float("Dropout (overfit guard)", cfg.dropout)
    cfg.lr = ask_float("Learning rate", cfg.lr)
    cfg.weight_decay = ask_float("Weight decay", cfg.weight_decay)
    cfg.gamma = ask_float("Discount factor gamma", cfg.gamma)
    cfg.tau = ask_float("Soft target update tau", cfg.tau)
    cfg.target_update_interval = ask_int("Target update interval (steps)", cfg.target_update_interval)
    cfg.epsilon_start = ask_float("Epsilon start", cfg.epsilon_start)
    cfg.epsilon_end = ask_float("Epsilon end", cfg.epsilon_end)
    cfg.epsilon_decay_steps = ask_int("Epsilon decay steps", cfg.epsilon_decay_steps)
    cfg.transaction_cost = ask_float("Transaction cost penalty", cfg.transaction_cost)
    cfg.hold_penalty = ask_float("Position holding penalty", cfg.hold_penalty)
    cfg.overtrade_penalty = ask_float("Over-trade penalty", cfg.overtrade_penalty)
    cfg.reward_scale = ask_float("Reward scale", cfg.reward_scale)
    cfg.checkpoint_interval = ask_int("Checkpoint save interval (episodes)", cfg.checkpoint_interval)
    cfg.eval_interval = ask_int("Evaluation interval (episodes)", cfg.eval_interval)
    cfg.grad_clip = ask_float("Gradient clip", cfg.grad_clip)
    return cfg


def main() -> None:
    while True:
        mode_menu()
        choice = input("Choose an option (1-3): ").strip()

        if choice == "1":
            while True:
                forecast_menu()
                f_choice = input("Choose an option (1-5): ").strip()

                if f_choice == "1":
                    output_path = ask("Output CSV path", "data/sp500.csv")
                    db_path = ask("SQLite DB path (leave blank to skip DB)", "data/sp500.db")
                    period = ask("Yahoo period (e.g., 1y, 5y, max)", "max")
                    fetch_sp500_history(output_path=output_path, period=period, db_path=db_path)
                    print(f"Saved normalized OHLCV data to {output_path}")
                elif f_choice == "2":
                    data_path = ask("Input CSV path", "data/sp500.csv")
                    output_dir = ask("Output artifact directory", "artifacts/run1")
                    train_once(data_path, output_dir, build_train_config())
                    print(f"Training artifacts saved in {output_dir}")
                elif f_choice == "3":
                    data_path = ask("Input CSV path", "data/sp500.csv")
                    model_path = ask("Model path", "artifacts/run1/best_model.pt")
                    scaler_path = ask("Scaler path", "artifacts/run1/scaler.pkl")
                    meta_path = ask("Meta path", "artifacts/run1/meta.json")
                    pred = predict_next_close(data_path, model_path, scaler_path, meta_path)
                    print(f"Predicted next close: {pred:.4f}")
                elif f_choice == "4":
                    data_path = ask("Input CSV path", "data/sp500.csv")
                    output_dir = ask("Output base directory", "artifacts/live")
                    interval_seconds = ask_int("Interval seconds", 3600)
                    print("Starting continuous training (Ctrl+C to stop)...")
                    run_continuous_training(data_path, output_dir, interval_seconds)
                elif f_choice == "5":
                    break
                else:
                    print("Invalid option. Please choose 1-5.")
        elif choice == "2":
            while True:
                dqn_menu()
                d_choice = input("Choose an option (1-4): ").strip()

                if d_choice == "1":
                    output_path = ask("Output CSV path", "data/sp500.csv")
                    db_path = ask("SQLite DB path (leave blank to skip DB)", "data/sp500.db")
                    period = ask("Yahoo period (e.g., 1y, 5y, max)", "max")
                    fetch_sp500_history(output_path=output_path, period=period, db_path=db_path)
                    print(f"Saved normalized OHLCV data to {output_path}")
                elif d_choice == "2":
                    data_path = ask("Input CSV path", "data/sp500.csv")
                    output_dir = ask("DQN output artifact directory", "artifacts/dqn_run1")
                    train_dqn(data_path, output_dir, build_dqn_config())
                    print(f"DQN artifacts saved in {output_dir}")
                elif d_choice == "3":
                    data_path = ask("Input CSV path", "data/sp500.csv")
                    model_path = ask("DQN model path", "artifacts/dqn_run1/best_dqn_policy.pt")
                    scaler_path = ask("DQN scaler path", "artifacts/dqn_run1/dqn_scaler.pkl")
                    meta_path = ask("DQN meta path", "artifacts/dqn_run1/dqn_meta.json")
                    signal = predict_dqn_action(data_path, model_path, scaler_path, meta_path)
                    print(f"DQN action signal for next daily candle: {signal}")
                elif d_choice == "4":
                    break
                else:
                    print("Invalid option. Please choose 1-4.")
        elif choice == "3":
            print("Bye!")
            return
        else:
            print("Invalid option. Please choose 1-3.")


if __name__ == "__main__":
    main()
