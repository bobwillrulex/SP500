# S&P 500 Advanced Forecasting Pipeline

This repository provides deterministic, GPU-ready deep learning pipelines for:
- **next-day S&P 500 close forecasting**,
- **DQN-based daily BUY/SELL/HOLD signal generation**.

## Important reality check
No model can be "extremely accurate" every day on financial markets. This project is designed to be:
- technically advanced,
- reproducible (same seeds => same results),
- continuously trainable,
- extensible with technical indicators and structure features (support/resistance + zones).

## Features
- Deterministic training configuration (`seed`, deterministic cuDNN/torch settings).
- PyTorch hybrid model: temporal convolution + transformer encoder.
- Feature engineering:
  - returns/volatility,
  - RSI, MACD, ATR, ADX,
  - Bollinger and moving-average structure,
  - support/resistance distances,
  - demand/supply zone distances and strengths.
- Continuous training loop (24/7) polling for fresh historical data.
- Walk-forward style validation split.
- Yahoo Finance downloader for `^GSPC` with normalized OHLCV output.
- Optional SQLite persistence for downloaded OHLCV history.
- Interactive console menu (`py main.py` / `python main.py`) for fetching data, training, predicting, and continuous training.
- Separate DQN trading agent with dueling architecture, prioritized replay, Double-DQN target logic, and configurable regularization/exploration controls.
- DQN crash-recovery checkpoints saved every configurable N episodes (default: every 50).

## Expected data format
CSV with at least:

- `date`
- `open`
- `high`
- `low`
- `close`
- `volume`

Example:
```csv
date,open,high,low,close,volume
2020-01-02,3244.67,3258.14,3235.53,3257.85,3459930000
```

## Python libraries to install (pip)
Minimum required libraries:
- `numpy`
- `pandas`
- `scikit-learn`
- `torch`
- `joblib`
- `yfinance`

Install everything from this repo:
```bash
pip install -r requirements.txt
```

If you want CUDA-enabled PyTorch for an RTX 3070, install torch from the official PyTorch index for your CUDA version (example for CUDA 12.1):
```bash
pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio
```
Then install the remaining packages:
```bash
pip install numpy pandas scikit-learn joblib yfinance
```

## Quickstart
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 1) download S&P 500 history from Yahoo and print latest OHLCV row for verification
python -c "from sp500_ai.yahoo import fetch_sp500_history; fetch_sp500_history('data/sp500.csv', period='max')"

# 2) train once
python -m sp500_ai.train --data data/sp500.csv --output artifacts/run1

# 3) predict next-day close
python -m sp500_ai.predict --data data/sp500.csv --model artifacts/run1/best_model.pt --scaler artifacts/run1/scaler.pkl

# 4) optional continuous training daemon
python -m sp500_ai.continuous_train --data data/sp500.csv --output artifacts/live --interval-seconds 3600
```

## Interactive console GUI
Run:
```bash
py main.py
```
(or `python main.py` on Linux/macOS)

If you run `py main.py` directly from the repository checkout, `main.py` now auto-adds the local `src/` folder to `PYTHONPATH` so `sp500_ai` imports resolve without extra setup.

Top-level menu now asks you to choose a mode first:
1. Forecast pipeline
2. DQN trading bot
3. Exit

The forecast submenu keeps download/train/predict/continuous options.

The DQN submenu includes:
1. Download latest Yahoo historical data (`^GSPC`)
2. Train DQN
3. Predict latest action (BUY/SELL/HOLD)
4. Back

Both training flows expose important hyperparameters interactively (dropout, weight decay, sequence length, learning rate, etc.) so you can tune for overfitting/underfitting behavior.

For DQN training, key robustness knobs include replay buffer size, epsilon schedule, target update cadence, penalties for transaction/over-trading, reward scale, and checkpoint interval.

CLI DQN training is also available:
```bash
python -m sp500_ai.dqn --data data/sp500.csv --output artifacts/dqn_run1
```

When downloading from Yahoo, the app prints the **latest OHLCV row** to the console so you can confirm the columns are correctly matched.

You can also provide a SQLite path (default: `data/sp500.db`) and the same OHLCV rows will be upserted into table `sp500_ohlcv`.

## Reproducibility notes
Reproducibility is strongest on the same hardware + same software versions.
This project sets deterministic modes and seeds, but tiny GPU kernel differences across environments can still happen.
