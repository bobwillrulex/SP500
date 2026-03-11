# S&P 500 Advanced Forecasting Pipeline

This repository provides a deterministic, GPU-ready deep learning pipeline for **next-day S&P 500 close prediction**.

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
pip install numpy pandas scikit-learn joblib
```

## Quickstart
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# train once
python -m sp500_ai.train --data data/sp500.csv --output artifacts/run1

# predict next-day close
python -m sp500_ai.predict --data data/sp500.csv --model artifacts/run1/best_model.pt --scaler artifacts/run1/scaler.pkl

# continuous training daemon
python -m sp500_ai.continuous_train --data data/sp500.csv --output artifacts/live --interval-seconds 3600
```

## Reproducibility notes
Reproducibility is strongest on the same hardware + same software versions.
This project sets deterministic modes and seeds, but tiny GPU kernel differences across environments can still happen.
