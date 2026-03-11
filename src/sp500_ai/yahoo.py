from __future__ import annotations

from pathlib import Path

import pandas as pd
import yfinance as yf


def fetch_sp500_history(output_path: str, symbol: str = "^GSPC", period: str = "max") -> pd.DataFrame:
    """Download Yahoo Finance OHLCV history and store as normalized CSV.

    The output CSV columns are: date, open, high, low, close, volume.
    """
    data = yf.download(symbol, period=period, auto_adjust=False, progress=False)
    if data.empty:
        raise ValueError(f"No data returned from Yahoo Finance for symbol={symbol}")

    data = data.reset_index()
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = [c[0].lower().replace(" ", "_") if isinstance(c, tuple) else str(c).lower().replace(" ", "_") for c in data.columns]
    else:
        data.columns = [str(c).lower().replace(" ", "_") for c in data.columns]

    required = ["date", "open", "high", "low", "close", "volume"]
    missing = [c for c in required if c not in data.columns]
    if missing:
        raise ValueError(f"Downloaded data missing required columns: {missing}")

    normalized = data[required].copy()
    normalized["date"] = pd.to_datetime(normalized["date"]).dt.tz_localize(None)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    normalized.to_csv(out, index=False)

    latest = normalized.iloc[-1]
    print(
        "Latest Yahoo row used (OHLCV): "
        f"date={latest['date'].date()} open={latest['open']:.2f} high={latest['high']:.2f} "
        f"low={latest['low']:.2f} close={latest['close']:.2f} volume={int(latest['volume'])}"
    )
    return normalized
