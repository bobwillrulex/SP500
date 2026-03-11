from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd
import yfinance as yf


def store_ohlcv_in_db(df: pd.DataFrame, db_path: str, table_name: str = "sp500_ohlcv") -> None:
    """Persist normalized OHLCV rows into SQLite."""
    db_file = Path(db_path)
    db_file.parent.mkdir(parents=True, exist_ok=True)

    db_df = df.copy()
    db_df["date"] = pd.to_datetime(db_df["date"]).dt.strftime("%Y-%m-%d")

    with sqlite3.connect(db_file) as conn:
        conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {table_name} (
                date TEXT PRIMARY KEY,
                open REAL NOT NULL,
                high REAL NOT NULL,
                low REAL NOT NULL,
                close REAL NOT NULL,
                volume REAL NOT NULL
            )
            """
        )
        conn.executemany(
            f"""
            INSERT INTO {table_name} (date, open, high, low, close, volume)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(date) DO UPDATE SET
                open=excluded.open,
                high=excluded.high,
                low=excluded.low,
                close=excluded.close,
                volume=excluded.volume
            """,
            list(db_df[["date", "open", "high", "low", "close", "volume"]].itertuples(index=False, name=None)),
        )


def fetch_sp500_history(
    output_path: str,
    symbol: str = "^GSPC",
    period: str = "max",
    db_path: str | None = None,
    table_name: str = "sp500_ohlcv",
) -> pd.DataFrame:
    """Download Yahoo Finance OHLCV history and store as normalized CSV.

    The output CSV columns are: date, open, high, low, close, volume.
    Optionally also writes the rows into a SQLite table.
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

    if db_path:
        store_ohlcv_in_db(normalized, db_path=db_path, table_name=table_name)

    latest = normalized.iloc[-1]
    print(
        "Latest Yahoo row used (OHLCV): "
        f"date={latest['date'].date()} open={latest['open']:.2f} high={latest['high']:.2f} "
        f"low={latest['low']:.2f} close={latest['close']:.2f} volume={int(latest['volume'])}"
    )
    if db_path:
        print(f"Stored {len(normalized)} rows in SQLite database: {db_path} (table={table_name})")
    return normalized
