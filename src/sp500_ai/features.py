from __future__ import annotations

import numpy as np
import pandas as pd


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    roll_up = up.ewm(alpha=1 / period, adjust=False).mean()
    roll_down = down.ewm(alpha=1 / period, adjust=False).mean()
    rs = roll_up / (roll_down + 1e-9)
    return 100 - (100 / (1 + rs))


def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [
            (df["high"] - df["low"]).abs(),
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def _adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    up_move = df["high"].diff()
    down_move = -df["low"].diff()

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    atr = _atr(df, period)
    plus_di = 100 * pd.Series(plus_dm, index=df.index).ewm(alpha=1 / period, adjust=False).mean() / (atr + 1e-9)
    minus_di = 100 * pd.Series(minus_dm, index=df.index).ewm(alpha=1 / period, adjust=False).mean() / (atr + 1e-9)

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-9)
    return dx.ewm(alpha=1 / period, adjust=False).mean()


def _support_resistance_features(df: pd.DataFrame, lookback: int = 40) -> pd.DataFrame:
    roll_min = df["low"].rolling(lookback).min()
    roll_max = df["high"].rolling(lookback).max()
    close = df["close"]

    out = pd.DataFrame(index=df.index)
    out["dist_to_support"] = (close - roll_min) / (close + 1e-9)
    out["dist_to_resistance"] = (roll_max - close) / (close + 1e-9)
    out["range_position"] = (close - roll_min) / (roll_max - roll_min + 1e-9)
    return out


def _zone_features(df: pd.DataFrame, lookback: int = 30) -> pd.DataFrame:
    avg_vol = df["volume"].rolling(lookback).mean()
    candle_body = (df["close"] - df["open"]).abs()
    candle_range = (df["high"] - df["low"]).abs() + 1e-9

    base_score = (1 - candle_body / candle_range).clip(0, 1)
    volume_score = (df["volume"] / (avg_vol + 1e-9)).clip(0, 3)

    demand_level = df["low"].rolling(lookback).quantile(0.2)
    supply_level = df["high"].rolling(lookback).quantile(0.8)

    out = pd.DataFrame(index=df.index)
    out["dist_to_demand_zone"] = (df["close"] - demand_level) / (df["close"] + 1e-9)
    out["dist_to_supply_zone"] = (supply_level - df["close"]) / (df["close"] + 1e-9)
    out["zone_strength"] = (base_score * volume_score).fillna(0.0)
    return out


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    close = df["close"]

    feat = pd.DataFrame(index=df.index)
    feat["ret_1"] = close.pct_change(1)
    feat["ret_5"] = close.pct_change(5)
    feat["ret_10"] = close.pct_change(10)
    feat["vol_10"] = feat["ret_1"].rolling(10).std()
    feat["vol_20"] = feat["ret_1"].rolling(20).std()

    feat["sma_10_ratio"] = close / (close.rolling(10).mean() + 1e-9)
    feat["sma_20_ratio"] = close / (close.rolling(20).mean() + 1e-9)
    feat["sma_50_ratio"] = close / (close.rolling(50).mean() + 1e-9)

    ema_fast = _ema(close, 12)
    ema_slow = _ema(close, 26)
    macd = ema_fast - ema_slow
    signal = _ema(macd, 9)
    feat["macd"] = macd
    feat["macd_signal"] = signal
    feat["macd_hist"] = macd - signal

    feat["rsi_14"] = _rsi(close, 14)
    feat["atr_14"] = _atr(df, 14) / (close + 1e-9)
    feat["adx_14"] = _adx(df, 14)

    std_20 = close.rolling(20).std()
    bb_mid = close.rolling(20).mean()
    feat["bb_width"] = (2 * std_20) / (bb_mid + 1e-9)
    feat["bb_pos"] = (close - (bb_mid - 2 * std_20)) / (4 * std_20 + 1e-9)

    feat = pd.concat(
        [
            feat,
            _support_resistance_features(df),
            _zone_features(df),
        ],
        axis=1,
    )

    feat = feat.replace([np.inf, -np.inf], np.nan).dropna()
    return feat
