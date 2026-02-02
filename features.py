from __future__ import annotations

import numpy as np
import pandas as pd


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["return"] = df["close"].pct_change()
    df["log_return"] = np.log(df["close"]).diff()

    df["range"] = df["high"] - df["low"]
    df["body"] = (df["close"] - df["open"]).abs()
    df["upper_wick"] = df["high"] - df[["open", "close"]].max(axis=1)
    df["lower_wick"] = df[["open", "close"]].min(axis=1) - df["low"]

    df["rolling_mean_10"] = df["close"].rolling(10).mean()
    df["rolling_std_10"] = df["close"].rolling(10).std()
    df["rolling_mean_30"] = df["close"].rolling(30).mean()
    df["rolling_std_30"] = df["close"].rolling(30).std()

    df["atr_14"] = _average_true_range(df, window=14)
    df["rsi_14"] = _rsi(df["close"], window=14)

    for window in [8, 21, 55, 100]:
        df[f"ema_{window}"] = df["close"].ewm(span=window, adjust=False).mean()

    for window in [5, 10, 20]:
        df[f"volatility_{window}"] = df["return"].rolling(window).std()
        df[f"momentum_{window}"] = df["close"].pct_change(window)

    df["volume_mean_20"] = df["volume"].rolling(20).mean()
    df["volume_std_20"] = df["volume"].rolling(20).std()
    df["volume_zscore_20"] = (df["volume"] - df["volume_mean_20"]) / df["volume_std_20"]

    return df


def _average_true_range(df: pd.DataFrame, window: int = 14) -> pd.Series:
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift()).abs()
    low_close = (df["low"] - df["close"].shift()).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return true_range.rolling(window).mean()


def _rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))
