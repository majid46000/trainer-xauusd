from __future__ import annotations

import numpy as np
import pandas as pd

from utils import FeatureConfig


def add_features(df: pd.DataFrame, config: FeatureConfig | None = None) -> pd.DataFrame:
    config = config or FeatureConfig()
    df = df.copy()

    df["return"] = df["close"].pct_change()
    df["log_return"] = np.log(df["close"]).diff()

    _add_multi_horizon_returns(df, config.horizons)
    _add_microstructure_features(df)
    _add_rolling_statistics(df, config)
    _add_volatility_measures(df, config)
    _add_lag_features(df, config)
    _add_time_features(df)
    _add_regime_features(df, config)

    return df


def _add_multi_horizon_returns(df: pd.DataFrame, horizons: tuple[int, ...] | list[int]) -> None:
    for horizon in horizons:
        df[f"return_{horizon}"] = df["close"].pct_change(horizon)
        df[f"log_return_{horizon}"] = np.log(df["close"]).diff(horizon)


def _add_microstructure_features(df: pd.DataFrame) -> None:
    candle_range = (df["high"] - df["low"]).replace(0, np.nan)
    body = (df["close"] - df["open"]).abs()
    upper_wick = df["high"] - df[["open", "close"]].max(axis=1)
    lower_wick = df[["open", "close"]].min(axis=1) - df["low"]

    df["range"] = candle_range
    df["body"] = body
    df["upper_wick"] = upper_wick
    df["lower_wick"] = lower_wick
    df["body_ratio"] = body / candle_range
    df["upper_wick_ratio"] = upper_wick / candle_range
    df["lower_wick_ratio"] = lower_wick / candle_range
    df["close_position"] = (df["close"] - df["low"]) / candle_range


def _add_rolling_statistics(df: pd.DataFrame, config: FeatureConfig) -> None:
    for window in config.rolling_windows:
        rolling_close = df["close"].rolling(window)
        df[f"rolling_mean_{window}"] = rolling_close.mean()
        df[f"rolling_std_{window}"] = rolling_close.std()
        df[f"rolling_min_{window}"] = rolling_close.min()
        df[f"rolling_max_{window}"] = rolling_close.max()

        rolling_return = df["return"].rolling(window)
        df[f"return_mean_{window}"] = rolling_return.mean()
        df[f"return_std_{window}"] = rolling_return.std()
        df[f"return_zscore_{window}"] = (
            df["return"] - df[f"return_mean_{window}"]
        ) / df[f"return_std_{window}"]
        df[f"return_skew_{window}"] = rolling_return.skew()
        df[f"return_kurt_{window}"] = rolling_return.kurt()

        df[f"volume_mean_{window}"] = df["volume"].rolling(window).mean()
        df[f"volume_std_{window}"] = df["volume"].rolling(window).std()
        df[f"volume_zscore_{window}"] = (
            df["volume"] - df[f"volume_mean_{window}"]
        ) / df[f"volume_std_{window}"]


def _add_volatility_measures(df: pd.DataFrame, config: FeatureConfig) -> None:
    for window in config.volatility_windows:
        df[f"volatility_{window}"] = df["log_return"].rolling(window).std()

    df["atr"] = _average_true_range(df, window=config.atr_window)
    df["parkinson_vol"] = (
        (1 / (4 * np.log(2))) * (np.log(df["high"] / df["low"]) ** 2)
    ).rolling(config.atr_window).mean().pow(0.5)


def _add_lag_features(df: pd.DataFrame, config: FeatureConfig) -> None:
    for lag in config.lag_windows:
        df[f"return_lag_{lag}"] = df["return"].shift(lag)
        df[f"log_return_lag_{lag}"] = df["log_return"].shift(lag)
        df[f"volume_lag_{lag}"] = df["volume"].shift(lag)


def _add_time_features(df: pd.DataFrame) -> None:
    timestamp = pd.to_datetime(df["timestamp"], utc=True)
    df["hour"] = timestamp.dt.hour
    df["minute"] = timestamp.dt.minute
    df["day_of_week"] = timestamp.dt.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    df["session_asia"] = ((df["hour"] >= 0) & (df["hour"] < 7)).astype(int)
    df["session_europe"] = ((df["hour"] >= 7) & (df["hour"] < 15)).astype(int)
    df["session_us"] = ((df["hour"] >= 13) & (df["hour"] < 21)).astype(int)


def _add_regime_features(df: pd.DataFrame, config: FeatureConfig) -> None:
    window = config.regime_window
    vol = df["log_return"].rolling(window).std()
    trend = df["log_return"].rolling(window).mean()

    vol_q_low = vol.rolling(window).quantile(config.regime_quantiles[0])
    vol_q_high = vol.rolling(window).quantile(config.regime_quantiles[1])
    trend_q_low = trend.rolling(window).quantile(config.regime_quantiles[0])
    trend_q_high = trend.rolling(window).quantile(config.regime_quantiles[1])

    df["vol_regime"] = np.select(
        [vol < vol_q_low, vol > vol_q_high],
        [0, 2],
        default=1,
    )
    df["trend_regime"] = np.select(
        [trend < trend_q_low, trend > trend_q_high],
        [0, 2],
        default=1,
    )


def _average_true_range(df: pd.DataFrame, window: int = 14) -> pd.Series:
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift()).abs()
    low_close = (df["low"] - df["close"].shift()).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return true_range.rolling(window).mean()
