from __future__ import annotations
import numpy as np
import pandas as pd
from utils import FeatureConfig


def add_features(df: pd.DataFrame, config: FeatureConfig | None = None) -> pd.DataFrame:
    """
    إضافة كل الميزات الأساسية + المتقدمة من الاستراتيجيات:
    - Trend Following (EMA, MACD)
    - SMC (FVG, simple Order Blocks)
    - Breakout (Donchian + volatility filter)
    - Macro correlations (DXY, VIX, US10Y إذا متوفرة)
    """
    config = config or FeatureConfig()
    df = df.copy()

    # ── الميزات الأساسية (الأصلية) ──
    df["return"] = df["close"].pct_change()
    df["log_return"] = np.log(df["close"]).diff()
    _add_multi_horizon_returns(df, config.horizons)
    _add_microstructure_features(df)
    _add_rolling_statistics(df, config)
    _add_volatility_measures(df, config)
    _add_lag_features(df, config)
    _add_time_features(df)
    _add_regime_features(df, config)

    # ── الميزات المتقدمة من الاستراتيجيات ──
    _add_trend_following_features(df)
    _add_smc_features(df)
    _add_breakout_features(df)
    _add_macro_correlation_features(df)

    # تنظيف NaN الناتجة عن الـ rolling/shift
    return df.dropna().reset_index(drop=True)


# ────────────────────────────────────────────────────────────────
#               الدوال الأساسية (كما كانت)
# ────────────────────────────────────────────────────────────────

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
        df[f"return_zscore_{window}"] = (df["return"] - df[f"return_mean_{window}"]) / df[f"return_std_{window}"]
        df[f"return_skew_{window}"] = rolling_return.skew()
        df[f"return_kurt_{window}"] = rolling_return.kurt()

        df[f"volume_mean_{window}"] = df["volume"].rolling(window).mean()
        df[f"volume_std_{window}"] = df["volume"].rolling(window).std()
        df[f"volume_zscore_{window}"] = (df["volume"] - df[f"volume_mean_{window}"]) / df[f"volume_std_{window}"]


def _add_volatility_measures(df: pd.DataFrame, config: FeatureConfig) -> None:
    for window in config.volatility_windows:
        df[f"volatility_{window}"] = df["log_return"].rolling(window).std()
    df["atr"] = _average_true_range(df, window=config.atr_window)
    df["parkinson_vol"] = ((1 / (4 * np.log(2))) * (np.log(df["high"] / df["low"]) ** 2)).rolling(config.atr_window).mean().pow(0.5)


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


# ────────────────────────────────────────────────────────────────
#               الميزات المتقدمة من الاستراتيجيات
# ────────────────────────────────────────────────────────────────


def _add_trend_following_features(df: pd.DataFrame) -> None:
    """Trend Following + MACD"""
    df["ema_fast"] = df["close"].ewm(span=12, adjust=False).mean()
    df["ema_slow"] = df["close"].ewm(span=26, adjust=False).mean()
    df["macd"] = df["ema_fast"] - df["ema_slow"]
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]
    
    # EMA crossover direction
    df["trend_dir"] = np.where(df["ema_fast"] > df["ema_slow"], 1,
                              np.where(df["ema_fast"] < df["ema_slow"], -1, 0))


def _add_smc_features(df: pd.DataFrame) -> None:
    """Fair Value Gaps + simple Order Block proxy"""
    # Bullish/Bearish FVG
    df["bull_fvg"] = ((df["low"].shift(2) > df["high"]) & (df["close"] > df["open"])).astype(int)
    df["bear_fvg"] = ((df["high"].shift(2) < df["low"]) & (df["close"] < df["open"])).astype(int)
    
    # Order Block proxy (previous swing levels)
    df["swing_high"] = df["high"].rolling(20).max().shift(1)
    df["swing_low"]  = df["low"].rolling(20).min().shift(1)
    df["in_bull_ob"] = (df["close"] >= df["swing_low"]).astype(int)
    df["in_bear_ob"] = (df["close"] <= df["swing_high"]).astype(int)


def _add_breakout_features(df: pd.DataFrame) -> None:
    """Donchian Breakout + volatility filter"""
    period = 20
    df["donchian_high"] = df["high"].rolling(period).max().shift(1)
    df["donchian_low"]  = df["low"].rolling(period).min().shift(1)
    
    df["breakout_up"]   = (df["close"] > df["donchian_high"]).astype(int)
    df["breakout_down"] = (df["close"] < df["donchian_low"]).astype(int)
    
    # Volatility-adjusted (stronger breakout when vol > avg)
    atr = _average_true_range(df, 14)
    atr_avg = atr.rolling(50).mean()
    df["strong_break_up"]   = ((df["breakout_up"] == 1) & (atr > atr_avg)).astype(int)
    df["strong_break_down"] = ((df["breakout_down"] == 1) & (atr > atr_avg)).astype(int)


def _add_macro_correlation_features(df: pd.DataFrame) -> None:
    """Rolling correlation with macro assets (if columns exist)"""
    window = 60
    macro_pairs = [
        ("dxy_close", "gold_dxy_cor"),
        ("vix_close", "gold_vix_cor"),
        ("us10y_close", "gold_us10y_cor"),
        ("spx_close", "gold_spx_cor")
    ]
    
    for macro_col, cor_col in macro_pairs:
        if macro_col in df.columns:
            df[cor_col] = df["close"].rolling(window).corr(df[macro_col])
            df[f"{cor_col}_diff"] = df[cor_col].diff()  # تغير الارتباط
