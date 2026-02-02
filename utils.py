from __future__ import annotations
import datetime as dt
import os
import random
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DataConfig:
    symbol: str = "XAUUSD"
    timeframe: str = "M5"  # M1, M5, M15, H1, H4, D1, ...
    start_year: int = 2006
    end_year: Optional[int] = None
    source: str = "ctrader"  # "ctrader" or "dukascopy"
    data_dir: Path = Path("data")
    cache_dir: Path = Path("data/cache")
    output_dir: Path = Path("data/outputs")
    price_scale: int = 1000  # لتحويل الأسعار إلى نقاط (pipettes)
    max_workers: int = 4
    fill_method: str = "ffill"
    timezone: str = "UTC"

    def __post_init__(self):
        valid_sources = {"ctrader", "dukascopy"}
        if self.source not in valid_sources:
            raise ValueError(f"Invalid source: {self.source}. Must be one of {valid_sources}")


@dataclass(frozen=True)
class FeatureConfig:
    horizons: Sequence[int] = (1, 3, 6, 12, 24)
    rolling_windows: Sequence[int] = (5, 10, 20, 30, 50, 100)
    lag_windows: Sequence[int] = tuple(range(1, 31))
    volatility_windows: Sequence[int] = (5, 10, 20, 50)
    atr_window: int = 14
    regime_window: int = 100
    regime_quantiles: Sequence[float] = (0.2, 0.8)

    def __post_init__(self):
        if any(h <= 0 for h in self.horizons):
            raise ValueError("Horizons must be positive integers")
        if any(w <= 0 for w in self.rolling_windows):
            raise ValueError("Rolling windows must be positive")


@dataclass(frozen=True)
class TrainConfig:
    horizon: int = 3
    test_splits: int = 5
    threshold: float = 0.0
    seed: int = 7
    rolling_window: int = 4000
    expanding_window: int = 4000
    optuna_trials: int = 30
    optuna_timeout: int = 600
    ensemble_top_k: int = 2


@dataclass(frozen=True)
class CVConfig:
    test_splits: int
    rolling_window: int
    expanding_window: int


def ensure_dir(path: Path) -> None:
    """Create directory and parents if not exist"""
    path.mkdir(parents=True, exist_ok=True)


def get_date_range(start_year: int, end_year: Optional[int] = None) -> List[dt.date]:
    """Generate list of dates from start_year to end_year (or current year)"""
    if end_year is None:
        end_year = dt.date.today().year
    if end_year < start_year:
        raise ValueError(f"end_year ({end_year}) cannot be before start_year ({start_year})")
    
    start_date = dt.date(start_year, 1, 1)
    end_date = dt.date(end_year, 12, 31)
    delta = end_date - start_date
    return [start_date + dt.timedelta(days=i) for i in range(delta.days + 1)]


def to_parquet(df: pd.DataFrame, path: Path, compression: str = "snappy") -> None:
    """Save DataFrame to parquet with error handling"""
    ensure_dir(path.parent)
    try:
        df.to_parquet(path, index=False, compression=compression)
    except Exception as e:
        raise RuntimeError(f"Failed to save parquet: {path}\nError: {e}")


def to_csv(df: pd.DataFrame, path: Path) -> None:
    """Save DataFrame to CSV with error handling"""
    ensure_dir(path.parent)
    try:
        df.to_csv(path, index=False)
    except Exception as e:
        raise RuntimeError(f"Failed to save CSV: {path}\nError: {e}")


def load_cached(path: Path) -> Optional[pd.DataFrame]:
    """Load from cache (parquet or csv)"""
    if not path.exists():
        return None
    
    try:
        if path.suffix == ".parquet":
            return pd.read_parquet(path)
        elif path.suffix == ".csv":
            return pd.read_csv(path, parse_dates=["timestamp"])
        else:
            warnings.warn(f"Unsupported cache format: {path.suffix}")
            return None
    except Exception as e:
        warnings.warn(f"Failed to load cache {path}: {e}")
        return None


def set_deterministic(seed: int) -> None:
    """Set all random seeds for reproducibility"""
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    # إضافة لـ PyTorch إذا كنت تستخدمه لاحقًا
    # import torch
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False


def chunked(iterable: Sequence, size: int) -> Iterable[List]:
    """Split sequence into chunks"""
    for idx in range(0, len(iterable), size):
        yield list(iterable[idx : idx + size])


def safe_to_datetime(series: pd.Series, tz: str = "UTC") -> pd.Series:
    """Safely convert to datetime with timezone handling"""
    timestamp = pd.to_datetime(series, utc=True, errors="coerce")
    if tz != "UTC":
        timestamp = timestamp.dt.tz_convert(tz)
    return timestamp


def compute_drawdown(equity_curve: pd.Series | np.ndarray) -> pd.Series:
    """Compute drawdown series from equity curve"""
    if isinstance(equity_curve, np.ndarray):
        equity_curve = pd.Series(equity_curve)
    running_max = equity_curve.cummax()
    drawdown = (equity_curve / running_max) - 1
    return drawdown


def validate_dataframe(df: pd.DataFrame, required_cols: List[str]) -> None:
    """Basic validation for input dataframe"""
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if df.empty:
        raise ValueError("DataFrame is empty")
    if df["timestamp"].isna().any():
        raise ValueError("NaN values found in timestamp column")
