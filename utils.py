from __future__ import annotations

import datetime as dt
import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DataConfig:
    symbol: str = "XAUUSD"
    timeframe: str = "M5"
    start_year: int = 2006
    end_year: Optional[int] = None
    source: str = "ctrader"  # ctrader or dukascopy
    data_dir: Path = Path("data")
    cache_dir: Path = Path("data/cache")
    output_dir: Path = Path("data/outputs")
    price_scale: int = 1000
    max_workers: int = 4
    fill_method: str = "ffill"
    timezone: str = "UTC"


@dataclass(frozen=True)
class FeatureConfig:
    horizons: Sequence[int] = (1, 3, 6, 12, 24)
    rolling_windows: Sequence[int] = (5, 10, 20, 30, 50, 100)
    lag_windows: Sequence[int] = tuple(range(1, 31))
    volatility_windows: Sequence[int] = (5, 10, 20, 50)
    atr_window: int = 14
    regime_window: int = 100
    regime_quantiles: Sequence[float] = (0.2, 0.8)


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
    path.mkdir(parents=True, exist_ok=True)


def get_date_range(start_year: int, end_year: Optional[int] = None) -> List[dt.date]:
    if end_year is None:
        end_year = dt.date.today().year
    start_date = dt.date(start_year, 1, 1)
    end_date = dt.date(end_year, 12, 31)
    delta = end_date - start_date
    return [start_date + dt.timedelta(days=i) for i in range(delta.days + 1)]


def to_parquet(df: pd.DataFrame, path: Path) -> None:
    ensure_dir(path.parent)
    df.to_parquet(path, index=False)


def to_csv(df: pd.DataFrame, path: Path) -> None:
    ensure_dir(path.parent)
    df.to_csv(path, index=False)


def load_cached(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path, parse_dates=["timestamp"])


def set_deterministic(seed: int) -> None:
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def chunked(iterable: Sequence[dt.date], size: int) -> Iterable[List[dt.date]]:
    for idx in range(0, len(iterable), size):
        yield list(iterable[idx : idx + size])


def safe_to_datetime(series: pd.Series, tz: str) -> pd.Series:
    timestamp = pd.to_datetime(series, utc=True, errors="coerce")
    return timestamp.dt.tz_convert(tz)


def compute_drawdown(equity_curve: pd.Series) -> pd.Series:
    running_max = equity_curve.cummax()
    drawdown = equity_curve / running_max - 1
    return drawdown
