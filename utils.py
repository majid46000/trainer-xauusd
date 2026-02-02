from __future__ import annotations

import datetime as dt
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

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


@dataclass(frozen=True)
class TrainConfig:
    horizon: int = 3
    test_splits: int = 5
    threshold: float = 0.0
    seed: int = 7


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
    os.environ["PYTHONHASHSEED"] = str(seed)
