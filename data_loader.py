from __future__ import annotations

import datetime as dt
import importlib.util
import logging
import lzma
import struct
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
import requests

from utils import DataConfig, ensure_dir, get_date_range, load_cached, safe_to_datetime, to_csv, to_parquet

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DataArtifacts:
    dataframe: pd.DataFrame
    parquet_path: Path
    csv_path: Path


def _dukascopy_day_worker(
    day: dt.date,
    symbol: str,
    cache_dir: Path,
    price_scale: int,
    timeframe: str,
) -> Optional[pd.DataFrame]:
    raw_path = cache_dir / "dukascopy" / symbol / f"{day:%Y-%m-%d}.bi5"
    processed_path = cache_dir / "dukascopy" / symbol / f"{day:%Y-%m-%d}_{timeframe}.parquet"
    ensure_dir(raw_path.parent)
    if processed_path.exists():
        try:
            return pd.read_parquet(processed_path)
        except Exception:
            processed_path.unlink(missing_ok=True)

    if not raw_path.exists():
        month = day.month - 1
        url = (
            "https://datafeed.dukascopy.com/datafeed/"
            f"{symbol}/{day.year}/{month:02d}/{day.day:02d}/BID_candles_min_1.bi5"
        )
        response = requests.get(url, timeout=30)
        if response.status_code != 200:
            return None
        raw_path.write_bytes(response.content)

    raw = raw_path.read_bytes()
    if not raw:
        return None

    try:
        decompressed = lzma.decompress(raw)
    except lzma.LZMAError:
        return None

    records = []
    for unpacked in struct.iter_unpack(">6i", decompressed
    ):
        time_ms, open_, close, low, high, volume = unpacked
        timestamp = dt.datetime.combine(day, dt.time()) + dt.timedelta(milliseconds=time_ms)
        records.append(
            {
                "timestamp": timestamp,
                "open": open_ / price_scale,
                "high": high / price_scale,
                "low": low / price_scale,
                "close": close / price_scale,
                "volume": volume,
            }
        )

    df = pd.DataFrame(records)
    if df.empty:
        return None
    df = df.sort_values("timestamp").reset_index(drop=True)
    df = _resample_to_timeframe(df, timeframe)
    if not df.empty:
        df.to_parquet(processed_path, index=False)
    return df


def _resample_to_timeframe(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    df = df.set_index("timestamp")
    freq = _timeframe_to_pandas(timeframe)
    df = df.resample(freq).agg(
        {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }
    )
    df = df.dropna().reset_index()
    return df


def _timeframe_to_pandas(timeframe: str) -> str:
    if timeframe.upper() == "M5":
        return "5min"
    if timeframe.upper().startswith("M"):
        minutes = timeframe[1:]
        return f"{int(minutes)}min"
    raise ValueError(f"Unsupported timeframe: {timeframe}")


class DataLoader:
    def __init__(self, config: DataConfig) -> None:
        self.config = config
        ensure_dir(self.config.cache_dir)
        ensure_dir(self.config.output_dir)

    def load(self) -> DataArtifacts:
        parquet_path = self.config.output_dir / f"{self.config.symbol}_{self.config.timeframe}.parquet"
        csv_path = self.config.output_dir / f"{self.config.symbol}_{self.config.timeframe}.csv"
        cached = load_cached(parquet_path)

        if cached is not None and not cached.empty:
            cached["timestamp"] = safe_to_datetime(cached["timestamp"], self.config.timezone)
            last_timestamp = cached["timestamp"].max()
            start_date = last_timestamp.date() + dt.timedelta(days=1)
        else:
            last_timestamp = None
            start_date = dt.date(self.config.start_year, 1, 1)

        if self.config.source.lower() == "ctrader":
            df = self._load_from_ctrader(last_timestamp)
            if df is None:
                df = self._load_from_dukascopy(start_date)
        else:
            df = self._load_from_dukascopy(start_date)

        if df is None:
            df = pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])

        if cached is not None and not cached.empty:
            combined = pd.concat([cached, df], ignore_index=True)
        else:
            combined = df

        combined = self._validate_and_fill(combined)
        to_parquet(combined, parquet_path)
        to_csv(combined, csv_path)
        return DataArtifacts(dataframe=combined, parquet_path=parquet_path, csv_path=csv_path)

    def _load_from_ctrader(self, last_timestamp: Optional[pd.Timestamp]) -> Optional[pd.DataFrame]:
        if importlib.util.find_spec("ctrader_open_api") is None:
            return None

        from ctrader_open_api import CTraderClient  # type: ignore

        client = CTraderClient()
        df = client.fetch_candles(
            symbol=self.config.symbol,
            timeframe=self.config.timeframe,
            start_time=last_timestamp,
        )
        if df is None or df.empty:
            return None
        return df

    def _load_from_dukascopy(self, start_date: dt.date) -> pd.DataFrame:
        end_year = self.config.end_year or dt.date.today().year
        all_days = get_date_range(start_date.year, end_year)
        all_days = [day for day in all_days if day >= start_date]
        if not all_days:
            return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])

        frames: list[pd.DataFrame] = []
        with ProcessPoolExecutor(max_workers=self.config.max_workers) as executor:
            results = executor.map(
                _dukascopy_day_worker,
                all_days,
                [self.config.symbol] * len(all_days),
                [self.config.cache_dir] * len(all_days),
                [self.config.price_scale] * len(all_days),
                [self.config.timeframe] * len(all_days),
            )
            for result in results:
                if result is not None and not result.empty:
                    frames.append(result)

        if not frames:
            return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
        df = pd.concat(frames, ignore_index=True)
        df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
        return df

    def _validate_and_fill(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["timestamp"] = safe_to_datetime(df["timestamp"], self.config.timezone)
        df = df.dropna(subset=["timestamp"]).drop_duplicates(subset=["timestamp"])
        df = df.sort_values("timestamp").reset_index(drop=True)
        if df.empty:
            return df

        freq = _timeframe_to_pandas(self.config.timeframe)
        expected_index = pd.date_range(df["timestamp"].iloc[0], df["timestamp"].iloc[-1], freq=freq)
        observed_index = pd.DatetimeIndex(df["timestamp"])
        unexpected = observed_index.difference(expected_index)
        if not unexpected.empty:
            df = df[~df["timestamp"].isin(unexpected)]
            df = df.sort_values("timestamp").reset_index(drop=True)

        missing = expected_index.difference(pd.DatetimeIndex(df["timestamp"]))
        if not missing.empty:
            missing_df = pd.DataFrame({"timestamp": missing})
            missing_df["open"] = np.nan
            missing_df["high"] = np.nan
            missing_df["low"] = np.nan
            missing_df["close"] = np.nan
            missing_df["volume"] = 0
            df = pd.concat([df, missing_df], ignore_index=True).sort_values("timestamp").reset_index(drop=True)
            if self.config.fill_method == "ffill":
                df[["open", "high", "low", "close"]] = df[["open", "high", "low", "close"]].ffill()
            elif self.config.fill_method == "bfill":
                df[["open", "high", "low", "close"]] = df[["open", "high", "low", "close"]].bfill()
        return df
