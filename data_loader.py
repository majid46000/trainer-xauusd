from __future__ import annotations

import datetime as dt
import importlib.util
import lzma
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests

from utils import DataConfig, ensure_dir, get_date_range, load_cached, to_csv, to_parquet


@dataclass(frozen=True)
class DataArtifacts:
    dataframe: pd.DataFrame
    parquet_path: Path
    csv_path: Path


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
            last_timestamp = cached["timestamp"].max()
            start_date = pd.to_datetime(last_timestamp).date() + dt.timedelta(days=1)
        else:
            last_timestamp = None
            start_date = dt.date(self.config.start_year, 1, 1)

        if self.config.source.lower() == "ctrader":
            df = self._load_from_ctrader(last_timestamp)
            if df is None:
                df = self._load_from_dukascopy(start_date)
        else:
            df = self._load_from_dukascopy(start_date)

        if cached is not None and not cached.empty:
            combined = pd.concat([cached, df], ignore_index=True)
        else:
            combined = df

        combined = combined.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
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
        frames = []
        for day in all_days:
            daily = self._load_dukascopy_day(day)
            if daily is not None and not daily.empty:
                frames.append(daily)
        if not frames:
            return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
        df = pd.concat(frames, ignore_index=True)
        df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
        return df

    def _load_dukascopy_day(self, day: dt.date) -> Optional[pd.DataFrame]:
        raw_path = self.config.cache_dir / "dukascopy" / self.config.symbol / f"{day:%Y-%m-%d}.bi5"
        ensure_dir(raw_path.parent)
        if not raw_path.exists():
            url = self._dukascopy_url(day)
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
        for unpacked in struct.iter_unpack(">6i", decompressed):
            time_ms, open_, close, low, high, volume = unpacked
            timestamp = dt.datetime.combine(day, dt.time()) + dt.timedelta(milliseconds=time_ms)
            records.append(
                {
                    "timestamp": timestamp,
                    "open": open_ / self.config.price_scale,
                    "high": high / self.config.price_scale,
                    "low": low / self.config.price_scale,
                    "close": close / self.config.price_scale,
                    "volume": volume,
                }
            )

        df = pd.DataFrame(records)
        if df.empty:
            return None
        df = df.sort_values("timestamp").reset_index(drop=True)
        df = self._resample_to_timeframe(df)
        return df

    def _dukascopy_url(self, day: dt.date) -> str:
        month = day.month - 1
        return (
            "https://datafeed.dukascopy.com/datafeed/"
            f"{self.config.symbol}/{day.year}/{month:02d}/{day.day:02d}/BID_candles_min_1.bi5"
        )

    def _resample_to_timeframe(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.set_index("timestamp")
        freq = self._timeframe_to_pandas()
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

    def _validate_and_fill(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=False)
        df = df.sort_values("timestamp").reset_index(drop=True)
        freq = self._timeframe_to_pandas()
        expected = pd.date_range(df["timestamp"].iloc[0], df["timestamp"].iloc[-1], freq=freq)
        missing = expected.difference(df["timestamp"])
        if not missing.empty:
            missing_df = pd.DataFrame({"timestamp": missing})
            missing_df["open"] = np.nan
            missing_df["high"] = np.nan
            missing_df["low"] = np.nan
            missing_df["close"] = np.nan
            missing_df["volume"] = 0
            df = pd.concat([df, missing_df], ignore_index=True).sort_values("timestamp").reset_index(drop=True)
            df[["open", "high", "low", "close"]] = df[["open", "high", "low", "close"]].ffill()
        return df

    def _timeframe_to_pandas(self) -> str:
        if self.config.timeframe.upper() == "M5":
            return "5min"
        if self.config.timeframe.upper().startswith("M"):
            minutes = self.config.timeframe[1:]
            return f"{int(minutes)}min"
        raise ValueError(f"Unsupported timeframe: {self.config.timeframe}")
