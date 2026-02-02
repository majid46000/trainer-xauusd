from __future__ import annotations

import pandas as pd


def add_labels(df: pd.DataFrame, horizon: int = 3, threshold: float = 0.0) -> pd.DataFrame:
    df = df.copy()
    df["future_return"] = df["close"].shift(-horizon) / df["close"] - 1
    df["direction_next"] = (df["close"].shift(-1) > df["close"]).astype(int)

    df["label"] = 0
    df.loc[df["future_return"] > threshold, "label"] = 1
    df.loc[df["future_return"] < -threshold, "label"] = -1
    return df
