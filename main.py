from __future__ import annotations

from pathlib import Path

from data_loader import DataLoader
from evaluate import evaluate_models
from features import add_features
from labeling import add_labels
from train import train_models
from utils import DataConfig, FeatureConfig, TrainConfig, set_deterministic


def run_pipeline() -> None:
    data_config = DataConfig(
        symbol="XAUUSD",
        timeframe="M5",
        start_year=2006,
        source="ctrader",
        data_dir=Path("data"),
        cache_dir=Path("data/cache"),
        output_dir=Path("data/outputs"),
        price_scale=1000,
        max_workers=4,
    )
    feature_config = FeatureConfig()
    train_config = TrainConfig(
        horizon=3,
        test_splits=5,
        threshold=0.0,
        seed=7,
        rolling_window=4000,
        expanding_window=4000,
        optuna_trials=30,
        optuna_timeout=600,
        ensemble_top_k=2,
    )
    set_deterministic(train_config.seed)

    loader = DataLoader(data_config)
    artifacts = loader.load()
    df = artifacts.dataframe

    df = add_features(df, feature_config)
    df = add_labels(df, horizon=train_config.horizon, threshold=train_config.threshold)
    df = df.dropna().reset_index(drop=True)

    feature_cols = [
        col
        for col in df.columns
        if col not in {"timestamp", "label", "future_return", "direction_next"}
    ]
    train_result = train_models(
        df,
        feature_columns=feature_cols,
        label_column="label",
        config=train_config,
    )

    evaluate_models(
        df,
        train_result=train_result,
        label_column="label",
        output_dir=data_config.output_dir,
        ensemble_top_k=train_config.ensemble_top_k,
    )


if __name__ == "__main__":
    run_pipeline()
