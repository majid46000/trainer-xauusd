from __future__ import annotations

import logging
from pathlib import Path

from data_loader import DataLoader
from evaluate import evaluate_models
from features import add_features
from labeling import add_labels
from train import train_models
from utils import DataConfig, FeatureConfig, TrainConfig, set_deterministic

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def run_pipeline() -> None:
    try:
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

        logger.info("Loading market data")
        loader = DataLoader(data_config)
        artifacts = loader.load()
        df = artifacts.dataframe

        logger.info("Building features")
        df = add_features(df, feature_config)
        logger.info("Generating labels")
        df = add_labels(df, horizon=train_config.horizon, threshold=train_config.threshold)
        df = df.dropna().reset_index(drop=True)

        feature_cols = [
            col
            for col in df.columns
            if col not in {"timestamp", "label", "future_return", "direction_next", "sample_weight"}
        ]
        logger.info("Training models")
        train_result = train_models(
            df,
            feature_columns=feature_cols,
            label_column="label",
            config=train_config,
        )

        logger.info("Evaluating models")
        evaluate_models(
            df,
            train_result=train_result,
            label_column="label",
            output_dir=data_config.output_dir,
            ensemble_top_k=train_config.ensemble_top_k,
        )
        logger.info("Pipeline completed successfully")
    except Exception as exc:
        logger.error("Pipeline failed", exc_info=True)
        raise


if __name__ == "__main__":
    run_pipeline()
