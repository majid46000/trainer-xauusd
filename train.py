from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple
import numpy as np
import optuna
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from utils import CVConfig, TrainConfig

try:
    import lightgbm as lgb
    BOOSTING_BACKEND = "lightgbm"
except ImportError:  # pragma: no cover - optional dependency
    lgb = None
    try:
        import xgboost as xgb
        BOOSTING_BACKEND = "xgboost"
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("Either lightgbm or xgboost must be installed for the main model.") from exc


@dataclass
class FoldPrediction:
    model: str
    strategy: str
    fold: int
    indices: np.ndarray
    y_true: np.ndarray
    y_pred: np.ndarray


@dataclass
class FoldModel:
    model: str
    strategy: str
    fold: int
    estimator: object


@dataclass
class TrainResult:
    models: Dict[str, object]
    metrics: pd.DataFrame
    fold_predictions: List[FoldPrediction]
    fold_models: List[FoldModel]
    feature_columns: List[str]
    best_params: Dict[str, dict]


def add_trend_following_features(df: pd.DataFrame) -> pd.DataFrame:
    df['ema50'] = df['close'].ewm(span=50, adjust=False).mean()
    df['ema200'] = df['close'].ewm(span=200, adjust=False).mean()
    df['trend_signal'] = np.where(df['ema50'] > df['ema200'], 1, -1)
    return df


def add_fvg_features(df: pd.DataFrame) -> pd.DataFrame:
    df['fvg_bull'] = (df['low'].shift(2) > df['high'].shift()) & (df['low'] > df['high'].shift())
    df['fvg_bear'] = (df['high'].shift(2) < df['low'].shift()) & (df['high'] < df['low'].shift())
    df['fvg_bull'] = df['fvg_bull'].astype(int)
    df['fvg_bear'] = df['fvg_bear'].astype(int)
    return df


def add_breakout_features(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
    df['high_roll'] = df['high'].rolling(period).max()
    df['low_roll'] = df['low'].rolling(period).min()
    df['breakout_up'] = (df['close'] > df['high_roll'].shift(1)).astype(int)
    df['breakout_down'] = (df['close'] < df['low_roll'].shift(1)).astype(int)
    return df


def add_macro_correlation_features(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    # Assume df has additional macro columns like 'dxy_close', 'vix_close', 'us10y_close'
    # If not, you need to load and merge them from external sources
    if 'dxy_close' in df.columns:
        df['gold_dxy_cor'] = df['close'].rolling(window).corr(df['dxy_close'])
    if 'vix_close' in df.columns:
        df['gold_vix_cor'] = df['close'].rolling(window).corr(df['vix_close'])
    if 'us10y_close' in df.columns:
        df['gold_us10y_cor'] = df['close'].rolling(window).corr(df['us10y_close'])
    return df


def train_models(
    df: pd.DataFrame,
    feature_columns: List[str],
    label_column: str,
    config: TrainConfig,
) -> TrainResult:
    # Add advanced strategy features
    df = add_trend_following_features(df)
    df = add_fvg_features(df)
    df = add_breakout_features(df)
    df = add_macro_correlation_features(df)
    
    # Update feature_columns to include new features from strategies
    new_features = [
        'ema50', 'ema200', 'trend_signal',  # Trend Following
        'fvg_bull', 'fvg_bear',  # SMC FVG
        'high_roll', 'low_roll', 'breakout_up', 'breakout_down',  # Breakout
        'gold_dxy_cor', 'gold_vix_cor', 'gold_us10y_cor'  # Macro Correlation (if macro data available)
    ]
    feature_columns = list(set(feature_columns + new_features) - {label_column, 'timestamp', 'future_return', 'direction_next'})
    
    # Drop NaNs after adding features
    df = df.dropna().reset_index(drop=True)
    
    X = df[feature_columns].values
    y = df[label_column].values

    cv_config = CVConfig(
        test_splits=config.test_splits,
        rolling_window=config.rolling_window,
        expanding_window=config.expanding_window,
    )

    models: Dict[str, object] = {}
    fold_predictions: List[FoldPrediction] = []
    fold_models: List[FoldModel] = []
    metrics_rows = []
    best_params_by_strategy: Dict[str, dict] = {}

    for strategy in ["rolling", "expanding"]:
        splits = list(_generate_splits(len(df), cv_config, strategy))
        if not splits:
            continue

        tuning_train_idx, _ = splits[0]
        tuned_params = _tune_boosting_model(
            X[tuning_train_idx],
            y[tuning_train_idx],
            config,
        )

        model_factories = _build_model_factories(config.seed, tuned_params)
        best_params_by_strategy[strategy] = tuned_params

        for model_name, model_factory in model_factories.items():
            fold_idx = 0
            for train_idx, test_idx in splits:
                fold_idx += 1
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]

                model = model_factory()
                model = _fit_model(model, model_name, X_train, y_train)
                preds = _predict_model(model, model_name, X_test)

                metrics_rows.append(
                    _fold_metrics(
                        model_name=model_name,
                        strategy=strategy,
                        fold=fold_idx,
                        y_true=y_test,
                        y_pred=preds,
                    )
                )

                fold_predictions.append(
                    FoldPrediction(
                        model=model_name,
                        strategy=strategy,
                        fold=fold_idx,
                        indices=test_idx,
                        y_true=y_test,
                        y_pred=preds,
                    )
                )

                fold_models.append(
                    FoldModel(
                        model=model_name,
                        strategy=strategy,
                        fold=fold_idx,
                        estimator=model,
                    )
                )

            # ── الجزء المصحح ──
            final_model = model_factory()
            final_model = _fit_model(final_model, model_name, X, y)

            model_key = f"{strategy}_{model_name}"
            models[model_key] = final_model
            # ──────────────────────

    metrics = pd.DataFrame(metrics_rows)

    return TrainResult(
        models=models,
        metrics=metrics,
        fold_predictions=fold_predictions,
        fold_models=fold_models,
        feature_columns=feature_columns,
        best_params=best_params_by_strategy,
    )


def _build_model_factories(seed: int, tuned_params: dict) -> Dict[str, callable]:
    return {
        "logistic_regression": lambda: Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        max_iter=2000,
                        random_state=seed,
                        multi_class="multinomial",
                        class_weight="balanced",
                    ),
                ),
            ]
        ),
        "random_forest": lambda: RandomForestClassifier(
            n_estimators=400,
            random_state=seed,
            max_depth=8,
            n_jobs=-1,
            class_weight="balanced_subsample",
        ),
        "boosting": lambda: _build_boosting_model(tuned_params, seed),
    }


def _build_boosting_model(params: dict, seed: int) -> object:
    if BOOSTING_BACKEND == "lightgbm":
        return lgb.LGBMClassifier(
            objective="multiclass",
            num_class=3,
            random_state=seed,
            n_jobs=-1,
            **params,
        )

    return xgb.XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        random_state=seed,
        eval_metric="mlogloss",
        n_estimators=params.get("n_estimators", 500),
        max_depth=params.get("max_depth", 6),
        learning_rate=params.get("learning_rate", 0.05),
        subsample=params.get("subsample", 0.8),
        colsample_bytree=params.get("colsample_bytree", 0.8),
        min_child_weight=params.get("min_child_weight", 1.0),
        reg_alpha=params.get("reg_alpha", 0.0),
        reg_lambda=params.get("reg_lambda", 1.0),
        n_jobs=-1,
    )


def _fit_model(model: object, model_name: str, X_train: np.ndarray, y_train: np.ndarray) -> object:
    unique_labels = np.unique(y_train)
    if unique_labels.size < 2:
        model.fit(X_train, y_train)
        return model

    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=unique_labels,
        y=y_train,
    )
    weights = {label: weight for label, weight in zip(unique_labels, class_weights)}

    if model_name in ("logistic_regression", "random_forest"):
        model.fit(X_train, y_train)
        return model

    if model_name == "boosting":
        sample_weight = np.array([weights.get(label, 1.0) for label in y_train])
        model.fit(X_train, _to_boosting_labels(y_train), sample_weight=sample_weight)
        return model

    # fallback
    sample_weight = np.array([weights.get(label, 1.0) for label in y_train])
    model.fit(X_train, y_train, sample_weight=sample_weight)
    return model


def _predict_model(model: object, model_name: str, X: np.ndarray) -> np.ndarray:
    preds = model.predict(X)
    if model_name == "boosting":
        return _from_boosting_labels(preds)
    return preds


def _generate_splits(n_samples: int, config: CVConfig, strategy: str) -> Iterable[Tuple[np.ndarray, np.ndarray]]:
    if config.test_splits <= 0:
        return
    if strategy == "rolling":
        train_window = config.rolling_window
    else:
        train_window = config.expanding_window

    if n_samples <= train_window:
        return

    test_window = max(1, int((n_samples - train_window) / config.test_splits))

    for split in range(config.test_splits):
        train_end = train_window + split * test_window
        test_start = train_end
        test_end = min(test_start + test_window, n_samples)

        if test_start >= n_samples:
            break

        if strategy == "rolling":
            train_start = max(0, train_end - train_window)
        else:
            train_start = 0

        train_idx = np.arange(train_start, train_end)
        test_idx = np.arange(test_start, test_end)

        if len(train_idx) == 0 or len(test_idx) == 0:
            continue

        yield train_idx, test_idx


def _fold_metrics(model_name: str, strategy: str, fold: int, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    accuracy = np.mean(y_true == y_pred)

    return {
        "model": model_name,
        "strategy": strategy,
        "fold": fold,
        "accuracy": accuracy,
        "f1_macro": f1,
    }


def _tune_boosting_model(X: np.ndarray, y: np.ndarray, config: TrainConfig) -> dict:
    split_idx = int(len(X) * 0.8)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 300, 900),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_float("min_child_weight", 0.5, 10.0, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 5.0),
        }

        model = _build_boosting_model(params, config.seed)
        model.fit(X_train, _to_boosting_labels(y_train))

        preds = _from_boosting_labels(model.predict(X_val))
        return f1_score(y_val, preds, average="macro", zero_division=0)

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=config.seed)
    )
    study.optimize(objective, n_trials=config.optuna_trials, timeout=config.optuna_timeout)

    return study.best_params


def _to_boosting_labels(y: np.ndarray) -> np.ndarray:
    return (y + 1).astype(int)


def _from_boosting_labels(y: np.ndarray) -> np.ndarray:
    return y.astype(int) - 1
