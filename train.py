from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass
class TrainResult:
    models: Dict[str, object]
    metrics: pd.DataFrame
    feature_columns: List[str]


def train_models(
    df: pd.DataFrame,
    feature_columns: List[str],
    label_column: str,
    splits: int = 5,
    seed: int = 7,
) -> TrainResult:
    X = df[feature_columns].values
    y = df[label_column].values

    models = _build_models(seed)
    tscv = TimeSeriesSplit(n_splits=splits)
    rows = []

    for name, model in models.items():
        fold_idx = 0
        for train_idx, test_idx in tscv.split(X):
            fold_idx += 1
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            report = classification_report(y_test, preds, output_dict=True, zero_division=0)
            rows.append(
                {
                    "model": name,
                    "fold": fold_idx,
                    "accuracy": report["accuracy"],
                    "precision_macro": report["macro avg"]["precision"],
                    "recall_macro": report["macro avg"]["recall"],
                    "f1_macro": report["macro avg"]["f1-score"],
                }
            )
        model.fit(X, y)

    metrics = pd.DataFrame(rows)
    return TrainResult(models=models, metrics=metrics, feature_columns=feature_columns)


def _build_models(seed: int) -> Dict[str, object]:
    return {
        "logistic_regression": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        max_iter=1000,
                        random_state=seed,
                        multi_class="multinomial",
                    ),
                ),
            ]
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=200,
            random_state=seed,
            max_depth=6,
            n_jobs=-1,
        ),
        "gradient_boosting": GradientBoostingClassifier(random_state=seed),
    }
