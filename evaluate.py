from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from train import FoldModel, FoldPrediction, TrainResult
from utils import compute_drawdown, ensure_dir


@dataclass
class EvaluationArtifacts:
    metrics_path: Path
    fold_metrics_path: Path
    equity_curve_path: Path


def evaluate_models(
    df: pd.DataFrame,
    train_result: TrainResult,
    label_column: str,
    output_dir: Path,
    ensemble_top_k: int,
) -> EvaluationArtifacts:
    ensure_dir(output_dir)

    fold_metrics = _compute_fold_metrics(df, train_result.fold_predictions, label_column)
    fold_metrics_path = output_dir / "fold_metrics.csv"
    fold_metrics.to_csv(fold_metrics_path, index=False)

    summary_metrics = _summarize_metrics(fold_metrics)
    summary_metrics_path = output_dir / "metrics.csv"
    summary_metrics.to_csv(summary_metrics_path, index=False)

    equity_curve_path = output_dir / "equity_curve.png"
    _plot_equity_curve(df, train_result.fold_predictions, ensemble_top_k, equity_curve_path)

    return EvaluationArtifacts(summary_metrics_path, fold_metrics_path, equity_curve_path)


def _compute_fold_metrics(
    df: pd.DataFrame,
    fold_predictions: List[FoldPrediction],
    label_column: str,
) -> pd.DataFrame:
    rows = []
    for prediction in fold_predictions:
        y_true = prediction.y_true
        y_pred = prediction.y_pred
        future_returns = df.loc[prediction.indices, "future_return"].fillna(0).values
        positions = y_pred
        strategy_returns = future_returns * positions
        equity_curve = (1 + strategy_returns).cumprod()
        drawdown = compute_drawdown(pd.Series(equity_curve))

        rows.append(
            {
                "model": prediction.model,
                "strategy": prediction.strategy,
                "fold": prediction.fold,
                "accuracy": accuracy_score(y_true, y_pred),
                "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
                "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
                "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
                "sharpe": _sharpe_like(strategy_returns),
                "max_drawdown": float(drawdown.min()) if not drawdown.empty else 0.0,
                "winrate": float((strategy_returns > 0).mean()),
            }
        )

    return pd.DataFrame(rows)


def _summarize_metrics(fold_metrics: pd.DataFrame) -> pd.DataFrame:
    grouped = fold_metrics.groupby(["model", "strategy"])
    summary = grouped.agg(
        accuracy_mean=("accuracy", "mean"),
        accuracy_std=("accuracy", "std"),
        precision_mean=("precision_macro", "mean"),
        precision_std=("precision_macro", "std"),
        recall_mean=("recall_macro", "mean"),
        recall_std=("recall_macro", "std"),
        f1_mean=("f1_macro", "mean"),
        f1_std=("f1_macro", "std"),
        sharpe_mean=("sharpe", "mean"),
        sharpe_std=("sharpe", "std"),
        max_drawdown_mean=("max_drawdown", "mean"),
        winrate_mean=("winrate", "mean"),
    ).reset_index()
    summary["stability_score"] = summary["f1_std"].fillna(0)
    return summary


def _plot_equity_curve(
    df: pd.DataFrame,
    fold_predictions: List[FoldPrediction],
    ensemble_top_k: int,
    path: Path,
) -> None:
    plt.figure(figsize=(10, 5))

    for model_name, strategy in sorted({(p.model, p.strategy) for p in fold_predictions}):
        model_predictions = [
            p for p in fold_predictions if p.model == model_name and p.strategy == strategy
        ]
        if not model_predictions:
            continue
        combined = _combine_predictions(df, model_predictions)
        equity_curve = (1 + combined["strategy_returns"]).cumprod()
        label = f"{strategy}_{model_name}"
        plt.plot(combined["timestamp"], equity_curve, label=label)

        ensemble_curve = _ensemble_curve(
            df,
            model_predictions,
            ensemble_top_k,
        )
        if ensemble_curve is not None:
            plt.plot(
                ensemble_curve.index,
                ensemble_curve.values,
                label=f"{label}_ensemble",
            )

    plt.title("Equity Curves (OOS by Fold)")
    plt.xlabel("Timestamp")
    plt.ylabel("Equity")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def _combine_predictions(df: pd.DataFrame, predictions: List[FoldPrediction]) -> pd.DataFrame:
    rows = []
    for prediction in predictions:
        data = df.loc[prediction.indices, ["timestamp", "future_return"]].copy()
        data["position"] = prediction.y_pred
        rows.append(data)
    combined = pd.concat(rows).sort_values("timestamp")
    combined["strategy_returns"] = combined["future_return"].fillna(0) * combined["position"]
    return combined


def _ensemble_curve(
    df: pd.DataFrame,
    predictions: List[FoldPrediction],
    ensemble_top_k: int,
) -> pd.Series | None:
    if ensemble_top_k <= 1:
        return None
    metrics = pd.DataFrame(
        [
            {
                "fold": p.fold,
                "f1": f1_score(p.y_true, p.y_pred, average="macro", zero_division=0),
            }
            for p in predictions
        ]
    ).sort_values("f1", ascending=False)
    top_folds = metrics.head(ensemble_top_k)["fold"].tolist()
    selected = [p for p in predictions if p.fold in top_folds]
    if not selected:
        return None

    combined = _combine_predictions(df, selected)
    equity_curve = (1 + combined["strategy_returns"]).cumprod()
    return equity_curve


def _sharpe_like(strategy_returns: np.ndarray) -> float:
    if strategy_returns.size == 0:
        return 0.0
    mean = np.mean(strategy_returns)
    std = np.std(strategy_returns)
    if std == 0:
        return 0.0
    return float(mean / std * np.sqrt(252 * 288))
