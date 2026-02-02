from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

from utils import ensure_dir


@dataclass
class EvaluationArtifacts:
    metrics_path: Path
    confusion_matrix_path: Path
    equity_curve_path: Path


def evaluate_models(
    df: pd.DataFrame,
    models: Dict[str, object],
    feature_columns: list[str],
    label_column: str,
    output_dir: Path,
) -> EvaluationArtifacts:
    ensure_dir(output_dir)
    metrics_rows = []

    for name, model in models.items():
        preds = model.predict(df[feature_columns].values)
        cm = confusion_matrix(df[label_column].values, preds, labels=[-1, 0, 1])
        metrics_rows.append(_equity_metrics(df, preds, name))
        _plot_confusion(cm, name, output_dir)

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_path = output_dir / "metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)

    equity_curve_path = output_dir / "equity_curve.png"
    _plot_equity_curve(metrics_df, equity_curve_path)

    confusion_matrix_path = output_dir / "confusion_matrix_gradient_boosting.png"
    return EvaluationArtifacts(metrics_path, confusion_matrix_path, equity_curve_path)


def _plot_confusion(cm: np.ndarray, model_name: str, output_dir: Path) -> None:
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["-1", "0", "1"])
    disp.plot(cmap="Blues", values_format="d")
    plt.title(f"Confusion Matrix: {model_name}")
    path = output_dir / f"confusion_matrix_{model_name}.png"
    plt.savefig(path, bbox_inches="tight")
    plt.close()


def _equity_metrics(df: pd.DataFrame, preds: np.ndarray, model_name: str) -> dict:
    returns = df["future_return"].fillna(0).values
    positions = preds
    strategy_returns = returns * positions
    equity_curve = (1 + strategy_returns).cumprod()
    sharpe_like = np.mean(strategy_returns) / (np.std(strategy_returns) + 1e-9) * np.sqrt(252 * 288)
    return {
        "model": model_name,
        "final_equity": float(equity_curve[-1]),
        "sharpe_like": float(sharpe_like),
    }


def _plot_equity_curve(metrics_df: pd.DataFrame, path: Path) -> None:
    plt.figure(figsize=(8, 4))
    plt.bar(metrics_df["model"], metrics_df["final_equity"])
    plt.title("Final Equity by Model (Research Only)")
    plt.ylabel("Equity")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
