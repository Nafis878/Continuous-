"""Shared evaluation metrics for RUL regression and classification."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    recall_score,
)


def phm_score(y_true, y_pred) -> float:
    """NASA PHM08 asymmetric scoring function (lower is better)."""
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    d = y_pred - y_true
    score = np.where(d < 0, np.exp(-d / 13.0) - 1.0, np.exp(d / 10.0) - 1.0)
    return float(np.sum(score))


def regression_metrics_full(y_true, y_pred) -> dict:
    """RMSE, MAE, R², PHM Score for ablation comparisons."""
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred)) if len(y_true) > 1 else float("nan")
    return {
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "phm_score": phm_score(y_true, y_pred),
        "n_test": int(len(y_true)),
    }


def rul_tolerance_accuracy(y_true, y_pred, tol: float = 10.0) -> float:
    """Fraction of predictions within ±tol of true RUL (common PHM accuracy proxy)."""
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    return float(np.mean(np.abs(y_true - y_pred) <= tol))


# IDA 2024 SCANIA Component X misclassification costs (Cost_actual_predicted).
# Diagonal = 0. Missing off-diagonals filled from challenge PDF.
SCANIA_COST = np.array(
    [
        [0, 7, 8, 9, 10],
        [200, 0, 7, 8, 9],
        [300, 200, 0, 7, 8],
        [400, 300, 200, 0, 7],
        [500, 400, 300, 200, 0],
    ],
    dtype=float,
)


def scania_total_cost(y_true, y_pred, cost: np.ndarray = SCANIA_COST) -> float:
    y_true = np.asarray(y_true, dtype=int).ravel()
    y_pred = np.asarray(y_pred, dtype=int).ravel()
    return float(np.sum(cost[y_true, y_pred]))


def scania_metrics(y_true, y_pred) -> dict:
    """SCANIA family metrics: challenge cost (primary), macro-F1, recalls, accuracy."""
    y_true = np.asarray(y_true, dtype=int).ravel()
    y_pred = np.asarray(y_pred, dtype=int).ravel()
    recalls = recall_score(
        y_true, y_pred, labels=[0, 1, 2, 3, 4], average=None, zero_division=0
    )
    return {
        "family": "SCANIA",
        "total_cost": scania_total_cost(y_true, y_pred),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "recall_0": float(recalls[0]),
        "recall_1": float(recalls[1]),
        "recall_2": float(recalls[2]),
        "recall_3": float(recalls[3]),
        "recall_4": float(recalls[4]),
        "n_test": int(len(y_true)),
        "primary_metric": "total_cost",
        "primary_lower_is_better": True,
    }


def rul_family_metrics(y_true, y_pred) -> dict:
    """RUL family metrics: RMSE (primary), MAE, R², PHM Score."""
    m = regression_metrics_full(y_true, y_pred)
    m.update(
        {
            "family": "RUL",
            "tol_acc_10": rul_tolerance_accuracy(y_true, y_pred, tol=10.0),
            "primary_metric": "rmse",
            "primary_lower_is_better": True,
        }
    )
    return m
