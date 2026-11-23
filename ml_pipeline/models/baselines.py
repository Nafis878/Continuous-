"""Classical baselines that consume flattened sequence windows."""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LogisticRegression, Ridge


def flatten_sequences(X: np.ndarray) -> np.ndarray:
    return np.asarray(X, dtype=np.float32).reshape(X.shape[0], -1)


class FlatModel:
    """Wraps an sklearn estimator that expects flattened (N, T*F) inputs."""

    def __init__(self, estimator):
        self.estimator = estimator

    def fit(self, X, y, X_val=None, y_val=None):
        self.estimator.fit(flatten_sequences(X), np.asarray(y).ravel())
        return self

    def predict(self, X):
        return self.estimator.predict(flatten_sequences(X))


def make_random_forest(task: str, seed: int = 42):
    if task == "classification":
        est = RandomForestClassifier(
            n_estimators=100,
            max_depth=16,
            n_jobs=-1,
            random_state=seed,
            class_weight="balanced_subsample",
        )
    else:
        est = RandomForestRegressor(
            n_estimators=100,
            max_depth=16,
            n_jobs=-1,
            random_state=seed,
        )
    return FlatModel(est)


def make_gradient_boosting(task: str, seed: int = 42):
    if task == "classification":
        est = HistGradientBoostingClassifier(
            max_depth=8,
            max_iter=100,
            learning_rate=0.08,
            random_state=seed,
            class_weight="balanced",
        )
    else:
        est = HistGradientBoostingRegressor(
            max_depth=8,
            max_iter=100,
            learning_rate=0.08,
            random_state=seed,
        )
    return FlatModel(est)


def make_linear(task: str, seed: int = 42):
    if task == "classification":
        est = LogisticRegression(
            max_iter=500,
            n_jobs=-1,
            random_state=seed,
            class_weight="balanced",
        )
    else:
        est = Ridge(alpha=1.0)
    return FlatModel(est)


class ConstantModel:
    """
    Trivial reference predictor (sanity baseline).

    regression: predicts the training-set mean RUL.
    classification: always predicts ``constant`` (e.g. SCANIA class 0 or 4).
    Any real model should beat these; if it doesn't, it has learned nothing.
    """

    def __init__(self, task: str, constant: int | None = None, n_classes: int | None = None):
        self.task = task
        self.constant = constant
        self.n_classes = n_classes
        self.value_: float | int | None = None

    def fit(self, X, y, X_val=None, y_val=None):
        y = np.asarray(y).ravel()
        if self.task == "classification":
            self.value_ = int(self.constant if self.constant is not None else np.bincount(y).argmax())
            self.n_classes = int(self.n_classes or (int(y.max()) + 1))
        else:
            self.value_ = float(y.mean())
        return self

    def predict(self, X):
        n = len(X)
        if self.task == "classification":
            return np.full(n, self.value_, dtype=int)
        return np.full(n, self.value_, dtype=float)

    def predict_proba(self, X):
        proba = np.zeros((len(X), self.n_classes), dtype=float)
        proba[:, self.value_] = 1.0
        return proba
