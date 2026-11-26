"""Classical baselines that consume flattened sequence windows."""

from __future__ import annotations

import os

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

    backend = "sklearn"

    def __init__(self, estimator):
        self.estimator = estimator

    def fit(self, X, y, X_val=None, y_val=None):
        self.estimator.fit(flatten_sequences(X), np.asarray(y).ravel())
        return self

    def predict(self, X):
        return self.estimator.predict(flatten_sequences(X))


_RF_WARNED = False


def rf_backend() -> str:
    """
    RandomForest backend, from env CRB_RF_BACKEND:
      sklearn (default) - scikit-learn RandomForest on CPU (used for the 2026-11-24 benchmark)
      gpu               - XGBoost random-forest mode on CUDA (falls back to sklearn without a GPU)
    """
    global _RF_WARNED
    want = os.environ.get("CRB_RF_BACKEND", "sklearn").strip().lower()
    if want not in ("gpu", "cuda", "xgboost"):
        return "sklearn"
    try:
        import torch
        import xgboost

        if not xgboost.build_info().get("USE_CUDA", False):
            reason = "xgboost was built without CUDA"
        elif torch.cuda.is_available():
            return "xgboost-cuda"
        else:
            reason = "no CUDA GPU"
    except ImportError as exc:
        reason = f"{exc}"
    if not _RF_WARNED:
        print(f"[RandomForest] CRB_RF_BACKEND={want} requested but unavailable ({reason}) -> using sklearn on CPU")
        _RF_WARNED = True
    return "sklearn"


class XGBRandomForest:
    """
    Random forest trained on the GPU with XGBoost's random-forest mode, configured to mirror
    the sklearn baseline: 100 trees, max depth 16, bootstrap-like row subsampling (0.632),
    all features per split for regression / sqrt(p) for classification, balanced class weights.
    Prediction runs on CPU (avoids host/device copy warnings; fast for 100 trees).
    """

    def __init__(self, task: str, seed: int = 42, device: str = "cuda"):
        self.task, self.seed, self.device = task, seed, device
        self.estimator = None
        self.backend = "xgboost-cuda" if device == "cuda" else "xgboost-cpu"

    def fit(self, X, y, X_val=None, y_val=None):
        try:
            return self._fit_xgb(X, y)
        except Exception as exc:  # e.g. GPU out of memory -> never abort a long run
            print(f"[RandomForest] XGBoost ({self.device}) failed: {exc} -> falling back to sklearn on CPU")
            fallback = _make_sklearn_rf(self.task, self.seed)
            fallback.fit(X, y)
            self.estimator, self.backend = fallback.estimator, "sklearn(fallback)"
            return self

    def _fit_xgb(self, X, y):
        from xgboost import XGBRFClassifier, XGBRFRegressor

        Xf = flatten_sequences(X)
        y = np.asarray(y).ravel()
        common = dict(n_estimators=100, max_depth=16, learning_rate=1.0, subsample=0.632,
                      tree_method="hist", max_bin=256, device=self.device, random_state=self.seed,
                      n_jobs=-1, verbosity=0)
        if self.task == "classification":
            p = Xf.shape[1]
            self.estimator = XGBRFClassifier(colsample_bynode=max(1.0 / p, float(np.sqrt(p)) / p), **common)
            y = y.astype(int)
            counts = np.bincount(y).astype(float)
            w = (len(y) / (len(counts) * np.maximum(counts, 1.0)))[y]  # sklearn "balanced"
            self.estimator.fit(Xf, y, sample_weight=w)
        else:
            self.estimator = XGBRFRegressor(colsample_bynode=1.0, **common)
            self.estimator.fit(Xf, y.astype(np.float32))
        self.estimator.set_params(device="cpu")
        return self

    def predict(self, X):
        return self.estimator.predict(flatten_sequences(X))

    def predict_proba(self, X):
        return self.estimator.predict_proba(flatten_sequences(X))


def make_random_forest(task: str, seed: int = 42):
    if rf_backend() == "xgboost-cuda":
        return XGBRandomForest(task, seed=seed, device="cuda")
    return _make_sklearn_rf(task, seed)


def _make_sklearn_rf(task: str, seed: int = 42):
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
