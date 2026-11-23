"""
Continuous RUL Benchmark (CRB) — protocol.

Defines, for every dataset, exactly how windows are built, scaled, scored, so
all models are compared on identical inputs. See BENCHMARK.md for the spec.

Protocol summary
----------------
* Splits: fixed unit-level splits (seed 42) from ``datasets.ablation_data``.
* Windows: continuous sliding windows, step 1, fixed length per dataset
  (CMAPSS/PHM08: 30, N-CMAPSS: 20); units shorter than the window are
  left-padded with their first row.
* Scaling: StandardScaler fit on training windows only.
* Features: CMAPSS FD001/FD003 "selected" (17), FD002/FD004/PHM08 "all" (24),
  N-CMAPSS W + X_s.
* Targets: piece-wise linear RUL, capped (125; N-CMAPSS 100).
* Scoring:
    continuous  - every test window (primary)
    last-window - last window of every test unit (classic CMAPSS benchmark;
                  PHM08 test units are cut at random pre-failure cycles first;
                  not defined for N-CMAPSS, whose last window is always RUL 0)
* Seeds: model seeds vary; data splits never do.
"""

from __future__ import annotations

import time

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

from datasets.ablation_data import load_raw_cmapss, load_raw_n_cmapss, load_raw_phm08
from datasets.sequence_utils import build_windows_from_units
from metrics import phm_score

HARD = {"CMAPSS-FD002", "CMAPSS-FD004", "PHM08"}
CRITICAL_RUL = 30.0

# name -> loader. N-CMAPSS entries are skipped automatically if the .h5 is missing.
DATASETS = {
    "FD001": lambda: load_raw_cmapss("FD001"),
    "FD002": lambda: load_raw_cmapss("FD002"),
    "FD003": lambda: load_raw_cmapss("FD003"),
    "FD004": lambda: load_raw_cmapss("FD004"),
    "PHM08": load_raw_phm08,
    "NCMAPSS-DS01": lambda: load_raw_n_cmapss("N-CMAPSS_DS01-005.h5"),
    "NCMAPSS-DS02": lambda: load_raw_n_cmapss("N-CMAPSS_DS02-006.h5"),
    "NCMAPSS-DS03": lambda: load_raw_n_cmapss("N-CMAPSS_DS03-012.h5"),
    "NCMAPSS-DS05": lambda: load_raw_n_cmapss("N-CMAPSS_DS05.h5"),
    "NCMAPSS-DS07": lambda: load_raw_n_cmapss("N-CMAPSS_DS07.h5"),
}


def _windows(df, feats, seq_len):
    X, y, u = build_windows_from_units(df, feats, "RUL", seq_len=seq_len, last_only=False)
    return X, y.astype(np.float32), np.asarray(u).astype(np.int64)


def _head(arrs, n):
    """Keep the first n windows (units stay contiguous) — quick mode only."""
    return tuple(a[:n] for a in arrs)


def build_bundle(pack: dict, quick: bool = False) -> dict:
    name = pack["name"]
    feat_key = "all" if name in HARD and "all" in pack["feature_presets"] else pack["default_features"]
    feats = list(pack["feature_presets"][feat_key])
    seq = int(pack["default_seq_len"])

    tr = _windows(pack["train_df"], feats, seq)
    va = _windows(pack["val_df"], feats, seq)
    has_cont_df = "test_cont_df" in pack
    tc = _windows(pack["test_cont_df"] if has_cont_df else pack["test_df"], feats, seq)
    last_eval = bool(pack.get("last_window_eval", True))
    tl = _windows(pack["test_df"], feats, seq) if (last_eval and has_cont_df) else None

    if quick:
        tr, va, tc = _head(tr, 3000), _head(va, 1500), _head(tc, 2000)
        tl = _head(tl, 2000) if tl is not None else None

    f = tr[0].shape[-1]
    scaler = StandardScaler().fit(tr[0].reshape(-1, f))

    def sc(X):
        return scaler.transform(X.reshape(-1, f)).reshape(X.shape).astype(np.float32) if len(X) else X

    b = {
        "name": name,
        "features": feats,
        "feature_set": feat_key,
        "seq_len": seq,
        "cond_idx": [feats.index(c) for c in pack.get("cond_features", []) if c in feats],
        "X_train": sc(tr[0]), "y_train": tr[1], "u_train": tr[2],
        "X_val": sc(va[0]), "y_val": va[1], "u_val": va[2],
        "X_test": sc(tc[0]), "y_test": tc[1], "u_test": tc[2],
        "last_window_eval": last_eval,
        "last_from_cont": last_eval and not has_cont_df,  # last windows = end of continuous trajectories
        "note": pack.get("note"),
    }
    if tl is not None:
        b.update({"X_test_last": sc(tl[0]), "y_test_last": tl[1], "u_test_last": tl[2]})
    return b


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #


def last_index(units: np.ndarray) -> np.ndarray:
    units = np.asarray(units)
    if len(units) == 0:
        return np.zeros(0, dtype=int)
    return np.flatnonzero(np.r_[units[1:] != units[:-1], True])


def continuous_metrics(y, p, units) -> dict:
    y, p, units = np.asarray(y, float), np.asarray(p, float), np.asarray(units)
    d = p - y
    crit = y <= CRITICAL_RUL
    same = units[1:] == units[:-1]
    tce = np.abs(np.diff(p) - np.diff(y))[same]
    return {
        "rmse": float(np.sqrt(mean_squared_error(y, p))),
        "mae": float(mean_absolute_error(y, p)),
        "r2": float(r2_score(y, p)) if len(y) > 1 and np.ptp(y) > 0 else float("nan"),
        "phm_per_window": phm_score(y, p) / max(len(y), 1),
        "rmse_critical": float(np.sqrt(np.mean(d[crit] ** 2))) if crit.any() else float("nan"),
        "late_rate": float(np.mean(d > 0)),
        "tce": float(tce.mean()) if len(tce) else float("nan"),
        "n_windows": int(len(y)),
        "n_units": int(len(np.unique(units))),
    }


def last_metrics(y, p, units) -> dict:
    idx = last_index(units)
    y, p = np.asarray(y, float)[idx], np.asarray(p, float)[idx]
    return {
        "last_rmse": float(np.sqrt(np.mean((p - y) ** 2))),
        "last_mae": float(np.mean(np.abs(p - y))),
        "last_phm": phm_score(y, p),
        "last_n": int(len(y)),
    }


# --------------------------------------------------------------------------- #
# Evaluation of one (model, dataset)
# --------------------------------------------------------------------------- #


def _predict(model, X, units):
    p = np.maximum(np.asarray(model.predict(X), dtype=float).ravel(), 0.0)
    if hasattr(model, "postprocess"):
        p = np.maximum(model.postprocess(p, units), 0.0)
    return p


def evaluate(model, b: dict) -> dict:
    t0 = time.time()
    if getattr(model, "uses_units", False):
        model.fit(b["X_train"], b["y_train"], b["X_val"], b["y_val"],
                  units=b["u_train"], val_units=b["u_val"])
    else:
        model.fit(b["X_train"], b["y_train"], b["X_val"], b["y_val"])
    fit_s = time.time() - t0

    pv = _predict(model, b["X_val"], b["u_val"])
    pt = _predict(model, b["X_test"], b["u_test"])
    row = continuous_metrics(b["y_test"], pt, b["u_test"])
    row["val_rmse"] = float(np.sqrt(np.mean((pv - b["y_val"]) ** 2)))
    if b["last_window_eval"]:
        if b["last_from_cont"]:
            row.update(last_metrics(b["y_test"], pt, b["u_test"]))
        elif "X_test_last" in b:
            pl = _predict(model, b["X_test_last"], b["u_test_last"])
            row.update(last_metrics(b["y_test_last"], pl, b["u_test_last"]))
    row["fit_seconds"] = round(fit_s, 2)
    row["seconds"] = round(time.time() - t0, 2)
    if hasattr(model, "n_params"):
        row["n_params"] = int(model.n_params())
    if getattr(model, "alpha_", None) is not None:
        row["smoothing_alpha"] = float(model.alpha_)
    return row
