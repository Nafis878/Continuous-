#!/usr/bin/env python3
"""
Multi-regime RUL: cluster operating conditions (op1/op2/op3) on TRAIN only,
then z-score sensors within each cluster before windowing/modeling.
Compare vs plain scaling on FD002/FD004.

Continuous sliding window (stride 1). Primary metric = continuous test RMSE;
last-window RMSE also reported. The winner per dataset is chosen by mean
VALIDATION RMSE over seeds (test is never used for selection).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from datasets.ablation_data import FEAT_ALL, load_raw_cmapss  # noqa: E402
from datasets.sequence_utils import build_windows_from_units  # noqa: E402
from metrics import rul_family_metrics  # noqa: E402
from models import ConstantModel, LSTMModel, make_gradient_boosting, make_linear, make_random_forest  # noqa: E402

SEEDS = [42, 43, 44]
SPLIT_SEED = 42  # clustering / scalers are data prep, kept fixed across model seeds
OP_COLS = ["op1", "op2", "op3"]
SENSOR_COLS = [c for c in FEAT_ALL if c.startswith("s")]


def _condition_normalize(train_df, val_df, test_df, n_clusters: int = 6):
    """Fit KMeans + per-cluster sensor scalers on TRAIN units only."""
    km = KMeans(n_clusters=n_clusters, random_state=SPLIT_SEED, n_init=10)
    op_tr = train_df[OP_COLS].to_numpy(dtype=np.float64)
    op_scaler = StandardScaler().fit(op_tr)
    km.fit(op_scaler.transform(op_tr))

    def _assign(df):
        labs = km.predict(op_scaler.transform(df[OP_COLS].to_numpy(dtype=np.float64)))
        out = df.copy()
        out["_cond"] = labs
        return out

    train_df, val_df, test_df = _assign(train_df), _assign(val_df), _assign(test_df)

    scalers = {}
    for c in range(n_clusters):
        mask = train_df["_cond"].to_numpy() == c
        if mask.sum() < 10:
            continue
        scalers[c] = StandardScaler().fit(train_df.loc[mask, SENSOR_COLS].to_numpy(dtype=np.float64))
    global_sc = StandardScaler().fit(train_df[SENSOR_COLS].to_numpy(dtype=np.float64))

    def _apply(df):
        out = df.copy()
        arr = out[SENSOR_COLS].to_numpy(dtype=np.float64)
        for c, sc in scalers.items():
            m = out["_cond"].to_numpy() == c
            if m.any():
                arr[m] = sc.transform(arr[m])
        missing = ~np.isin(out["_cond"].to_numpy(), list(scalers.keys()))
        if missing.any():
            arr[missing] = global_sc.transform(arr[missing])
        out[SENSOR_COLS] = arr
        out[OP_COLS] = op_scaler.transform(out[OP_COLS].to_numpy(dtype=np.float64))
        return out

    return _apply(train_df), _apply(val_df), _apply(test_df)


def _windows(train_df, val_df, test_df, feats, seq_len):
    X_tr, y_tr, _ = build_windows_from_units(train_df, feats, "RUL", seq_len=seq_len)
    X_va, y_va, _ = build_windows_from_units(val_df, feats, "RUL", seq_len=seq_len)
    X_tc, y_tc, _ = build_windows_from_units(test_df, feats, "RUL", seq_len=seq_len)
    X_tl, y_tl, _ = build_windows_from_units(test_df, feats, "RUL", seq_len=seq_len, last_only=True)
    return [X_tr, y_tr, X_va, y_va, X_tc, y_tc, X_tl, y_tl]


def _plain(train_df, val_df, test_df, feats, seq_len=30):
    w = _windows(train_df, val_df, test_df, feats, seq_len)
    f = w[0].shape[-1]
    sc = StandardScaler().fit(w[0].reshape(-1, f))
    for i in (0, 2, 4, 6):
        w[i] = sc.transform(w[i].reshape(-1, f)).reshape(w[i].shape)
    return [a.astype(np.float32) for a in w]


def _cond(train_df, val_df, test_df, feats, seq_len=30):
    tr, va, te = _condition_normalize(train_df, val_df, test_df, n_clusters=6)
    return [a.astype(np.float32) for a in _windows(tr, va, te, feats, seq_len)]


def _models(seed):
    return {
        "LSTM": LSTMModel(task="regression", hidden_dim=128, num_layers=2, dropout=0.2,
                          epochs=25, patience=5, seed=seed),
        "RandomForest": make_random_forest("regression", seed=seed),
        "GradientBoosting": make_gradient_boosting("regression", seed=seed),
        "Linear": make_linear("regression", seed=seed),
        "MeanRUL": ConstantModel("regression"),
    }


def run_fd(fd: str, seeds):
    pack = load_raw_cmapss(fd)
    rows = []
    for mode, builder in (("plain", _plain), ("cond_cluster", _cond)):
        X_tr, y_tr, X_va, y_va, X_tc, y_tc, X_tl, y_tl = builder(
            pack["train_df"], pack["val_df"], pack["test_df"], FEAT_ALL, seq_len=30
        )
        for seed in seeds:
            print(f"\n=== {pack['name']} :: {mode} :: seed={seed} ===", flush=True)
            for name, model in _models(seed).items():
                t0 = time.time()
                model.fit(X_tr, y_tr, X_va, y_va)
                m = rul_family_metrics(y_tc, np.maximum(model.predict(X_tc), 0.0))
                m["val_rmse"] = rul_family_metrics(y_va, np.maximum(model.predict(X_va), 0.0))["rmse"]
                m["last_rmse"] = rul_family_metrics(y_tl, np.maximum(model.predict(X_tl), 0.0))["rmse"]
                m.update({"dataset": pack["name"], "mode": mode, "model": name, "seed": seed,
                          "seconds": round(time.time() - t0, 2)})
                print(f"  {name:16s} val={m['val_rmse']:.3f}  cont={m['rmse']:.3f}  last={m['last_rmse']:.3f}")
                rows.append(m)
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    args = ap.parse_args()

    out = ROOT / "results" / "family_comparison"
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for fd in ("FD002", "FD004"):
        rows.extend(run_fd(fd, args.seeds))
        pd.DataFrame(rows).to_csv(out / "condition_rul_metrics.csv", index=False)
    df = pd.DataFrame(rows)
    df.to_csv(out / "condition_rul_metrics.csv", index=False)

    agg = (
        df.groupby(["dataset", "mode", "model"], as_index=False)
        .agg(val_rmse=("val_rmse", "mean"), rmse_mean=("rmse", "mean"), rmse_std=("rmse", "std"),
             last_rmse_mean=("last_rmse", "mean"), r2_mean=("r2", "mean"), phm_mean=("phm_score", "mean"))
    )
    agg.to_csv(out / "condition_rul_summary.csv", index=False)

    winners, deltas = [], []
    for ds, g in agg.groupby("dataset"):
        best = g.sort_values("val_rmse").iloc[0]  # selected on validation only
        winners.append(best.to_dict())
        plain = g[g["mode"] == "plain"].sort_values("val_rmse").iloc[0]
        cond = g[g["mode"] == "cond_cluster"].sort_values("val_rmse").iloc[0]
        deltas.append({
            "dataset": ds,
            "plain_selected": f"{plain.model} test RMSE={plain.rmse_mean:.3f}",
            "cond_selected": f"{cond.model} test RMSE={cond.rmse_mean:.3f}",
            "delta_rmse_cond_minus_plain": float(cond.rmse_mean - plain.rmse_mean),
        })
    summary = {"selection": "lowest mean validation RMSE", "seeds": args.seeds,
               "winners": winners, "cond_vs_plain": deltas}
    (out / "condition_rul_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print("\n=== Selected on validation ===")
    print(pd.DataFrame(winners)[["dataset", "mode", "model", "val_rmse", "rmse_mean", "rmse_std", "last_rmse_mean"]].to_string(index=False))
    print(pd.DataFrame(deltas).to_string(index=False))
    print(f"Saved -> {out}")


if __name__ == "__main__":
    main()
