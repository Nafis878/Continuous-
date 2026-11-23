#!/usr/bin/env python3
"""
One-factor-at-a-time LSTM ablation study (leakage-safe, reproducible).

Ablates vs a fixed baseline:
  - normalization (on/off; scaler fit on TRAIN windows only)
  - sliding window length
  - feature set
  - dropout
  - architecture (depth / width)

Metrics: RMSE, MAE, R², PHM Score.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from datasets.ablation_data import load_all_raw_datasets  # noqa: E402
from datasets.sequence_utils import build_windows_from_units  # noqa: E402
from metrics import regression_metrics_full  # noqa: E402
from models.lstm import LSTMModel  # noqa: E402

SEED = 42
EPOCHS = 15
PATIENCE = 4
BATCH = 64
LR = 1e-3

# Baseline configuration
BASELINE = {
    "normalize": True,
    "seq_len": None,  # dataset default
    "features": "selected",
    "dropout": 0.2,
    "hidden_dim": 64,
    "num_layers": 2,
}


def build_windows_pack(pack: dict, feature_cols: list[str], seq_len: int):
    """Build continuous sliding windows (step 1) for train/val/test.

    Test is scored continuously: every window over each test unit's history
    (full trajectories via test_cont_df when the pack provides them).
    """
    stride = int(pack.get("stride_train", 1))
    max_per = pack.get("max_windows_per_vehicle")

    def _windows(df, last_only: bool):
        X, y, units = build_windows_from_units(
            df,
            feature_cols,
            "RUL",
            seq_len=seq_len,
            unit_col="unit",
            cycle_col="cycle",
            last_only=last_only,
        )
        return X, y, units

    # SCANIA val vehicles are pre-cut before repair; validate on last window per cut
    val_last_only = max_per is not None
    X_tr, y_tr, u_tr = _windows(pack["train_df"], last_only=False)
    X_va, y_va, u_va = _windows(pack["val_df"], last_only=val_last_only)
    X_te, y_te, u_te = _windows(pack.get("test_cont_df", pack["test_df"]), last_only=False)

    # Optional per-vehicle train window subsample (SCANIA) — by unit, after windowing
    if max_per is not None:
        rng = np.random.default_rng(SEED)
        keep = []
        for u in np.unique(u_tr):
            idx = np.flatnonzero(u_tr == u)
            if len(idx) > max_per:
                idx = rng.choice(idx, size=max_per, replace=False)
            keep.extend(idx.tolist())
        keep = np.asarray(sorted(keep))
        X_tr, y_tr = X_tr[keep], y_tr[keep]

    if stride > 1:
        X_tr, y_tr = X_tr[::stride], y_tr[::stride]

    return (
        X_tr.astype(np.float32),
        y_tr.astype(np.float32),
        X_va.astype(np.float32),
        y_va.astype(np.float32),
        X_te.astype(np.float32),
        y_te.astype(np.float32),
    )


def apply_normalization(X_tr, X_va, X_te, enabled: bool):
    """Fit scaler on TRAIN only — no leakage into val/test."""
    if not enabled:
        return X_tr, X_va, X_te
    n, t, f = X_tr.shape
    scaler = StandardScaler()
    scaler.fit(X_tr.reshape(-1, f))
    X_tr = scaler.transform(X_tr.reshape(-1, f)).reshape(n, t, f).astype(np.float32)
    X_va = (
        scaler.transform(X_va.reshape(-1, f)).reshape(X_va.shape).astype(np.float32)
    )
    X_te = (
        scaler.transform(X_te.reshape(-1, f)).reshape(X_te.shape).astype(np.float32)
    )
    return X_tr, X_va, X_te


def ablation_configs(pack: dict) -> list[dict]:
    """One-factor-at-a-time configs; first entry is baseline."""
    default_seq = int(pack["default_seq_len"])
    configs = [
        {
            "ablation": "baseline",
            "component": "none",
            "normalize": True,
            "seq_len": default_seq,
            "features": "selected",
            "dropout": 0.2,
            "hidden_dim": 64,
            "num_layers": 2,
        },
        # normalization
        {
            "ablation": "no_normalization",
            "component": "normalization",
            "normalize": False,
            "seq_len": default_seq,
            "features": "selected",
            "dropout": 0.2,
            "hidden_dim": 64,
            "num_layers": 2,
        },
        # sliding windows
        {
            "ablation": "window_len_1",
            "component": "sliding_windows",
            "normalize": True,
            "seq_len": 1,
            "features": "selected",
            "dropout": 0.2,
            "hidden_dim": 64,
            "num_layers": 2,
        },
        {
            "ablation": "window_short",
            "component": "sliding_windows",
            "normalize": True,
            "seq_len": max(5, default_seq // 3),
            "features": "selected",
            "dropout": 0.2,
            "hidden_dim": 64,
            "num_layers": 2,
        },
        {
            "ablation": "window_long",
            "component": "sliding_windows",
            "normalize": True,
            "seq_len": min(60, default_seq * 2),
            "features": "selected",
            "dropout": 0.2,
            "hidden_dim": 64,
            "num_layers": 2,
        },
        # features
        {
            "ablation": "features_all",
            "component": "features",
            "normalize": True,
            "seq_len": default_seq,
            "features": "all",
            "dropout": 0.2,
            "hidden_dim": 64,
            "num_layers": 2,
        },
        {
            "ablation": "features_sensors_only",
            "component": "features",
            "normalize": True,
            "seq_len": default_seq,
            "features": "sensors_only",
            "dropout": 0.2,
            "hidden_dim": 64,
            "num_layers": 2,
        },
        {
            "ablation": "features_ops_only",
            "component": "features",
            "normalize": True,
            "seq_len": default_seq,
            "features": "ops_only",
            "dropout": 0.2,
            "hidden_dim": 64,
            "num_layers": 2,
        },
        {
            "ablation": "features_reduced",
            "component": "features",
            "normalize": True,
            "seq_len": default_seq,
            "features": "reduced",
            "dropout": 0.2,
            "hidden_dim": 64,
            "num_layers": 2,
        },
        # dropout
        {
            "ablation": "dropout_0",
            "component": "dropout",
            "normalize": True,
            "seq_len": default_seq,
            "features": "selected",
            "dropout": 0.0,
            "hidden_dim": 64,
            "num_layers": 2,
        },
        {
            "ablation": "dropout_0.5",
            "component": "dropout",
            "normalize": True,
            "seq_len": default_seq,
            "features": "selected",
            "dropout": 0.5,
            "hidden_dim": 64,
            "num_layers": 2,
        },
        # architecture
        {
            "ablation": "arch_1layer",
            "component": "architecture",
            "normalize": True,
            "seq_len": default_seq,
            "features": "selected",
            "dropout": 0.2,
            "hidden_dim": 64,
            "num_layers": 1,
        },
        {
            "ablation": "arch_hidden32",
            "component": "architecture",
            "normalize": True,
            "seq_len": default_seq,
            "features": "selected",
            "dropout": 0.2,
            "hidden_dim": 32,
            "num_layers": 2,
        },
        {
            "ablation": "arch_hidden128",
            "component": "architecture",
            "normalize": True,
            "seq_len": default_seq,
            "features": "selected",
            "dropout": 0.2,
            "hidden_dim": 128,
            "num_layers": 2,
        },
    ]
    return configs


def run_one(pack: dict, cfg: dict) -> dict:
    feat_name = cfg["features"]
    feature_cols = pack["feature_presets"][feat_name]
    # Guard empty feature sets
    feature_cols = [c for c in feature_cols if c in pack["train_df"].columns]
    if not feature_cols:
        return {
            "dataset": pack["name"],
            "ablation": cfg["ablation"],
            "component": cfg["component"],
            "status": "skipped_empty_features",
        }

    X_tr, y_tr, X_va, y_va, X_te, y_te = build_windows_pack(
        pack, feature_cols, seq_len=int(cfg["seq_len"])
    )
    X_tr, X_va, X_te = apply_normalization(X_tr, X_va, X_te, enabled=bool(cfg["normalize"]))

    model = LSTMModel(
        task="regression",
        hidden_dim=int(cfg["hidden_dim"]),
        num_layers=int(cfg["num_layers"]),
        dropout=float(cfg["dropout"]),
        lr=LR,
        epochs=EPOCHS,
        batch_size=BATCH,
        patience=PATIENCE,
        seed=SEED,
    )
    t0 = time.time()
    model.fit(X_tr, y_tr, X_va, y_va)
    pred = model.predict(X_te)
    # Clip negative RUL predictions for scoring stability
    pred = np.maximum(pred, 0.0)
    metrics = regression_metrics_full(y_te, pred)
    metrics.update(
        {
            "dataset": pack["name"],
            "ablation": cfg["ablation"],
            "component": cfg["component"],
            "normalize": cfg["normalize"],
            "seq_len": cfg["seq_len"],
            "features": cfg["features"],
            "dropout": cfg["dropout"],
            "hidden_dim": cfg["hidden_dim"],
            "num_layers": cfg["num_layers"],
            "n_train": int(len(y_tr)),
            "n_features": int(X_tr.shape[-1]),
            "seconds": round(time.time() - t0, 2),
            "status": "ok",
        }
    )
    return metrics


def summarize(df: pd.DataFrame) -> dict:
    """Per-component impact vs baseline using ΔRMSE (positive = worse)."""
    ok = df[df["status"] == "ok"].copy()
    impact = []
    for dataset, g in ok.groupby("dataset"):
        base = g[g["ablation"] == "baseline"]
        if base.empty:
            continue
        b = base.iloc[0]
        for _, row in g.iterrows():
            if row["ablation"] == "baseline":
                continue
            impact.append(
                {
                    "dataset": dataset,
                    "ablation": row["ablation"],
                    "component": row["component"],
                    "rmse": row["rmse"],
                    "baseline_rmse": b["rmse"],
                    "delta_rmse": float(row["rmse"] - b["rmse"]),
                    "mae": row["mae"],
                    "baseline_mae": b["mae"],
                    "delta_mae": float(row["mae"] - b["mae"]),
                    "r2": row["r2"],
                    "baseline_r2": b["r2"],
                    "delta_r2": float(row["r2"] - b["r2"]),
                    "phm_score": row["phm_score"],
                    "baseline_phm": b["phm_score"],
                    "delta_phm": float(row["phm_score"] - b["phm_score"]),
                }
            )
    impact_df = pd.DataFrame(impact)
    if impact_df.empty:
        return {"error": "no impact rows"}

    # Mean |ΔRMSE| by component across datasets (importance)
    comp = (
        impact_df.groupby("component")
        .agg(
            mean_abs_delta_rmse=("delta_rmse", lambda s: float(np.mean(np.abs(s)))),
            mean_delta_rmse=("delta_rmse", "mean"),
            mean_abs_delta_mae=("delta_mae", lambda s: float(np.mean(np.abs(s)))),
            mean_delta_r2=("delta_r2", "mean"),
            mean_abs_delta_phm=("delta_phm", lambda s: float(np.mean(np.abs(s)))),
            n=("delta_rmse", "count"),
        )
        .sort_values("mean_abs_delta_rmse", ascending=False)
        .reset_index()
    )

    # Worst regressions from baseline
    worst = impact_df.sort_values("delta_rmse", ascending=False).head(15)
    best = impact_df.sort_values("delta_rmse", ascending=True).head(10)

    return {
        "component_importance_by_mean_abs_delta_rmse": comp.to_dict(orient="records"),
        "largest_degradations": worst.to_dict(orient="records"),
        "largest_improvements": best.to_dict(orient="records"),
    }


def main():
    out_dir = ROOT / "results" / "ablation"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading raw datasets (fixed unit splits, seed=42)...")
    packs = load_all_raw_datasets()
    for p in packs:
        note = p.get("note", "")
        print(
            f"  {p['name']}: train_units={p['train_df']['unit'].nunique()} "
            f"val={p['val_df']['unit'].nunique()} test={p['test_df']['unit'].nunique()} {note}"
        )

    rows = []
    for pack in packs:
        configs = ablation_configs(pack)
        print(f"\n===== {pack['name']} ({len(configs)} ablations) =====")
        for cfg in configs:
            print(f"  -> {cfg['ablation']} ...", flush=True)
            try:
                row = run_one(pack, cfg)
            except Exception as exc:
                row = {
                    "dataset": pack["name"],
                    "ablation": cfg["ablation"],
                    "component": cfg["component"],
                    "status": f"error: {exc}",
                }
                print(f"     FAILED: {exc}")
            else:
                if row.get("status") == "ok":
                    print(
                        f"     RMSE={row['rmse']:.3f} MAE={row['mae']:.3f} "
                        f"R2={row['r2']:.3f} PHM={row['phm_score']:.1f} ({row['seconds']}s)"
                    )
                else:
                    print(f"     {row.get('status')}")
            rows.append(row)
            # incremental save
            pd.DataFrame(rows).to_csv(out_dir / "ablation_metrics.csv", index=False)

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "ablation_metrics.csv", index=False)
    (out_dir / "ablation_metrics.json").write_text(json.dumps(rows, indent=2, default=str))

    summary = summarize(df)
    (out_dir / "ablation_summary.json").write_text(json.dumps(summary, indent=2, default=str))

    print("\n========== COMPONENT IMPORTANCE (mean |ΔRMSE| vs baseline) ==========")
    for rec in summary.get("component_importance_by_mean_abs_delta_rmse", []):
        print(
            f"  {rec['component']:18s}  mean|ΔRMSE|={rec['mean_abs_delta_rmse']:.3f}  "
            f"meanΔRMSE={rec['mean_delta_rmse']:.3f}  meanΔR²={rec['mean_delta_r2']:.3f}"
        )
    print(f"\nSaved -> {out_dir}")


if __name__ == "__main__":
    main()
