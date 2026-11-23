#!/usr/bin/env python3
"""
Within-family model comparison using each family's native metrics.

Families
--------
RUL  (CMAPSS FD001–FD004, PHM08, N-CMAPSS):
  primary = continuous RMSE (lower better); also MAE, R², PHM Score,
  and last-window ("benchmark") metrics where they are meaningful.
SCANIA:
  primary = Total_cost (lower better); also macro-F1, per-class recall, accuracy

Continuous sliding window
-------------------------
Train/val use every overlapping window (step 1). Test is scored two ways:
  * continuous  - every window over each test unit's history (primary)
  * last-window - the final window per test unit (classic CMAPSS benchmark;
                  skipped for N-CMAPSS where the last window is always RUL 0)

Every model runs over several seeds (``--seeds``, default 42 43 44) and is
reported as mean ± std. Trivial baselines are always included:
  RUL     -> "MeanRUL" (predicts the training-set mean)
  SCANIA  -> "Always-0" and "Always-4" (constant class predictions)

No cross-family leaderboard is produced.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parent
sys.path.insert(0, str(ROOT))

from datasets.ablation_data import (  # noqa: E402
    load_raw_cmapss,
    load_raw_n_cmapss,
    load_raw_phm08,
)
from datasets.scania import load_scania  # noqa: E402
from datasets.sequence_utils import build_windows_from_units  # noqa: E402
from metrics import rul_family_metrics, scania_metrics  # noqa: E402
from models import (  # noqa: E402
    ConstantModel,
    LSTMModel,
    make_gradient_boosting,
    make_linear,
    make_random_forest,
)
from training_utils import balance_resample  # noqa: E402

SEEDS = [42, 43, 44]
EPOCHS = 20
EPOCHS_HARD = 30  # multi-condition / hard RUL sets
PATIENCE = 5

# Multi-condition sets get stronger LSTM + all features
HARD_RUL = {"CMAPSS-FD002", "CMAPSS-FD004", "PHM08"}


def _scale_train_only(X_tr, others):
    n, t, f = X_tr.shape
    scaler = StandardScaler().fit(X_tr.reshape(-1, f))

    def tf(X):
        return scaler.transform(X.reshape(-1, f)).reshape(X.shape).astype(np.float32)

    return tf(X_tr), [tf(X) for X in others]


def pack_to_arrays(pack: dict):
    """Build scaled continuous sliding windows from a raw RUL pack."""
    hard = pack["name"] in HARD_RUL
    feat_key = "all" if hard and "all" in pack["feature_presets"] else pack["default_features"]
    feats = pack["feature_presets"][feat_key]
    seq = int(pack["default_seq_len"])
    stride = int(pack.get("stride_train", 1))

    def win(df, last_only):
        X, y, _ = build_windows_from_units(df, feats, "RUL", seq_len=seq, last_only=last_only)
        return X, y.astype(np.float32)

    X_tr, y_tr = win(pack["train_df"], False)
    X_va, y_va = win(pack["val_df"], False)
    X_tc, y_tc = win(pack.get("test_cont_df", pack["test_df"]), False)
    last_eval = bool(pack.get("last_window_eval", True))
    if last_eval:
        X_tl, y_tl = win(pack["test_df"], True)
    else:
        X_tl, y_tl = X_tc[:0], y_tc[:0]
    if stride > 1:
        X_tr, y_tr = X_tr[::stride], y_tr[::stride]
    X_tr, (X_va, X_tc, X_tl) = _scale_train_only(X_tr, [X_va, X_tc, X_tl])
    return {
        "name": pack["name"],
        "family": "RUL",
        "hard": hard,
        "X_train": X_tr,
        "y_train": y_tr,
        "X_val": X_va,
        "y_val": y_va,
        "X_test": X_tc,  # continuous (primary)
        "y_test": y_tc,
        "X_test_last": X_tl,
        "y_test_last": y_tl,
        "last_window_eval": last_eval,
        "note": pack.get("note"),
        "feature_set": feat_key,
    }


def make_models(task: str, n_classes: int | None = None, hard: bool = False, seed: int = 42):
    """Model registry. Add new architectures here to include them in every comparison."""
    epochs = EPOCHS_HARD if hard and task == "regression" else EPOCHS
    hidden = 128 if hard and task == "regression" else 64
    lstm_kw = dict(
        task=task,
        hidden_dim=hidden,
        num_layers=2,
        dropout=0.2,
        epochs=epochs,
        batch_size=64,
        patience=PATIENCE,
        n_classes=n_classes,
        seed=seed,
    )
    if task == "classification":
        lstm_kw["class_weight"] = "balanced"
    models = {
        "LSTM": LSTMModel(**lstm_kw),
        "RandomForest": make_random_forest(task, seed=seed),
        "GradientBoosting": make_gradient_boosting(task, seed=seed),
        "Linear": make_linear(task, seed=seed),
    }
    if task == "classification":
        models["Always-0"] = ConstantModel(task, constant=0, n_classes=n_classes)
        models["Always-4"] = ConstantModel(task, constant=4, n_classes=n_classes)
    else:
        models["MeanRUL"] = ConstantModel(task)
    return models


def evaluate_rul_dataset(bundle: dict, seed: int = 42, models: dict | None = None) -> list[dict]:
    rows = []
    print(f"\n===== RUL family :: {bundle['name']} (seed={seed}) =====")
    if bundle.get("note"):
        print(f"note: {bundle['note']}")
    print(
        f"windows train/val/test_cont/test_last: {len(bundle['y_train'])} / {len(bundle['y_val'])} / "
        f"{len(bundle['y_test'])} / {len(bundle['y_test_last'])}"
    )
    hard = bool(bundle.get("hard"))
    if models is None:
        models = make_models("regression", hard=hard, seed=seed)
    for name, model in models.items():
        print(f"  -> {name} ...", flush=True)
        t0 = time.time()
        model.fit(bundle["X_train"], bundle["y_train"], bundle["X_val"], bundle["y_val"])
        val_pred = np.maximum(model.predict(bundle["X_val"]), 0.0)
        pred = np.maximum(model.predict(bundle["X_test"]), 0.0)
        m = rul_family_metrics(bundle["y_test"], pred)
        m["val_rmse"] = rul_family_metrics(bundle["y_val"], val_pred)["rmse"]
        if bundle["last_window_eval"] and len(bundle["y_test_last"]):
            pl = np.maximum(model.predict(bundle["X_test_last"]), 0.0)
            ml = rul_family_metrics(bundle["y_test_last"], pl)
            for k in ("rmse", "mae", "r2", "phm_score", "n_test"):
                m[f"last_{k}"] = ml[k]
        m.update(
            {
                "dataset": bundle["name"],
                "model": name,
                "seed": seed,
                "seconds": round(time.time() - t0, 2),
                "status": "ok",
                "hard_mode": hard,
            }
        )
        last = f"  last-window RMSE={m['last_rmse']:.3f}" if "last_rmse" in m else ""
        print(
            f"     continuous RMSE={m['rmse']:.3f}  MAE={m['mae']:.3f}  R²={m['r2']:.3f}{last}"
        )
        rows.append(m)
    return rows


def load_scania_bundle(max_windows_per_vehicle: int | None = 5) -> dict:
    return load_scania(
        WORKSPACE / "SCANIA-Component-X" / "data",
        seq_len=8,
        max_train_vehicles=3500,
        max_windows_per_vehicle=max_windows_per_vehicle,
    )


def evaluate_scania(bundle: dict, seed: int = 42) -> list[dict]:
    print(f"\n===== SCANIA family :: Component-X classification (seed={seed}) =====")
    print("primary metric: Total_cost (IDA challenge cost matrix; lower is better)")
    X_tr_b, y_tr_b = balance_resample(
        bundle["X_train"], bundle["y_train"], max_per_class=2000, seed=seed
    )
    print(
        f"balanced train: {X_tr_b.shape} class counts={dict(zip(*np.unique(y_tr_b, return_counts=True)))}"
    )
    rows = []
    for name, model in make_models("classification", n_classes=5, seed=seed).items():
        print(f"  -> {name} ...", flush=True)
        t0 = time.time()
        model.fit(X_tr_b, y_tr_b, bundle["X_val"], bundle["y_val"])
        pred = model.predict(bundle["X_test"])
        m = scania_metrics(bundle["y_test"], pred)
        m.update(
            {
                "dataset": "SCANIA-Component-X",
                "model": name,
                "seed": seed,
                "seconds": round(time.time() - t0, 2),
                "status": "ok",
                "train_balanced": True,
            }
        )
        print(
            f"     Cost={m['total_cost']:.0f}  F1_macro={m['f1_macro']:.4f}  "
            f"Acc={m['accuracy']:.4f}  "
            f"R1-4=[{m['recall_1']:.2f},{m['recall_2']:.2f},{m['recall_3']:.2f},{m['recall_4']:.2f}]"
        )
        rows.append(m)
    return rows


def summarize_seeds(df: pd.DataFrame, primary: str, cols: list[str]) -> pd.DataFrame:
    """Mean ± std over seeds per (dataset, model), ranked by mean primary metric."""
    cols = [c for c in cols if c in df.columns]
    agg = {}
    for c in cols:
        agg[f"{c}_mean"] = (c, "mean")
        agg[f"{c}_std"] = (c, "std")
    agg["n_seeds"] = ("seed", "nunique")
    out = df.groupby(["dataset", "model"], as_index=False).agg(**agg)
    out["rank"] = out.groupby("dataset")[f"{primary}_mean"].rank(method="min", ascending=True)
    return out.sort_values(["dataset", "rank"]).reset_index(drop=True)


RUL_COLS = [
    "rmse", "mae", "r2", "phm_score", "tol_acc_10", "val_rmse",
    "last_rmse", "last_mae", "last_r2", "last_phm_score",
]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    ap.add_argument("--skip-scania", action="store_true")
    ap.add_argument(
        "--scania-windows-per-vehicle", type=int, default=5,
        help="train windows sampled per SCANIA vehicle (0 = every window; needs lots of RAM)",
    )
    args = ap.parse_args()

    out_dir = ROOT / "results" / "family_comparison"
    out_dir.mkdir(parents=True, exist_ok=True)

    rul_bundles = [pack_to_arrays(load_raw_cmapss(fd)) for fd in ("FD001", "FD002", "FD003", "FD004")]
    rul_bundles.append(pack_to_arrays(load_raw_phm08()))
    try:
        rul_bundles.append(pack_to_arrays(load_raw_n_cmapss()))
    except (FileNotFoundError, OSError) as exc:
        print(f"[skip] N-CMAPSS not available: {exc}")

    rul_rows = []
    for seed in args.seeds:
        for bundle in rul_bundles:
            rul_rows.extend(evaluate_rul_dataset(bundle, seed=seed))
            pd.DataFrame(rul_rows).to_csv(out_dir / "rul_family_metrics.csv", index=False)

    scania_rows = []
    if not args.skip_scania:
        per_vehicle = args.scania_windows_per_vehicle or None
        scania_bundle = load_scania_bundle(per_vehicle)
        for seed in args.seeds:
            scania_rows.extend(evaluate_scania(scania_bundle, seed=seed))
            pd.DataFrame(scania_rows).to_csv(out_dir / "scania_family_metrics.csv", index=False)

    rul_df = pd.DataFrame(rul_rows)
    rul_df.to_csv(out_dir / "rul_family_metrics.csv", index=False)
    rul_sum = summarize_seeds(rul_df, "rmse", RUL_COLS)
    rul_sum.to_csv(out_dir / "rul_family_summary.csv", index=False)

    summary = {
        "plan": "Compare models within each dataset/family using that family's metrics only.",
        "windowing": "continuous sliding window (stride 1); primary RUL metric = continuous test RMSE",
        "seeds": args.seeds,
        "families": {
            "RUL": {
                "datasets": sorted(rul_df["dataset"].unique().tolist()),
                "primary_metric": "rmse (continuous, mean over seeds)",
                "baseline": "MeanRUL",
                "winners": rul_sum[rul_sum["rank"] == 1].to_dict(orient="records"),
            },
        },
        "note": "No cross-family ranking is performed.",
    }

    if scania_rows:
        scania_df = pd.DataFrame(scania_rows)
        scania_df.to_csv(out_dir / "scania_family_metrics.csv", index=False)
        scania_sum = summarize_seeds(
            scania_df,
            "total_cost",
            ["total_cost", "f1_macro", "accuracy", "recall_1", "recall_2", "recall_3", "recall_4"],
        )
        ref = float(scania_sum.loc[scania_sum["model"] == "Always-4", "total_cost_mean"].iloc[0])
        scania_sum["cost_vs_always4_pct"] = 100.0 * (scania_sum["total_cost_mean"] - ref) / ref
        scania_sum.to_csv(out_dir / "scania_family_summary.csv", index=False)
        summary["families"]["SCANIA"] = {
            "datasets": ["SCANIA-Component-X"],
            "primary_metric": "total_cost (mean over seeds)",
            "baselines": ["Always-0", "Always-4"],
            "always4_cost": ref,
            "winners": scania_sum[scania_sum["rank"] == 1].to_dict(orient="records"),
        }
        print("\n========== SCANIA (mean over seeds; lowest Total_cost first) ==========")
        print(
            scania_sum[
                ["model", "total_cost_mean", "total_cost_std", "f1_macro_mean", "cost_vs_always4_pct"]
            ].to_string(index=False)
        )

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))

    print("\n========== RUL FAMILY (continuous RMSE, mean over seeds) ==========")
    show = [c for c in ["dataset", "model", "rmse_mean", "rmse_std", "r2_mean", "last_rmse_mean", "rank"] if c in rul_sum.columns]
    print(rul_sum[show].to_string(index=False))
    print(f"\nSaved -> {out_dir}")


if __name__ == "__main__":
    main()
