#!/usr/bin/env python3
"""
SCANIA val-cost tuning: fit on balanced train, choose the decision rule
(argmax vs. Bayes cost decode) on validation Total_cost, then evaluate once
on test (family metric = Total_cost).

Constant predictors ("Always-0", "Always-4") are reported alongside. Always-4
is a strong trivial baseline under the challenge cost matrix (test cost
49,671), so every model is also reported as % cost vs. Always-4.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parent
sys.path.insert(0, str(ROOT))
# Results root; run_pipeline.py points this at its --out folder (e.g. a mounted Google Drive)
RESULTS = Path(os.environ.get("CRB_RESULTS_DIR") or ROOT / "results")

from datasets.scania import load_scania  # noqa: E402
from metrics import SCANIA_COST, scania_metrics, scania_total_cost  # noqa: E402
from models import ConstantModel, LSTMModel, make_gradient_boosting, make_random_forest  # noqa: E402
from models.baselines import flatten_sequences  # noqa: E402
from training_utils import balance_resample  # noqa: E402

SEEDS = [42, 43, 44]
METRIC_KEYS = ("total_cost", "f1_macro", "accuracy", "recall_1", "recall_2", "recall_3", "recall_4")


def _predict_proba(model, X):
    if isinstance(model, LSTMModel):
        import torch

        model.model.eval()
        X = np.asarray(X, dtype=np.float32)
        probs = []
        with torch.no_grad():
            for i in range(0, len(X), 4096):
                logits = model.model(torch.from_numpy(X[i : i + 4096]).to(model.device))
                probs.append(torch.softmax(logits, dim=1).cpu().numpy())
        return np.concatenate(probs, axis=0)
    if isinstance(model, ConstantModel):
        return model.predict_proba(X)
    return model.estimator.predict_proba(flatten_sequences(X))


def _cost_decode(proba: np.ndarray, cost: np.ndarray) -> np.ndarray:
    """Bayes decision under the cost matrix [actual, predicted]: argmin_j sum_i P(i) cost[i,j]."""
    return (proba @ cost).argmin(axis=1).astype(int)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    args = ap.parse_args()

    out = RESULTS / "family_comparison"
    out.mkdir(parents=True, exist_ok=True)

    bundle = load_scania(
        WORKSPACE / "SCANIA-Component-X" / "data",
        seq_len=8,
        max_train_vehicles=3500,
        max_windows_per_vehicle=5,
    )
    X_va, y_va = bundle["X_val"], bundle["y_val"]
    X_te, y_te = bundle["X_test"], bundle["y_test"]

    rows = []
    for name, k in (("Always-0", 0), ("Always-4", 4)):
        m = scania_metrics(y_te, np.full(len(y_te), k))
        rows.append({"dataset": "SCANIA-Component-X", "model": name, "seed": None,
                     "decision_rule": "constant",
                     "val_cost_argmax": scania_total_cost(y_va, np.full(len(y_va), k)),
                     "val_cost_decode": None,
                     **{f"test_{key}": m[key] for key in METRIC_KEYS}})

    for seed in args.seeds:
        X_tr, y_tr = balance_resample(bundle["X_train"], bundle["y_train"], max_per_class=2000, seed=seed)
        models = {
            "LSTM": LSTMModel(task="classification", hidden_dim=64, num_layers=2, dropout=0.2,
                              epochs=20, patience=5, n_classes=5, class_weight="balanced", seed=seed),
            "RandomForest": make_random_forest("classification", seed=seed),
            "GradientBoosting": make_gradient_boosting("classification", seed=seed),
        }
        for name, model in models.items():
            print(f"-> {name} (seed={seed}): fit ...", flush=True)
            model.fit(X_tr, y_tr, X_va, y_va)
            pred_va = model.predict(X_va)
            pred_te = model.predict(X_te)
            p_va, p_te = _predict_proba(model, X_va), _predict_proba(model, X_te)
            dec_va, dec_te = _cost_decode(p_va, SCANIA_COST), _cost_decode(p_te, SCANIA_COST)
            base_va = scania_total_cost(y_va, pred_va)
            cs_va = scania_total_cost(y_va, dec_va)
            chosen = "cost_decode" if cs_va <= base_va else "argmax"  # chosen on validation
            final = scania_metrics(y_te, dec_te if chosen == "cost_decode" else pred_te)
            print(f"   val argmax={base_va:.0f} decode={cs_va:.0f} -> {chosen}; "
                  f"test cost={final['total_cost']:.0f} F1={final['f1_macro']:.4f}")
            rows.append({"dataset": "SCANIA-Component-X", "model": name, "seed": seed,
                         "decision_rule": chosen, "val_cost_argmax": float(base_va),
                         "val_cost_decode": float(cs_va),
                         **{f"test_{key}": final[key] for key in METRIC_KEYS}})
            pd.DataFrame(rows).to_csv(out / "scania_cost_tune.csv", index=False)

    df = pd.DataFrame(rows)
    df.to_csv(out / "scania_cost_tune.csv", index=False)
    summ = (df.groupby("model", as_index=False)
              .agg(test_cost_mean=("test_total_cost", "mean"), test_cost_std=("test_total_cost", "std"),
                   f1_macro_mean=("test_f1_macro", "mean"), accuracy_mean=("test_accuracy", "mean"),
                   recall_4_mean=("test_recall_4", "mean")))
    ref = float(summ.loc[summ["model"] == "Always-4", "test_cost_mean"].iloc[0])
    summ["cost_vs_always4_pct"] = 100.0 * (summ["test_cost_mean"] - ref) / ref
    summ = summ.sort_values("test_cost_mean")
    (out / "scania_cost_tune.json").write_text(
        json.dumps({"seeds": args.seeds, "always4_test_cost": ref,
                    "summary": summ.to_dict(orient="records"), "rows": rows}, indent=2, default=str))
    print("\n=== Mean test Total_cost over seeds (negative % = beats Always-4) ===")
    print(summ.to_string(index=False))
    print(f"Saved -> {out}")


if __name__ == "__main__":
    main()
