#!/usr/bin/env python3
"""
Continuous RUL Benchmark (CRB) runner.

Runs every (dataset, model, seed) under the CRB protocol (see benchmark.py /
BENCHMARK.md), appends each result to runs.csv immediately, and rebuilds the
leaderboards after every run. Re-running the same command resumes: finished
(dataset, model, seed) runs are skipped — safe for Colab disconnects.

Examples
--------
  python ml_pipeline/run_benchmark.py --quick                      # 1-epoch smoke test
  python ml_pipeline/run_benchmark.py                              # full benchmark, 3 seeds
  python ml_pipeline/run_benchmark.py --models CATT+TCT TCN --datasets FD001 FD004
  python ml_pipeline/run_benchmark.py --out results/crb             # custom output dir (resumable)
"""

from __future__ import annotations

import argparse
import os
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
for _stream in (sys.stdout, sys.stderr):  # never crash on console encoding (e.g. Windows cp1252)
    try:
        _stream.reconfigure(errors="replace")
    except AttributeError:
        pass

import torch  # noqa: E402

from benchmark import DATASETS, build_bundle, evaluate  # noqa: E402
from models import ConstantModel, make_gradient_boosting, make_linear, make_random_forest  # noqa: E402
from models.deep import DeepRUL  # noqa: E402

SEEDS = [42, 43, 44]
EPOCHS = 40
PATIENCE = 8

# TCT (Trajectory-Consistent Training) hyper-parameters — fixed for all datasets
TCT = dict(consistency=1.0, max_offset=10, late_penalty=0.5, smoothing="auto")

BASELINES = ["MeanRUL", "Ridge", "HistGB", "LSTM", "BiLSTM-Attn", "CNN1D", "CNN-LSTM", "TCN", "Transformer"]
PROPOSED = ["CATT", "CATT+TCT"]          # architecture alone, architecture + method
METHOD_ABLATION = ["TCN+TCT"]           # TCT applied to a strong baseline
DEFAULT_MODELS = BASELINES + PROPOSED + METHOD_ABLATION
OPTIONAL_MODELS = ["RandomForest"]      # slow on Colab's 2 CPU cores
# TCT component ablation (run into a separate output folder, compared against CATT / CATT+TCT):
# each variant changes exactly one thing relative to CATT+TCT.
TCT_ABLATION = {
    "CATT+TCT-noConsistency": dict(TCT, consistency=0.0),   # no trajectory-consistency loss
    "CATT+TCT-noLatePenalty": dict(TCT, late_penalty=0.0),  # symmetric loss
    "CATT+TCT-noSmoothing": dict(TCT, smoothing=None),      # no causal smoothing at inference
    "CATT+TCT-lambda0.3": dict(TCT, consistency=0.3),       # weaker consistency weight
    "CATT+TCT-lambda3": dict(TCT, consistency=3.0),         # stronger consistency weight
}


def make_model(name: str, seed: int, bundle: dict, epochs: int, patience: int):
    deep = dict(epochs=epochs, patience=patience, seed=seed)
    cond = {"cond_idx": bundle["cond_idx"]}
    registry = {
        "MeanRUL": lambda: ConstantModel("regression"),
        "Ridge": lambda: make_linear("regression", seed=seed),
        "HistGB": lambda: make_gradient_boosting("regression", seed=seed),
        "RandomForest": lambda: make_random_forest("regression", seed=seed),
        "LSTM": lambda: DeepRUL("lstm", {"hidden": 64, "layers": 2}, **deep),
        "BiLSTM-Attn": lambda: DeepRUL("lstm", {"hidden": 64, "layers": 2, "bidirectional": True, "attention": True}, **deep),
        "CNN1D": lambda: DeepRUL("cnn1d", {}, **deep),
        "CNN-LSTM": lambda: DeepRUL("cnn_lstm", {}, **deep),
        "TCN": lambda: DeepRUL("tcn", {}, **deep),
        "Transformer": lambda: DeepRUL("transformer", {}, **deep),
        "CATT": lambda: DeepRUL("catt", cond, **deep),
        "CATT+TCT": lambda: DeepRUL("catt", cond, **deep, **TCT),
        "TCN+TCT": lambda: DeepRUL("tcn", {}, **deep, **TCT),
    }
    for abl_name, abl_cfg in TCT_ABLATION.items():
        registry[abl_name] = (lambda cfg=abl_cfg: DeepRUL("catt", cond, **deep, **cfg))
    if name not in registry:
        raise SystemExit(f"Unknown model {name!r}. Available: {', '.join(registry)}")
    return registry[name]()


# --------------------------------------------------------------------------- #
# Leaderboards
# --------------------------------------------------------------------------- #

METRICS = ["rmse", "mae", "r2", "rmse_critical", "tce", "phm_per_window", "last_rmse", "last_phm", "val_rmse"]


def _fmt(m, s, nd=2):
    if pd.isna(m):
        return "–"
    return f"{m:.{nd}f}" if pd.isna(s) else f"{m:.{nd}f} ± {s:.{nd}f}"


def build_leaderboards(runs: pd.DataFrame, out: Path, quick: bool):
    ok = runs[runs["status"] == "ok"].copy()
    if ok.empty:
        return
    cols = [c for c in METRICS if c in ok.columns]
    agg = ok.groupby(["dataset", "model"])[cols].agg(["mean", "std"])
    agg.columns = [f"{a}_{b}" for a, b in agg.columns]
    agg["n_seeds"] = ok.groupby(["dataset", "model"])["seed"].nunique()
    agg = agg.reset_index()
    agg["rank"] = agg.groupby("dataset")["rmse_mean"].rank(method="min")
    agg = agg.sort_values(["dataset", "rank"])
    agg.to_csv(out / "leaderboard.csv", index=False)

    # average rank over datasets where the model ran (within the RUL family)
    avg = (agg.groupby("model").agg(avg_rank=("rank", "mean"), datasets=("dataset", "nunique"),
                                    mean_rmse=("rmse_mean", "mean")).sort_values("avg_rank").reset_index())
    avg.to_csv(out / "leaderboard_avg_rank.csv", index=False)

    lines = ["# Continuous RUL Benchmark (CRB) — leaderboard", ""]
    if quick:
        lines += ["> **QUICK MODE (smoke test, 1 epoch, subsampled) — numbers are not meaningful.**", ""]
    lines += [
        "Primary metric: **continuous RMSE** over every test window (mean ± std over seeds). "
        "RMSE≤30: RMSE where true RUL ≤ 30. TCE: trajectory consistency error "
        "(mean |Δpred − Δtrue| between consecutive windows). Last: classic last-window RMSE / PHM score.",
        "",
        "## Average rank across datasets", "",
        "| Model | Avg rank | Datasets | Mean RMSE |", "|---|---|---|---|",
    ]
    for _, r in avg.iterrows():
        lines.append(f"| {r['model']} | {r['avg_rank']:.2f} | {int(r['datasets'])} | {r['mean_rmse']:.2f} |")
    for ds, g in agg.groupby("dataset", sort=True):
        lines += ["", f"## {ds}", "",
                  "| # | Model | RMSE | MAE | R² | RMSE≤30 | TCE | Last RMSE | Last PHM | Seeds |",
                  "|---|---|---|---|---|---|---|---|---|---|"]
        for _, r in g.iterrows():
            lines.append(
                f"| {int(r['rank'])} | {r['model']} | {_fmt(r['rmse_mean'], r['rmse_std'])} | "
                f"{_fmt(r['mae_mean'], r['mae_std'])} | {_fmt(r['r2_mean'], r['r2_std'], 3)} | "
                f"{_fmt(r.get('rmse_critical_mean'), r.get('rmse_critical_std'))} | "
                f"{_fmt(r.get('tce_mean'), r.get('tce_std'))} | "
                f"{_fmt(r.get('last_rmse_mean', np.nan), r.get('last_rmse_std', np.nan))} | "
                f"{_fmt(r.get('last_phm_mean', np.nan), r.get('last_phm_std', np.nan), 0)} | {int(r['n_seeds'])} |"
            )
    (out / "LEADERBOARD.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="+", default=list(DATASETS), help=f"subset of {list(DATASETS)}")
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS,
                    help=f"default: {DEFAULT_MODELS}; optional: {OPTIONAL_MODELS}")
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--patience", type=int, default=PATIENCE)
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "benchmark")
    ap.add_argument("--quick", action="store_true", help="smoke test: 1 epoch, subsampled windows, 1 seed")
    ap.add_argument("--fresh", action="store_true", help="ignore existing runs.csv (do not resume)")
    args = ap.parse_args()

    if args.quick:
        args.epochs, args.patience = 1, 1
        if args.seeds == SEEDS:
            args.seeds = [42]
    unknown = [d for d in args.datasets if d not in DATASETS]
    if unknown:
        raise SystemExit(f"Unknown dataset(s) {unknown}. Available: {list(DATASETS)}")

    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    runs_path = out / "runs.csv"
    runs = pd.read_csv(runs_path) if runs_path.exists() and not args.fresh else pd.DataFrame()
    done = set()
    if not runs.empty:
        runs = runs[runs["status"] == "ok"].copy()  # failed runs are retried
        done = set(zip(runs["dataset"], runs["model"], runs["seed"].astype(int)))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    gpu = torch.cuda.get_device_name(0) if device == "cuda" else "none"
    print(f"CRB | device={device} ({gpu}) | torch {torch.__version__} | seeds={args.seeds} | "
          f"epochs={args.epochs} | quick={args.quick}")
    print(f"models: {args.models}")
    print(f"output: {out}  (resuming {len(done)} finished runs)" if done else f"output: {out}")
    (out / "config.json").write_text(json.dumps({
        "datasets": args.datasets, "models": args.models, "seeds": args.seeds, "epochs": args.epochs,
        "patience": args.patience, "quick": args.quick, "tct": TCT, "device": device, "gpu": gpu,
        "torch": torch.__version__,
    }, indent=2))

    t_start = time.time()
    rows = runs.to_dict("records") if not runs.empty else []
    for ds in args.datasets:
        todo = [(m, s) for m in args.models for s in args.seeds if (ds, m, s) not in done]
        if not todo:
            print(f"\n=== {ds}: all runs already done ===")
            continue
        try:
            bundle = build_bundle(DATASETS[ds](), quick=args.quick)
        except (FileNotFoundError, OSError) as exc:
            print(f"\n=== {ds}: SKIPPED ({exc}) ===")
            continue
        print(f"\n=== {ds} ({bundle['name']}) | features={bundle['feature_set']}({len(bundle['features'])}) "
              f"seq={bundle['seq_len']} | windows train/val/test = {len(bundle['y_train'])}/"
              f"{len(bundle['y_val'])}/{len(bundle['y_test'])} ===")
        for model_name, seed in todo:
            t0 = time.time()
            print(f"  {model_name:12s} seed={seed} ...", end=" ", flush=True)
            row = {"dataset": ds, "model": model_name, "seed": seed}
            try:
                model = make_model(model_name, seed, bundle, args.epochs, args.patience)
                row.update(evaluate(model, bundle))
                row["backend"] = getattr(model, "backend", "torch" if isinstance(model, DeepRUL) else "sklearn")
                row["status"] = "ok"
                last = f" last={row['last_rmse']:.2f}" if "last_rmse" in row else ""
                print(f"RMSE={row['rmse']:.2f} <=30:{row['rmse_critical']:.2f} TCE={row['tce']:.2f}{last} "
                      f"({time.time() - t0:.0f}s)", flush=True)
            except Exception as exc:  # keep the benchmark going; record the failure
                row["status"] = f"error: {type(exc).__name__}: {exc}"
                print(f"FAILED: {exc}", flush=True)
                traceback.print_exc()
            finally:
                if device == "cuda":
                    torch.cuda.empty_cache()
            rows.append(row)
            tmp = runs_path.with_name("runs.csv.tmp")  # atomic replace: never a half-written runs.csv
            pd.DataFrame(rows).to_csv(tmp, index=False)
            os.replace(tmp, runs_path)
            build_leaderboards(pd.DataFrame(rows), out, args.quick)

    all_runs = pd.DataFrame(rows)
    if all_runs.empty:
        print("\nNo runs executed.")
        return
    build_leaderboards(all_runs, out, args.quick)
    failed = all_runs[all_runs["status"] != "ok"]
    print(f"\nDone in {(time.time() - t_start) / 60:.1f} min. Leaderboard: {out / 'LEADERBOARD.md'}")
    if len(failed):
        print(f"WARNING: {len(failed)} run(s) failed:")
        print(failed[["dataset", "model", "seed", "status"]].to_string(index=False))
        sys.exit(1)


if __name__ == "__main__":
    main()
