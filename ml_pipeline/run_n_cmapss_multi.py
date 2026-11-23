#!/usr/bin/env python3
"""
N-CMAPSS across multiple DS files (RUL metrics), continuous sliding window.

Test units run to failure, so only continuous scoring (every test window) is
meaningful; the old last-window protocol always had RUL == 0 targets.
Set NCMAPSS_DIR to the folder holding the .h5 files (default: N-CMAPSS/data_set).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from datasets.ablation_data import load_raw_n_cmapss  # noqa: E402
from run_family_comparison import SEEDS, evaluate_rul_dataset, pack_to_arrays, summarize_seeds, RUL_COLS  # noqa: E402

# DS08d is known to be truncated in some downloads -> not included by default
DATASETS = [
    "N-CMAPSS_DS01-005.h5",
    "N-CMAPSS_DS02-006.h5",
    "N-CMAPSS_DS03-012.h5",
    "N-CMAPSS_DS05.h5",
    "N-CMAPSS_DS07.h5",
]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    ap.add_argument("--files", nargs="+", default=DATASETS)
    args = ap.parse_args()

    out = ROOT / "results" / "family_comparison"
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for fname in args.files:
        print(f"\n===== {fname} =====", flush=True)
        try:
            pack = load_raw_n_cmapss(filename=fname)
        except (FileNotFoundError, OSError) as exc:
            print(f"  skip: {exc}")
            continue
        bundle = pack_to_arrays(pack)
        for seed in args.seeds:
            for r in evaluate_rul_dataset(bundle, seed=seed):
                r["source_file"] = fname
                rows.append(r)
        pd.DataFrame(rows).to_csv(out / "n_cmapss_multi_metrics.csv", index=False)

    if not rows:
        print("No N-CMAPSS files found; nothing to report. See DATASETS.md / NCMAPSS_DIR.")
        return

    df = pd.DataFrame(rows)
    df.to_csv(out / "n_cmapss_multi_metrics.csv", index=False)
    summ = summarize_seeds(df, "rmse", RUL_COLS)
    summ.to_csv(out / "n_cmapss_multi_summary.csv", index=False)
    summary = {
        "primary_metric": "continuous rmse (mean over seeds)",
        "seeds": args.seeds,
        "files": args.files,
        "winners": summ[summ["rank"] == 1].to_dict(orient="records"),
        "pooled_mean_rmse_by_model": df.groupby("model")["rmse"].mean().sort_values().to_dict(),
    }
    (out / "n_cmapss_multi_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(summ[["dataset", "model", "rmse_mean", "rmse_std", "r2_mean", "rank"]].to_string(index=False))
    print(f"Saved -> {out}")


if __name__ == "__main__":
    main()
