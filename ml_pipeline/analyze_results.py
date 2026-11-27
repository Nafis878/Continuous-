#!/usr/bin/env python3
"""
Statistical analysis of Continuous RUL Benchmark results.

Inputs : benchmark runs.csv (+ optional TCT-ablation runs.csv)
Outputs: <out>/REPORT.md, stats.json, cd_diagram.png, tct_ablation.csv

1. Friedman test over datasets on mean continuous RMSE + average ranks and the
   Nemenyi critical difference (CD) at alpha = 0.05 (Demsar, 2006), CD diagram.
2. Proposed vs. every baseline: paired Wilcoxon signed-rank tests over
   (dataset, seed) pairs on continuous RMSE, Holm-corrected; win counts.
3. Effect of TCT on the same architecture (CATT+TCT vs CATT, TCN+TCT vs TCN):
   RMSE, RMSE<=30, TCE, last-window PHM - paired Wilcoxon over (dataset, seed).
4. Protocol sensitivity: Kendall tau between continuous and last-window
   rankings per dataset, and whether the winner changes.
5. TCT component ablation: each variant vs CATT+TCT (paired), if provided.

Usage:
  python ml_pipeline/analyze_results.py --runs results/crb_2026-11-27_T4_full/benchmark/runs.csv \
      [--ablation-runs OUT/tct_ablation/runs.csv] --out OUT/analysis
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

PROPOSED = ["CATT+TCT", "CATT", "TCN+TCT"]
TCT_PAIRS = [("CATT+TCT", "CATT"), ("TCN+TCT", "TCN")]
METRICS = {"rmse": "Continuous RMSE", "rmse_critical": "RMSE (RUL<=30)", "tce": "TCE", "last_phm": "Last-window PHM"}


def load(path) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df[df["status"] == "ok"].copy()


def holm(pvals: list[float]) -> list[float]:
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj.tolist()


def paired(df: pd.DataFrame, a: str, b: str, metric: str):
    x = df[df.model == a].set_index(["dataset", "seed"])[metric]
    y = df[df.model == b].set_index(["dataset", "seed"])[metric]
    common = x.index.intersection(y.index)
    x, y = x.loc[common].astype(float), y.loc[common].astype(float)
    ok = x.notna() & y.notna()
    x, y = x[ok], y[ok]
    if len(x) < 5:
        return None
    d = x - y
    try:
        p = float(stats.wilcoxon(x, y, zero_method="zsplit").pvalue) if (d != 0).any() else 1.0
    except ValueError:
        p = float("nan")
    return {"n_pairs": int(len(d)), "mean_diff": float(d.mean()), "median_diff": float(d.median()),
            "a_better": int((d < 0).sum()), "b_better": int((d > 0).sum()), "p": p}


def friedman(df: pd.DataFrame):
    means = df.groupby(["dataset", "model"])["rmse"].mean().unstack()
    means = means.dropna(axis=1)  # models present on every dataset
    k, n = means.shape[1], means.shape[0]
    ranks = means.rank(axis=1, method="average")
    avg_rank = ranks.mean().sort_values()
    chi2, p = stats.friedmanchisquare(*[means[c].values for c in means.columns])
    q = stats.studentized_range.ppf(0.95, k, np.inf) / np.sqrt(2)
    cd = float(q * np.sqrt(k * (k + 1) / (6.0 * n)))
    return {"n_datasets": int(n), "k_models": int(k), "chi2": float(chi2), "p": float(p), "cd_0.05": cd,
            "avg_rank": {m: float(r) for m, r in avg_rank.items()}}


def cd_diagram(fr: dict, path: Path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ranks = fr["avg_rank"]
    names, vals = list(ranks), list(ranks.values())
    k = fr["k_models"]
    fig, ax = plt.subplots(figsize=(10, 0.42 * len(names) + 1.6))
    ax.set_xlim(1, k)
    ax.set_ylim(-0.5, len(names) + 1)
    ax.invert_yaxis()
    for i, (n, v) in enumerate(zip(names, vals)):
        color = "#c0392b" if n in PROPOSED else "#2c3e50"
        ax.plot([v], [i + 1], "o", color=color)
        ax.text(v + 0.08, i + 1, f"{n} ({v:.2f})", va="center", fontsize=9, color=color)
    # bars joining models whose rank difference is below the CD (not significantly different)
    cd = fr["cd_0.05"]
    best = vals[0]
    ax.plot([best, min(best + cd, k)], [0, 0], color="#7f8c8d", lw=4)
    ax.text(best, -0.3, f"CD = {cd:.2f} (Nemenyi, alpha=0.05)", fontsize=9)
    ax.set_xlabel("average rank across datasets (continuous RMSE, lower is better)")
    ax.set_yticks([])
    ax.set_title(f"Friedman chi2={fr['chi2']:.1f}, p={fr['p']:.2g}  ({fr['n_datasets']} datasets)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", required=True, help="benchmark runs.csv")
    ap.add_argument("--ablation-runs", help="TCT ablation runs.csv (optional)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    df = load(a.runs)
    rep = ["# CRB statistical analysis", "",
           f"Runs: {len(df)} ({df.dataset.nunique()} datasets, {df.model.nunique()} models, "
           f"seeds per model: {df.groupby('model').seed.nunique().min()}-{df.groupby('model').seed.nunique().max()})", ""]
    res = {}

    # 1. Friedman + CD
    fr = friedman(df)
    res["friedman"] = fr
    cd_diagram(fr, out / "cd_diagram.png")
    rep += ["## 1. Overall ranking (Friedman + Nemenyi)", "",
            f"Friedman chi2 = {fr['chi2']:.2f}, p = {fr['p']:.3g} over {fr['n_datasets']} datasets, "
            f"{fr['k_models']} models. Nemenyi CD (alpha=0.05) = **{fr['cd_0.05']:.2f}** rank units: models whose "
            "average ranks differ by less than the CD are not significantly different. See `cd_diagram.png`.", "",
            "| Model | Avg rank |", "|---|---|"]
    rep += [f"| {m} | {r:.2f} |" for m, r in fr["avg_rank"].items()]

    # 2. Proposed vs baselines
    baselines = [m for m in df.model.unique() if m not in PROPOSED and m != "MeanRUL"]
    rep += ["", "## 2. Proposed vs. baselines (continuous RMSE, paired over dataset x seed)", "",
            "Negative mean diff = proposed better. p: Wilcoxon signed-rank, Holm-corrected per proposed model.", "",
            "| Proposed | Baseline | pairs | mean diff | proposed better / worse | p (Holm) |", "|---|---|---|---|---|---|"]
    res["proposed_vs_baselines"] = {}
    for prop in [p for p in PROPOSED if p in df.model.unique()]:
        rows = [(b, paired(df, prop, b, "rmse")) for b in sorted(baselines)]
        rows = [(b, r) for b, r in rows if r]
        adj = holm([r["p"] for _, r in rows]) if rows else []
        for (b, r), pa in zip(rows, adj):
            r["p_holm"] = pa
            sig = " *" if pa < 0.05 else ""
            rep.append(f"| {prop} | {b} | {r['n_pairs']} | {r['mean_diff']:+.2f} | {r['a_better']} / {r['b_better']} | {pa:.3g}{sig} |")
        res["proposed_vs_baselines"][prop] = dict(rows)

    # 3. TCT effect
    rep += ["", "## 3. Effect of TCT on the same architecture (paired over dataset x seed)", "",
            "| Comparison | Metric | pairs | mean diff | with TCT better / worse | p |", "|---|---|---|---|---|---|"]
    res["tct_effect"] = {}
    for a_m, b_m in TCT_PAIRS:
        for metric, label in METRICS.items():
            if metric not in df.columns:
                continue
            r = paired(df, a_m, b_m, metric)
            if not r:
                continue
            res["tct_effect"][f"{a_m} vs {b_m} | {metric}"] = r
            sig = " *" if r["p"] < 0.05 else ""
            rep.append(f"| {a_m} vs {b_m} | {label} | {r['n_pairs']} | {r['mean_diff']:+.2f} | "
                       f"{r['a_better']} / {r['b_better']} | {r['p']:.3g}{sig} |")

    # 4. Protocol sensitivity
    rep += ["", "## 4. Continuous vs. last-window ranking", "",
            "| Dataset | Kendall tau | winner (continuous) | winner (last-window) | changed |", "|---|---|---|---|---|"]
    res["protocol"] = {}
    m = df[df.model != "MeanRUL"].groupby(["dataset", "model"])[["rmse", "last_rmse"]].mean().reset_index()
    for ds, g in m.dropna(subset=["last_rmse"]).groupby("dataset"):
        tau = float(stats.kendalltau(g.rmse, g.last_rmse)[0])
        wc, wl = g.sort_values("rmse").model.iloc[0], g.sort_values("last_rmse").model.iloc[0]
        res["protocol"][ds] = {"kendall_tau": tau, "winner_continuous": wc, "winner_last": wl}
        rep.append(f"| {ds} | {tau:.2f} | {wc} | {wl} | {'yes' if wc != wl else 'no'} |")

    # 5. TCT component ablation
    if a.ablation_runs and Path(a.ablation_runs).exists():
        # reference rows always come from the main benchmark
        ab = pd.concat([df, load(a.ablation_runs)], ignore_index=True)
        ab = ab.drop_duplicates(["dataset", "model", "seed"], keep="first")
        variants = sorted(v for v in ab.model.unique() if v.startswith("CATT+TCT-")) + ["CATT"]
        rep += ["", "## 5. TCT component ablation (variant minus full CATT+TCT, paired over dataset x seed)", "",
                "Positive diff = removing/changing the component made the metric worse (higher).", "",
                "| Variant | pairs | dRMSE | dRMSE<=30 | dTCE | dLast-PHM |", "|---|---|---|---|---|---|"]
        rows = []
        for v in variants:
            cells, rec = [], {"variant": v}
            for metric in ("rmse", "rmse_critical", "tce", "last_phm"):
                r = paired(ab, v, "CATT+TCT", metric)
                rec[metric] = r
                cells.append("–" if not r else f"{r['mean_diff']:+.2f}{' *' if r['p'] < 0.05 else ''}")
            npairs = next((r["n_pairs"] for r in (rec[k] for k in ("rmse",)) if r), 0)
            rep.append(f"| {v} | {npairs} | " + " | ".join(cells) + " |")
            rows.append({"variant": v, **{f"{k}_mean_diff": (rec[k] or {}).get("mean_diff") for k in METRICS},
                         **{f"{k}_p": (rec[k] or {}).get("p") for k in METRICS}})
        pd.DataFrame(rows).to_csv(out / "tct_ablation.csv", index=False)
        res["tct_ablation"] = rows
        rep += ["", "`*` = p < 0.05 (Wilcoxon signed-rank)."]

        # 6. Smoothing-only baselines
        smooth = [m for m in ab.model.unique() if m.endswith("+Smooth")]
        if smooth:
            rep += ["", "## 6. Smoothing-only baselines (paired over dataset x seed)", "",
                    "`X+Smooth` = plain model X + the same validation-tuned causal smoothing as TCT, no TCT loss. "
                    "Rows 'X+Smooth vs X' show what smoothing alone does; 'X+TCT vs X+Smooth' show what TCT adds "
                    "beyond smoothing. Negative diff = first model lower (better).", "",
                    "| Comparison | pairs | dRMSE | dRMSE<=30 | dTCE | dLast-PHM |", "|---|---|---|---|---|---|"]
            res["smoothing_only"] = {}
            comps = [(m, m.replace("+Smooth", "")) for m in sorted(smooth)]
            comps += [(f"{base}+TCT", f"{base}+Smooth") for base in ("CATT", "TCN")
                      if f"{base}+Smooth" in smooth and f"{base}+TCT" in ab.model.unique()]
            for x, y in comps:
                cells, rec = [], {}
                for metric in ("rmse", "rmse_critical", "tce", "last_phm"):
                    r = paired(ab, x, y, metric)
                    rec[metric] = r
                    cells.append("–" if not r else f"{r['mean_diff']:+.2f}{' *' if r['p'] < 0.05 else ''}")
                n = (rec["rmse"] or {}).get("n_pairs", 0)
                rep.append(f"| {x} vs {y} | {n} | " + " | ".join(cells) + " |")
                res["smoothing_only"][f"{x} vs {y}"] = rec

        # 7. Consistency weight lambda selected on VALIDATION (no test information used)
        cand = {"CATT+TCT-lambda0.3": 0.3, "CATT+TCT": 1.0, "CATT+TCT-lambda3": 3.0}
        pool = ab[ab.model.isin(cand)].dropna(subset=["val_rmse"])
        if pool.model.nunique() == len(cand):
            best = pool.loc[pool.groupby(["dataset", "seed"]).val_rmse.idxmin()].copy()
            best["lambda"] = best.model.map(cand)
            best["model"] = "CATT+TCT(val-lambda)"
            sel = pd.concat([ab, best.drop(columns="lambda")], ignore_index=True)
            share = best["lambda"].value_counts(normalize=True).sort_index()
            rep += ["", "## 7. Consistency weight selected on validation", "",
                    "For every (dataset, seed), lambda in {0.3, 1, 3} is chosen by the lowest **validation** RMSE; "
                    "the test metrics of that choice are reported as `CATT+TCT(val-lambda)`. "
                    "Chosen lambda: " + ", ".join(f"{k:g} in {v:.0%}" for k, v in share.items()) + ".", "",
                    "| Comparison | pairs | dRMSE | dRMSE<=30 | dTCE | dLast-PHM |", "|---|---|---|---|---|---|"]
            res["val_lambda"] = {"lambda_share": {str(k): float(v) for k, v in share.items()}}
            for y in ["CATT+TCT", "CATT", "TCN", "Transformer", "LSTM", "BiLSTM-Attn"]:
                if y not in sel.model.unique():
                    continue
                cells, rec = [], {}
                for metric in ("rmse", "rmse_critical", "tce", "last_phm"):
                    r = paired(sel, "CATT+TCT(val-lambda)", y, metric)
                    rec[metric] = r
                    cells.append("–" if not r else f"{r['mean_diff']:+.2f}{' *' if r['p'] < 0.05 else ''}")
                n = (rec["rmse"] or {}).get("n_pairs", 0)
                rep.append(f"| CATT+TCT(val-lambda) vs {y} | {n} | " + " | ".join(cells) + " |")
                res["val_lambda"][f"vs {y}"] = rec
            rep += ["", "`*` = p < 0.05, Wilcoxon signed-rank, **not** multiplicity-corrected (6 comparisons)."]
            best[["dataset", "seed", "lambda", "val_rmse", "rmse", "tce", "rmse_critical", "last_rmse", "last_phm"]] \
                .to_csv(out / "val_selected_lambda.csv", index=False)

    (out / "stats.json").write_text(json.dumps(res, indent=2, default=float))
    (out / "REPORT.md").write_text("\n".join(rep) + "\n", encoding="utf-8")
    print("\n".join(rep))
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
