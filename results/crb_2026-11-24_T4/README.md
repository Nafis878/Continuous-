# CRB run — 2026-11-24, Google Colab Tesla T4

First full run of the Continuous RUL Benchmark (`python run_pipeline.py --all`), copied from the
Google Drive output folder (`My Drive/CRB/results`) unchanged.

| | |
|---|---|
| GPU | Tesla T4 (torch 2.14.1+cu130) |
| Benchmark | **complete** — 390/390 runs OK: 13 models × 10 datasets × 3 seeds (42, 43, 44), 355 min |
| Datasets | CMAPSS FD001–FD004, PHM08, N-CMAPSS DS01/02/03/05/07 |
| Smoke test | complete — 130/130 runs OK (1 epoch; numbers not meaningful) |
| Studies | condition clustering (FD002 + FD004, 3 seeds) and N-CMAPSS multi-file: **complete** (second run, finished 2026-11-25, 6.2 h — almost all sklearn RandomForest). Family comparison, SCANIA cost tuning, LSTM ablation: **not run** (that Colab runtime had no git-lfs; fixed — the pipeline now downloads SCANIA directly from GitHub) |

The benchmark's training and evaluation code has not changed since this run (later changes only add
a `backend` column, atomic `runs.csv` writes and the TCT-ablation models), so these results are
consistent with the current `main`. The benchmark's RandomForest is always sklearn.

## Contents

```
benchmark/LEADERBOARD.md            per-dataset tables (mean ± std over seeds) + average rank
benchmark/runs.csv                  every run, every metric (390 rows)
benchmark/leaderboard.csv           aggregated metrics per (dataset, model)
benchmark/leaderboard_avg_rank.csv  average rank across datasets
benchmark/config.json               exact settings, GPU, torch version
smoke/                              smoke-test outputs (1 epoch)
logs/benchmark.log, smoke.log, ncmapss_download.log
studies/family_comparison/condition_rul_*        condition clustering vs. plain scaling (complete)
studies/family_comparison/n_cmapss_multi_*       N-CMAPSS multi-file study (complete)
logs/condition_rul.log, logs/n_cmapss_multi.log  logs of the 2026-11-25 study run
pipeline_status_2026-11-25_studies.json          status of the 2026-11-25 study run
```

## Key findings (see the analysis in the project notes)

- Best average rank: TCN (5.2), Transformer (5.4), LSTM (5.7); CATT+TCT 5.8.
- CATT+TCT is #1 only on NCMAPSS-DS01 and beats the best baseline on FD003; ~+2 RMSE behind on FD001/FD002/FD004.
- TCT lowers trajectory consistency error (TCE) on 10/10 datasets for both CATT and TCN, RMSE-neutral.
- Continuous vs. last-window scoring picks a different winner on 3/5 CMAPSS/PHM08 datasets.
- N-CMAPSS test sets are small (145–324 windows from 3–6 units) → noisy; to be fixed (sub-cycle resolution).
- RandomForest used 5.05 h of the 5.9 h benchmark compute (CPU-bound on Colab's 2 cores).
- Condition clustering (validation-selected LSTM): FD002 18.59 -> 17.79, FD004 17.25 -> **14.91** RMSE
  (better than every benchmark model on FD004).

## Complete everything on another Drive / account (Colab T4)

Open `colab_run.ipynb` and Run all (defaults are set for this), or see the README for the cells.
`--resume-from results/crb_2026-11-24_T4` copies this folder into the new output, so the 390 benchmark
runs and the two completed studies are skipped. What still runs: benchmark seeds 45 and 46, the TCT
component ablation (5 seeds), family comparison, SCANIA cost tuning, LSTM ablation (studies use the
GPU RandomForest), and the statistical analysis.
