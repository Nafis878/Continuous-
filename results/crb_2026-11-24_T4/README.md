# CRB run — 2026-11-24, Google Colab Tesla T4

First full run of the Continuous RUL Benchmark (`python run_pipeline.py --all`), copied from the
Google Drive output folder (`My Drive/CRB/results`) unchanged.

| | |
|---|---|
| GPU | Tesla T4 (torch 2.14.1+cu130) |
| Benchmark | **complete** — 390/390 runs OK: 13 models × 10 datasets × 3 seeds (42, 43, 44), 355 min |
| Datasets | CMAPSS FD001–FD004, PHM08, N-CMAPSS DS01/02/03/05/07 |
| Smoke test | complete — 130/130 runs OK (1 epoch; numbers not meaningful) |
| Studies | **incomplete** — condition clustering finished FD002 only; N-CMAPSS multi-file, family comparison, SCANIA cost tuning and LSTM ablation did not run |

The benchmark code has not changed since this run (only `runs.csv` is now written atomically), so
these results are consistent with the current `main`.

## Contents

```
benchmark/LEADERBOARD.md            per-dataset tables (mean ± std over seeds) + average rank
benchmark/runs.csv                  every run, every metric (390 rows)
benchmark/leaderboard.csv           aggregated metrics per (dataset, model)
benchmark/leaderboard_avg_rank.csv  average rank across datasets
benchmark/config.json               exact settings, GPU, torch version
smoke/                              smoke-test outputs (1 epoch)
logs/benchmark.log, smoke.log, ncmapss_download.log
studies/family_comparison/condition_rul_metrics.csv   partial: FD002 only
```

## Key findings (see the analysis in the project notes)

- Best average rank: TCN (5.2), Transformer (5.4), LSTM (5.7); CATT+TCT 5.8.
- CATT+TCT is #1 only on NCMAPSS-DS01 and beats the best baseline on FD003; ~+2 RMSE behind on FD001/FD002/FD004.
- TCT lowers trajectory consistency error (TCE) on 10/10 datasets for both CATT and TCN, RMSE-neutral.
- Continuous vs. last-window scoring picks a different winner on 3/5 CMAPSS/PHM08 datasets.
- N-CMAPSS test sets are small (145–324 windows from 3–6 units) → noisy; to be fixed (sub-cycle resolution).
- RandomForest used 5.05 h of the 5.9 h benchmark compute (CPU-bound on Colab's 2 cores).

## Run the remaining parts on another Drive / account

In Colab (mount Drive first), from a fresh clone of this repo:

```python
!python run_pipeline.py --install --skip-smoke --resume-from results/crb_2026-11-24_T4 \
  --with-ncmapss --with-studies --with-scania --with-ablation \
  --out "/content/drive/MyDrive/CRB/results" --ncmapss-dir /content/N-CMAPSS \
  --seeds 42 43 44 --models MeanRUL Ridge HistGB RandomForest LSTM BiLSTM-Attn CNN1D CNN-LSTM TCN Transformer CATT CATT+TCT TCN+TCT
```

`--resume-from` copies this folder into the new output, so all 390 benchmark runs are recognised as
finished and skipped; only the studies run. N-CMAPSS is re-downloaded from NASA (~8 min on Colab) —
it is too large for GitHub. The remaining studies are CPU-bound (RandomForest): prefer a runtime with
more CPU cores (e.g. A100) — roughly 4–6 h there vs. 12–16 h on a T4 (estimates).
