# CRB — complete run on Google Colab Tesla T4 (finished 2026-11-27)

Copied unchanged from the Google Drive output folder (`My Drive/CRB/results`). The results were
built up over three Colab sessions that resumed from each other:

| Session | What ran |
|---|---|
| 2026-11-24 | benchmark seeds 42–44 (13 models × 10 datasets, 390 runs, 355 min) |
| 2026-11-25 | condition clustering + N-CMAPSS multi-file studies (6.2 h, sklearn RandomForest) |
| 2026-11-26/27 | benchmark seeds 45–46 (12 models, 240 runs), TCT ablation (250 runs), family comparison, SCANIA cost tuning, LSTM ablation, statistical analysis |
| 2026-11-27 (follow-up) | smoothing-only baselines CATT/TCN/LSTM/Transformer+Smooth (200 runs, 31 min), analysis regenerated |

| | |
|---|---|
| Benchmark | **630/630 runs OK** — 12 models × 10 datasets × 5 seeds; RandomForest (sklearn) 3 seeds |
| TCT ablation | **450/450 runs OK** — 5 CATT+TCT variants + 4 smoothing-only baselines × 10 datasets × 5 seeds |
| Studies | condition clustering, N-CMAPSS multi-file, family comparison (+SCANIA), SCANIA cost tuning, LSTM ablation — all complete |
| Analysis | `analysis/REPORT.md` (sections 1–7 incl. 6: smoothing-only baselines, 7: lambda selected on validation), `stats.json`, `cd_diagram.png`, `tct_ablation.csv`, `val_selected_lambda.csv` |
| GPU | Tesla T4, torch 2.14.1+cu130 |

Note on RandomForest in `studies/family_comparison`: the GPU (XGBoost) forest ran out of GPU memory
on CMAPSS/PHM08 and fell back to sklearn (column `backend`), so those rows are sklearn — comparable
with the benchmark. Only N-CMAPSS DS01 used `xgboost-cuda`. (The GPU forest now grows leaf-wise to
avoid this; see `ml_pipeline/models/baselines.py`.)

## Contents

```
benchmark/        LEADERBOARD.md, runs.csv (630 rows), leaderboard*.csv, config.json
tct_ablation/     runs.csv (250 rows) + leaderboard for the TCT component ablation
studies/family_comparison/   condition_rul_*, n_cmapss_multi_*, rul_family_*, scania_family_*,
                             scania_cost_tune.*, summary.json
studies/ablation/            LSTM one-factor ablation (98 configs, 7 datasets)
analysis/         statistical report (Friedman/Nemenyi, Wilcoxon, TCT effect, protocol, ablation)
smoke/            smoke test of the last session (1 epoch; not meaningful)
logs/             one log per stage; *_2026-11-24_first_run.log = first session
pipeline_status*.json        status of the last session / of the 2026-11-25 study session
```

## Key findings (5 seeds)

- Friedman p = 0.0012 but Nemenyi CD = 5.77: BiLSTM-Attn, LSTM, Transformer (5.5), TCN (5.6), CATT (5.8)
  are tied; CATT+TCT 6.5. Proposed models are significantly better only than CNN1D.
- TCT lowers TCE on 49/50 (CATT) and 50/50 (TCN) dataset×seed pairs (p ≈ 1e-14), and last-window
  PHM (p = 0.04 / 0.003); RMSE-neutral.
- Ablation: removing the consistency loss barely changes TCE but worsens last-window PHM (+442*);
  removing causal smoothing raises TCE (+0.56*) and improves RMSE≤30 (−0.42*). → TCE gains come mostly
  from smoothing; the consistency loss mainly helps late-prediction safety.
- λ selected on **validation** (re-analysis, no extra training): λ = 3 chosen in 56% of cases;
  CATT+TCT(val-λ) is not worse in RMSE than any strong baseline and has significantly lower TCE
  than all of them.
- Continuous vs. last-window scoring changes the winner on 2/5 CMAPSS/PHM08 datasets.
- **Smoothing-only baselines:** causal smoothing alone lowers TCE as much as TCT (CATT −1.30 vs −1.20,
  TCN −2.61 vs −2.42); TCT vs smoothing-only shows no TCE difference. Smoothing alone costs accuracy
  (RMSE +0.25…+0.51; RMSE≤30 +0.62* for CATT/Transformer); TCT recovers near-failure accuracy vs
  smoothing-only for CATT (RMSE≤30 −0.61*, uncorrected). → TCE gains are due to smoothing; TCT's added
  value is preserving critical-zone accuracy. Smoothing is a strong, cheap baseline for continuous RUL.

## Continue from here

`python run_pipeline.py --resume-from results/crb_2026-11-27_T4_full ...` (the Colab notebook does
this by default): everything above is skipped; only new models (e.g. the smoothing-only baselines)
are trained, then the analysis is regenerated.
