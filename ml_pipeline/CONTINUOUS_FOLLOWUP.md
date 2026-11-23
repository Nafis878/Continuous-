# Continuous follow-up plan (within-family metrics)

## Rules
1. Compare models **within** a dataset/family only.
2. RUL primary = continuous RMSE (last-window RMSE also reported); SCANIA primary = Total_cost.
3. No cross-family accuracy leaderboard.
4. Keep splits leakage-safe (unit/vehicle hold-out, train-only scaling).
5. Continuous sliding window: step-1 windows for train/val; score every test window.
6. Never evaluate on a constant target (e.g. last window of a run-to-failure unit = RUL 0).
7. Always report trivial baselines (MeanRUL; SCANIA Always-0 / Always-4) and ≥3 seeds (mean ± std).
8. Select models/settings on validation only.

## Evaluation fixes (2026-11-24)
- PHM08 / N-CMAPSS / SCANIA-TTE test targets were all RUL 0 → fixed (random pre-failure cuts; continuous scoring).
- SCANIA-TTE: censored vehicles were labelled as failing → now repaired vehicles only.
- SCANIA "best" cost 49,039 ≈ Always-4 baseline 49,671 → baselines now reported; % vs. Always-4 column.
- Plain LSTM collapsed to the mean on FD002/FD004/PHM08 → RUL target standardized during training.
- Condition-clustering winner was picked on test → now picked on validation.
- Stride-2 training windows → step-1 continuous windows everywhere.
- Legacy `run_all.py` (cross-family "accuracy" leaderboard) removed.
- All earlier result files removed (see `results/README.md`); re-run on Colab to regenerate.

## Continuous RUL Benchmark (2026-11-24)
- Main goal: establish a new benchmark in the continuous domain → `BENCHMARK.md`, `run_benchmark.py`.
- Stronger baselines added: BiLSTM-Attn, CNN1D, CNN-LSTM, TCN, Transformer, HistGB, Ridge (+ MeanRUL floor).
- Proposed: CATT (Condition-Aware TCN-Transformer) + TCT (Trajectory-Consistent Training).
- Verified locally (CPU): quick mode, all 12 models × 5 datasets OK; synthetic N-CMAPSS file OK;
  resume OK; FD001 15-epoch check: MeanRUL 37.6 / LSTM 15.6 / CATT 14.6 / CATT+TCT 15.2 continuous RMSE.

## Next
1. Full benchmark on Colab T4 (`colab_run.ipynb`), 3 seeds → first official leaderboard.
2. N-CMAPSS (.h5, ~28 GB) is not on GitHub → on Colab, place files in `My Drive/N-CMAPSS/` of the mounted account.
3. Tune TCT weights (λ, late penalty) on validation if CATT+TCT does not beat CATT on RMSE; report
   the TCE / RMSE≤30 trade-off honestly either way.

## Commands
```bash
python ml_pipeline/run_benchmark.py --quick   # smoke test
python ml_pipeline/run_benchmark.py           # full benchmark
```
