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

## Next
1. Re-run everything on Colab (T4) with 3 seeds → new reference numbers.
2. Stronger baselines for the continuous setting (e.g. 1D-CNN, CNN-LSTM, TCN, Transformer, LightGBM).
3. New method + new architecture for the continuous domain, benchmarked against those baselines.

## Commands
```bash
python ml_pipeline/run_family_comparison.py --seeds 42 43 44
```
