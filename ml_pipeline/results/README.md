# Results

Earlier result files were removed because they were produced by a flawed evaluation:

- **PHM08 / N-CMAPSS / SCANIA-TTE**: the test target was the last window of run-to-failure
  units, which always has RUL = 0. Every model was scored against a constant, so R² was exactly 0
  and Linear got RMSE = 0 just by having its negative predictions clipped to 0.
- **SCANIA**: the "improved" cost (49,039) came from predicting class 4 for almost every vehicle.
  The constant "always predict 4" baseline costs 49,671, so it was only 1.3% better than a constant.
- **Legacy `summary.json`**: ranked all datasets by a single "accuracy". The SCANIA "accuracy 0.97"
  was the class-0 share (the model predicted 0 for everything).
- The plain LSTM collapsed to predicting the mean on FD002/FD004/PHM08 (unscaled RUL target).
- Single seed; condition-clustering winner was chosen on test.

All of these are fixed in the code (see `../README.md` for the protocol). Re-run the scripts,
e.g. with `../../colab_run.ipynb` on a Colab T4 GPU, to regenerate:

- `family_comparison/rul_family_metrics.csv`, `rul_family_summary.csv`
- `family_comparison/scania_family_metrics.csv`, `scania_family_summary.csv`
- `family_comparison/condition_rul_*`, `scania_cost_tune.*`, `n_cmapss_multi_*`, `summary.json`
- `ablation/ablation_*`
