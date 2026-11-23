# Research (Main) — Predictive Maintenance Pipelines

End-to-end ML experiments on CMAPSS, PHM08, N-CMAPSS, and SCANIA Component X, with within-family metrics (RUL → RMSE/MAE/R²/PHM Score; SCANIA → Total_cost / F1).

## Layout

- `ml_pipeline/` — training, ablation, family comparison, condition-clustered RUL
- `CMAPSS/`, `PHM08/`, `N-CMAPSS/`, `SCANIA-Component-X/`, `alfa-dataset-tools/`
- See `DATASETS.md` for what is in git vs download-only (N-CMAPSS `.h5`)

Evaluation uses a **continuous sliding window** (every overlapping window, step 1),
scores every test window (plus the classic last-window benchmark where valid), runs
3 seeds, and always reports trivial baselines. Details: `ml_pipeline/README.md`.

## Run on Google Colab (T4 GPU)

Open [`colab_run.ipynb`](colab_run.ipynb) in Colab, set Runtime → Change runtime type → **T4 GPU**,
and run all cells. It clones this repo, pulls the LFS data, installs dependencies, runs every
experiment and (optionally) saves results to Google Drive.

## Local setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r ml_pipeline/requirements.txt
git lfs install && git lfs pull   # SCANIA large CSVs
```

## Main commands

```bash
python ml_pipeline/run_family_comparison.py
python ml_pipeline/run_ablation.py
python ml_pipeline/run_condition_rul.py
python ml_pipeline/run_scania_cost_tune.py
python ml_pipeline/run_n_cmapss_multi.py
```

Results live under `ml_pipeline/results/`.
