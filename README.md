# Continuous RUL Benchmark (CRB)

A benchmark and reference implementation for **remaining-useful-life (RUL) prediction in the
continuous domain**: every model is scored on every sliding window along each test trajectory
(how it behaves when deployed cycle-by-cycle), with the classic last-window score reported alongside.

- **Protocol + metrics** — fixed splits, step-1 sliding windows, continuous RMSE (primary),
  critical-zone RMSE (RUL ≤ 30), trajectory consistency error (TCE), PHM score, last-window RMSE.
- **Strong baselines** — MeanRUL, Ridge, HistGB, LSTM, BiLSTM-Attn, 1D-CNN, CNN-LSTM, TCN, Transformer,
  all trained by the same code with the same budget, 3 seeds each.
- **Proposed** — **CATT** (Condition-Aware TCN-Transformer) architecture and **TCT**
  (Trajectory-Consistent Training) method.

Full specification: **[BENCHMARK.md](BENCHMARK.md)**.

## Run on Google Colab (T4 GPU)

1. Open [`colab_run.ipynb`](colab_run.ipynb) in Colab (File → Open notebook → GitHub, or upload it).
2. Runtime → Change runtime type → **T4 GPU**.
3. Run all cells. The code comes straight from this GitHub repo. When Colab asks to mount Google
   Drive, pick **any Google account** — results are saved to `My Drive/CRB/results` in *that*
   account (folder name configurable via `DRIVE_FOLDER`). Every finished run is saved immediately,
   so after a disconnect just Run all again and it resumes. Without Drive (`SAVE_TO_DRIVE = False`)
   results stay on the Colab machine and are downloaded as a zip.

## Run locally

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r ml_pipeline/requirements.txt
python ml_pipeline/run_benchmark.py --quick     # smoke test
python ml_pipeline/run_benchmark.py             # full benchmark
```

Results: `ml_pipeline/results/benchmark/LEADERBOARD.md` (+ `runs.csv`, `leaderboard.csv`).

## Datasets

| Dataset | In repo | Notes |
|---|---|---|
| CMAPSS FD001–FD004 | yes (`CMAPSS/`) | official train/test + RUL files |
| PHM08 | yes (`PHM08/`) | official test RUL hidden → held-out train units |
| N-CMAPSS DS01/02/03/05/07 | no (~28 GB, too large for GitHub) | Colab: put `.h5` files in `My Drive/N-CMAPSS/` of the mounted account. Local: `N-CMAPSS/data_set/`. Skipped if absent |
| SCANIA Component X | yes (Git LFS) | used by the separate classification study, not by CRB |

See [DATASETS.md](DATASETS.md).

## Other experiments (`ml_pipeline/`)

| Script | Purpose |
|---|---|
| `run_benchmark.py` | **CRB benchmark** (main entry point) |
| `run_family_comparison.py` | earlier within-family comparison incl. SCANIA classification (cost metric) |
| `run_condition_rul.py` | condition clustering vs. plain scaling on FD002/FD004 |
| `run_scania_cost_tune.py` | SCANIA cost-aware decoding vs. Always-0/Always-4 baselines |
| `run_n_cmapss_multi.py` | classic models across N-CMAPSS files |
| `run_ablation.py` | LSTM one-factor ablations |

SCANIA scripts need the LFS data: `git lfs pull`.
