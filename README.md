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
3. Run all cells and approve the Google Drive pop-up with **any Google account**. The code comes
   straight from this GitHub repo; **everything** (leaderboard, every run, all studies, logs, status)
   is saved in *that* account's Drive under `My Drive/CRB/results` (folder name: `DRIVE_FOLDER`).
   Every finished run is saved immediately, so after a disconnect just Run all again and it resumes.

With `RUN_EVERYTHING = True` (default) the notebook runs `run_pipeline.py --all`: N-CMAPSS is
downloaded from NASA automatically (~15 GB, streamed, CRC-checked), then the smoke test, the full
benchmark on CMAPSS FD001–FD004 + PHM08 + N-CMAPSS DS01/02/03/05/07, and all studies.

## Results of earlier runs (`results/`)

| Run | Status |
|---|---|
| [`results/crb_2026-11-24_T4`](results/crb_2026-11-24_T4/README.md) | benchmark complete (390/390 runs, T4); studies incomplete |

To continue from a committed run on any Drive/account, pass `--resume-from results/<run>` (the Colab
notebook does this by default via `RESUME_FROM`): finished benchmark runs are skipped and only the
missing parts run.

## One-command pipeline scripts

`run_pipeline.py` (any OS) and `run_pipeline.sh` (Colab / Linux / macOS) run everything in order:
environment check → N-CMAPSS download (optional) → smoke test (stops early if anything fails) →
full benchmark → optional studies → summary. Everything goes to one folder (`--out`) with
`benchmark/`, `studies/`, `logs/` (one log per stage) and `pipeline_status.json`. Re-running resumes.

```bash
python run_pipeline.py --quick                            # ~2 min check of every model
python run_pipeline.py                                    # smoke test + full benchmark (3 seeds)
python run_pipeline.py --out /content/drive/MyDrive/CRB   # Colab: results in the mounted Drive
python run_pipeline.py --models CATT+TCT CATT TCN --datasets FD001 FD004
python run_pipeline.py --all                              # everything incl. N-CMAPSS download
python run_pipeline.py --dry-run --all                    # show the plan only
bash run_pipeline.sh --help                               # all options
```

In Colab, two cells are enough (results land in the Drive of whichever account you mount):

```python
from google.colab import drive
drive.mount("/content/drive")
```

```python
import os
os.environ["GIT_LFS_SKIP_SMUDGE"] = "1"   # LFS data is fetched later only if needed
if not os.path.isdir("/content/Continuous"):
    !git clone -q https://github.com/Nafis878/Continuous-.git /content/Continuous
else:
    !git -C /content/Continuous pull -q
%cd /content/Continuous
!python run_pipeline.py --install --all --out /content/drive/MyDrive/CRB/results --ncmapss-dir /content/N-CMAPSS
```

| Option | Effect |
|---|---|
| `--all` | everything: `--with-ncmapss --with-studies --with-scania --with-ablation` |
| `--with-ncmapss` | download N-CMAPSS DS01/02/03/05/07 from NASA (~15 GB; skipped if already present) |
| `--ncmapss-dir` | where N-CMAPSS lives / is downloaded (default `N-CMAPSS/data_set`) |
| `--quick` | environment + smoke test only |
| `--seeds`, `--datasets`, `--models`, `--epochs` | passed to the benchmark |
| `--with-studies` | condition clustering (FD002/FD004) + N-CMAPSS multi-file study |
| `--with-scania` | SCANIA classification studies (downloads ~1.7 GB LFS data automatically) |
| `--with-ablation` | LSTM ablation study (needs SCANIA data too) |
| `--install` | `pip install -r ml_pipeline/requirements.txt` first |
| `--skip-smoke`, `--keep-going`, `--dry-run` | skip smoke test / run studies even if benchmark failed / print plan |

### If Google Drive disconnects (`[Errno 107] Transport endpoint is not connected`)

Colab's Drive mount can drop during long runs. The pipeline is built for this: when `--out` is on
Drive it works in a local folder (`/content/crb_work`) and copies results to Drive every 2 minutes
and after each stage; a dropped Drive only causes a warning and is retried. To recover, re-mount and
re-run the same command (finished work resumes, downloaded N-CMAPSS files are reused):

```python
from google.colab import drive
try:
    drive.flush_and_unmount()
except Exception:
    pass
drive.mount("/content/drive", force_remount=True)
```

If the run already finished while Drive was down, copy the results over with
`python run_pipeline.py --sync-only --out /content/drive/MyDrive/CRB/results`.

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
| N-CMAPSS DS01/02/03/05/07 | no (too large for GitHub) | downloaded automatically from NASA by `run_pipeline.py --with-ncmapss` / `--all` (~15 GB); skipped if absent |
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
