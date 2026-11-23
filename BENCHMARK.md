# Continuous RUL Benchmark (CRB)

A benchmark for **remaining-useful-life (RUL) prediction in the continuous domain**:
a model is scored on **every** sliding window along each test trajectory, not only on
the final window of each engine.

## Why a continuous benchmark?

The classic CMAPSS protocol scores one prediction per test engine (its last window, ~100–250
numbers per dataset). In deployment a model runs **continuously**, producing an estimate at every
cycle. A model can look good on last windows and still be erratic or dangerously late along the
trajectory. CRB measures the whole trajectory:

- **accuracy** at every cycle (continuous RMSE / MAE / R²),
- **safety** near failure (RMSE where true RUL ≤ 30; late-prediction rate; PHM score),
- **temporal coherence** (trajectory consistency error, TCE),

while still reporting the classic last-window RMSE / PHM score for comparison with published work.

## Protocol (fixed — do not change when comparing models)

| Item | Specification |
|---|---|
| Datasets | CMAPSS FD001, FD002, FD003, FD004; PHM08; N-CMAPSS DS01, DS02, DS03, DS05, DS07 (optional, data not in git) |
| Splits | unit-level, seed 42 (`datasets/ablation_data.py`). CMAPSS: official test set; 15% of train units → validation. PHM08: official test RUL is hidden → 20% of train units held out as test, 15% → validation. N-CMAPSS: official `dev`/`test`; 20% of dev units → validation |
| Windows | continuous sliding windows, **step 1**; length 30 (CMAPSS, PHM08) / 20 (N-CMAPSS); units shorter than the window are left-padded with their first row |
| Features | FD001/FD003: 3 settings + 14 informative sensors (17). FD002/FD004/PHM08: all 24. N-CMAPSS: cycle-averaged W (4) + X_s (14) |
| Scaling | `StandardScaler` fit on training windows only |
| Target | piece-wise linear RUL capped at 125 (N-CMAPSS: 100) |
| Test (continuous) | every window of every test unit (CMAPSS labels at each cycle = official last-cycle RUL + cycles remaining) |
| Test (last-window) | last window per test unit. PHM08: each held-out unit is cut at 5 random cycles before failure first. Not defined for N-CMAPSS (last window is always RUL 0) |
| Predictions | clipped at 0. Causal post-processing is allowed only if it uses past windows of the same unit and is tuned on validation |
| Seeds | 3 model seeds (42, 43, 44); report mean ± std. Data splits never change |
| Selection | any hyper-parameter / early-stopping / post-processing choice uses validation only |

## Metrics

| Metric | Definition | |
|---|---|---|
| **RMSE** (primary) | RMSE over all continuous test windows | ↓ |
| MAE, R² | over all continuous test windows | ↓ / ↑ |
| RMSE≤30 | RMSE on windows whose true RUL ≤ 30 (critical zone) | ↓ |
| PHM / window | NASA asymmetric score averaged per window | ↓ |
| late_rate | fraction of windows with prediction > truth (late = unsafe) | ↓ |
| **TCE** | trajectory consistency error: mean \|Δŷ − Δy\| between consecutive windows of a unit | ↓ |
| Last RMSE / Last PHM | classic last-window RMSE and summed PHM score | ↓ |

Ranking per dataset uses mean continuous RMSE; the summary table also reports average rank across datasets.

## Models

**Reference baselines** (all trained by the same code, same budget: AdamW, lr 1e-3, batch 256,
≤40 epochs, early stopping on validation, standardized target):

| Model | Description |
|---|---|
| MeanRUL | predicts the training-set mean (sanity floor) |
| Ridge | linear model on flattened windows |
| HistGB | histogram gradient boosting on flattened windows |
| LSTM | 2×64 LSTM |
| BiLSTM-Attn | bidirectional 2×64 LSTM + attention pooling |
| CNN1D | 4-layer 1D-CNN over time (in the spirit of Li et al., 2018) |
| CNN-LSTM | 2 conv layers → LSTM |
| TCN | 4 dilated causal residual blocks (dilations 1, 2, 4, 8) |
| Transformer | 2-layer pre-norm Transformer encoder, learned positions |
| RandomForest | optional (`--models RandomForest`), slow on Colab CPUs |

**Proposed**

- **CATT — Condition-Aware TCN-Transformer** (architecture, `models/deep.py::CATTNet`)
  1. *Condition-aware embedding*: sensor channels are projected and modulated by FiLM
     (per-timestep scale/shift) computed from the operating-condition channels
     (CMAPSS op1–op3, N-CMAPSS W). A learned replacement for per-regime normalization; FiLM
     is initialised to the identity.
  2. *Multi-scale causal TCN*: dilated causal residual blocks (1, 2, 4, 8) with skip aggregation.
  3. *Transformer encoder* over the multi-scale features for global context.
  4. Head on [last step ; attention-pooled summary].
- **TCT — Trajectory-Consistent Training** (method, `models/deep.py::DeepRUL`)
  1. *Consistency loss*: for two windows of the same unit `d ∈ [1, 10]` cycles apart, penalise
     `((ŷ_{t+d} − ŷ_t) − (y_{t+d} − y_t))²` — the prediction must evolve like true RUL
     (drop d cycles, or stay flat while capped). Weight λ = 1.
  2. *Asymmetric loss*: late errors (ŷ > y) weighted ×1.5, in line with the PHM score.
  3. *Causal trajectory smoothing* at inference: `s_t = α·ŷ_t + (1 − α)·(s_{t−1} − 1)`
     (held at the cap), α chosen on validation from {none, 0.7, 0.5, 0.35, 0.25, 0.15, 0.1}.
- Reported variants: `CATT` (architecture only), `CATT+TCT` (full proposal), `TCN+TCT`
  (method on a strong baseline, to show TCT is architecture-agnostic).

## Running

```bash
python ml_pipeline/run_benchmark.py --quick        # smoke test, ~2 min on CPU
python ml_pipeline/run_benchmark.py                # full benchmark (all models, 3 seeds)
python ml_pipeline/run_benchmark.py --datasets FD001 FD004 --models CATT+TCT TCN
python ml_pipeline/run_benchmark.py --out results/crb   # custom output dir (resumable)
```

On Google Colab use `colab_run.ipynb` (T4 GPU); it runs directly from this GitHub repo. Runs are
appended to `runs.csv` one at a time, and re-running the same command **resumes** (finished runs
are skipped). N-CMAPSS is not on GitHub (~28 GB), so Colab runs cover CMAPSS FD001–FD004 and PHM08.

Outputs (`ml_pipeline/results/benchmark/` or `--out`):

- `runs.csv` — one row per (dataset, model, seed) with every metric
- `leaderboard.csv`, `LEADERBOARD.md` — mean ± std per dataset, ranked by continuous RMSE
- `leaderboard_avg_rank.csv` — average rank across datasets
- `config.json` — exact settings, device and torch version

## Adding a model

Add an entry to `make_model()` in `ml_pipeline/run_benchmark.py`. A model needs
`fit(X, y, X_val, y_val)` and `predict(X)` on `(N, T, F)` windows. Optionally:
`uses_units = True` to receive `units=` / `val_units=` in `fit`, and `postprocess(pred, units)`
for causal trajectory post-processing. New PyTorch architectures can be registered in
`models/deep.py::ARCHS` and trained with `DeepRUL` to get the identical training budget.
