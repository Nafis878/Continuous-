# Datasets in this repository

## Included in git

| Dataset | Path | Notes |
|---------|------|-------|
| CMAPSS | `CMAPSS/` | Full NASA C-MAPSS text files + zip |
| PHM08 | `PHM08/` | Challenge train/test text + zip |
| SCANIA Component X | `SCANIA-Component-X/` | Specs, labels, TTE, operational CSVs (large CSVs via Git LFS) |
| Pipeline | `ml_pipeline/` | End-to-end LSTM/family comparison code + results |

## Not in git (too large for GitHub)

### N-CMAPSS HDF5 (`N-CMAPSS/data_set/*.h5`, ~28GB)

GitHub file limit is 100MB (LFS free tier ≈ 1GB storage). These files are 2–3.5GB each.

**Automatic download** (recommended) — streams NASA's archive once, extracts only the benchmark
files (DS01, DS02, DS03, DS05, DS07; ~15 GB), verifies CRCs, skips files already present:

```bash
python run_pipeline.py --with-ncmapss                       # as part of the pipeline
python ml_pipeline/datasets/ncmapss_download.py --dest N-CMAPSS/data_set   # download only
python ml_pipeline/datasets/ncmapss_download.py --dest N-CMAPSS/data_set --files all   # all files
```

Source archive: `https://phm-datasets.s3.amazonaws.com/NASA/17.+Turbofan+Engine+Degradation+Simulation+Data+Set+2.zip`

Or place the official `.h5` files here manually:

```text
N-CMAPSS/data_set/
  N-CMAPSS_DS01-005.h5
  N-CMAPSS_DS02-006.h5
  ...
```

Or keep them in another folder and point the pipeline at it:

```bash
export NCMAPSS_DIR=/path/to/N-CMAPSS
```

(`run_pipeline.py --ncmapss-dir DIR` does this for you.) On Colab, `colab_run.ipynb` downloads the
files automatically; set `KEEP_NCMAPSS_IN_DRIVE = True` to keep them in `My Drive/CRB/N-CMAPSS` of
the mounted account (needs ~15 GB free Drive space). Without the files, N-CMAPSS is skipped and every
other dataset still runs.

Notebook and PDF documentation **are** included under `N-CMAPSS/data_set/`.

Source: NASA / N-CMAPSS release referenced in  
`N-CMAPSS/data_set/Run_to_Failure_Simulation_Under_Real_Flight_Conditions_Dataset.pdf`

## Git LFS

Large SCANIA operational CSVs are tracked with Git LFS. After clone:

```bash
git lfs install
git lfs pull
```
