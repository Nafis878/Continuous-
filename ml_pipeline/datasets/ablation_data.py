"""
Leakage-safe raw dataset preparation for LSTM ablations.

Design rules:
- Unit/vehicle-level splits only (no sample-level leakage across engines).
- Scaler is NEVER fit here; callers fit StandardScaler on train windows only.
- Same RNG seed → identical splits across all ablations of a dataset.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

from .cmapss import COLS as CMAPSS_COLS
from .n_cmapss import _cycle_aggregate
from .sequence_utils import add_rul_labels, truncate_trajectories

WORKSPACE = Path(__file__).resolve().parents[2]
SEED = 42

# Feature presets for CMAPSS / PHM08 style tables
FEAT_SELECTED = [
    "op1", "op2", "op3",
    "s2", "s3", "s4", "s7", "s8", "s9", "s11", "s12", "s13", "s14", "s15", "s17", "s20", "s21",
]
FEAT_ALL = ["op1", "op2", "op3"] + [f"s{i}" for i in range(1, 22)]
FEAT_SENSORS = [f"s{i}" for i in range(1, 22)]
FEAT_OPS = ["op1", "op2", "op3"]
FEAT_REDUCED = ["op1", "op2", "s4", "s7", "s11", "s12", "s15", "s17", "s20", "s21"]


def _unit_splits(units: np.ndarray, val_frac=0.15, test_frac: float | None = None, seed=SEED):
    """Deterministic unit split. If test_frac is None, only train/val (external test provided)."""
    units = np.asarray(sorted(units))
    rng = np.random.default_rng(seed)
    units = units.copy()
    rng.shuffle(units)
    n = len(units)
    if test_frac is None:
        n_val = max(1, int(val_frac * n))
        return set(units[n_val:]), set(units[:n_val]), None
    n_test = max(1, int(test_frac * n))
    n_val = max(1, int(val_frac * n))
    test_u = set(units[:n_test])
    val_u = set(units[n_test : n_test + n_val])
    train_u = set(units[n_test + n_val :])
    return train_u, val_u, test_u


def _read_space(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep=r"\s+", header=None, names=CMAPSS_COLS)


def load_raw_cmapss(fd: str = "FD001", rul_cap: float = 125.0) -> dict:
    root = WORKSPACE / "CMAPSS"
    train = add_rul_labels(_read_space(root / f"train_{fd}.txt"), cap=rul_cap)
    test = _read_space(root / f"test_{fd}.txt")
    rul_true = pd.read_csv(root / f"RUL_{fd}.txt", sep=r"\s+", header=None, names=["RUL"])
    last_rul = rul_true["RUL"].to_numpy()
    unit_last = {u: float(last_rul[i]) for i, u in enumerate(sorted(test["unit"].unique()))}
    test = test.copy()
    test_max = test.groupby("unit")["cycle"].transform("max")
    test["RUL"] = test["unit"].map(unit_last) + (test_max - test["cycle"])
    test["RUL"] = np.minimum(test["RUL"], rul_cap)

    train_u, val_u, _ = _unit_splits(train["unit"].unique(), val_frac=0.15, test_frac=None)
    return {
        "name": f"CMAPSS-{fd}",
        "train_df": train.loc[train["unit"].isin(train_u)].reset_index(drop=True),
        "val_df": train.loc[train["unit"].isin(val_u)].reset_index(drop=True),
        "test_df": test.reset_index(drop=True),
        "feature_presets": {
            "selected": FEAT_SELECTED,
            "all": FEAT_ALL,
            "sensors_only": FEAT_SENSORS,
            "ops_only": FEAT_OPS,
            "reduced": FEAT_REDUCED,
        },
        "default_features": "selected",
        "cond_features": FEAT_OPS,  # operating-condition channels (used by CATT FiLM)
        "default_seq_len": 30,
        # Continuous sliding window: every overlapping window (step 1).
        "stride_train": 1,
        # Official test labels are known at every cycle (RUL_last + cycles to end),
        # so test_df supports both last-window and continuous scoring.
        "last_window_eval": True,
    }


def load_raw_phm08(rul_cap: float = 125.0) -> dict:
    root = WORKSPACE / "PHM08"
    train = add_rul_labels(_read_space(root / "train.txt"), cap=rul_cap)
    train_u, val_u, test_u = _unit_splits(
        train["unit"].unique(), val_frac=0.15, test_frac=0.20
    )
    # Held-out units are run-to-failure: their last cycle always has RUL == 0.
    # Cut each one at random cycles before failure (CMAPSS-style test cases).
    test_df = truncate_trajectories(
        train.loc[train["unit"].isin(test_u)], n_cuts=5, min_len=30, seed=SEED
    )
    return {
        "name": "PHM08",
        "train_df": train.loc[train["unit"].isin(train_u)].reset_index(drop=True),
        "val_df": train.loc[train["unit"].isin(val_u)].reset_index(drop=True),
        "test_df": test_df,
        # Continuous scoring runs over the full held-out trajectories.
        "test_cont_df": train.loc[train["unit"].isin(test_u)].reset_index(drop=True),
        "feature_presets": {
            "selected": FEAT_SELECTED,
            "all": FEAT_ALL,
            "sensors_only": FEAT_SENSORS,
            "ops_only": FEAT_OPS,
            "reduced": FEAT_REDUCED,
        },
        "default_features": "selected",
        "cond_features": FEAT_OPS,
        "default_seq_len": 30,
        "stride_train": 1,
        "last_window_eval": True,
        "note": "Official test RUL hidden; held-out train units, each cut at 5 random cycles before failure.",
    }


def load_raw_n_cmapss(filename: str = "N-CMAPSS_DS01-005.h5", rul_cap: float = 100.0) -> dict:
    # NCMAPSS_DIR lets the ~28 GB .h5 files live in another local folder.
    root = Path(os.environ.get("NCMAPSS_DIR", WORKSPACE / "N-CMAPSS" / "data_set"))
    path = root / filename
    if not path.exists():
        raise FileNotFoundError(f"{path} (download N-CMAPSS .h5 files; see DATASETS.md)")
    train_df = _cycle_aggregate(path, "dev")
    test_df = _cycle_aggregate(path, "test")
    train_df["RUL"] = np.minimum(train_df["RUL"], rul_cap)
    test_df["RUL"] = np.minimum(test_df["RUL"], rul_cap)
    feat_all = [c for c in train_df.columns if c not in ("unit", "cycle", "RUL")]
    # W vars first 4, X_s rest
    feat_w = feat_all[:4]
    feat_xs = feat_all[4:]
    train_u, val_u, _ = _unit_splits(train_df["unit"].unique(), val_frac=0.2, test_frac=None)
    return {
        "name": f"N-CMAPSS-{filename.replace('N-CMAPSS_', '').replace('.h5', '')}",
        "train_df": train_df.loc[train_df["unit"].isin(train_u)].reset_index(drop=True),
        "val_df": train_df.loc[train_df["unit"].isin(val_u)].reset_index(drop=True),
        "test_df": test_df.reset_index(drop=True),
        "feature_presets": {
            "selected": feat_all,
            "all": feat_all,
            "sensors_only": feat_xs,
            "ops_only": feat_w,
            "reduced": feat_xs[:8] + feat_w[:2],
        },
        "default_features": "selected",
        "cond_features": feat_w,  # flight conditions W (alt, Mach, TRA, T2)
        "default_seq_len": 20,
        "stride_train": 1,
        # Test units run to failure, so the last window is always RUL == 0:
        # last-window scoring is meaningless here. Score continuously only
        # (every test window; standard N-CMAPSS protocol).
        "last_window_eval": False,
    }


def load_raw_scania_tte(
    max_vehicles: int = 3000,
    max_windows_per_vehicle: int = 5,
    rul_cap: float = 100.0,
) -> dict:
    """
    SCANIA as TTE/RUL regression on train vehicles only (true event times).
    Val/test are held-out vehicles — no label leakage from challenge class files.
    """
    root = WORKSPACE / "SCANIA-Component-X" / "data"
    tte = pd.read_csv(root / "train_tte.csv")
    specs = pd.read_csv(root / "train_specifications.csv")
    # Only repaired vehicles have a known time-to-event. Censored vehicles
    # (no repair in study) have no true RUL, so treating end-of-study as
    # failure would give wrong labels -> excluded.
    rng = np.random.default_rng(SEED)
    repaired = tte.loc[tte["in_study_repair"] == 1, "vehicle_id"].to_numpy()
    if len(repaired) > max_vehicles:
        repaired = rng.choice(repaired, size=max_vehicles, replace=False)
    selected = set(repaired.tolist())

    ops = pd.read_csv(root / "train_operational_readouts.csv")
    ops = ops[ops["vehicle_id"].isin(selected)].copy()
    feature_cols = [c for c in ops.columns if c not in ("vehicle_id", "time_step")]
    ops[feature_cols] = ops[feature_cols].fillna(0.0)

    tte_i = tte.set_index("vehicle_id")
    # RUL target: time until repair
    fail = ops["vehicle_id"].map(tte_i["length_of_study_time_step"]).astype(float)
    repaired_f = ops["vehicle_id"].map(tte_i["in_study_repair"]).astype(int)
    rul = fail - ops["time_step"].astype(float)
    rul = np.clip(rul, 0, rul_cap)
    ops = ops.copy()
    ops["unit"] = ops["vehicle_id"]
    ops["cycle"] = ops.groupby("vehicle_id").cumcount() + 1
    ops["RUL"] = rul.astype(float)
    ops["_repaired"] = repaired_f.to_numpy()

    # Encode specs on selected vehicles only (fit dummies on train split later would be ideal;
    # we encode on all selected then split — categories are vehicle-static, no target leakage)
    specs_sel = specs[specs["vehicle_id"].isin(selected)].copy()
    spec_oh = pd.get_dummies(specs_sel.drop(columns=["vehicle_id"]), prefix_sep="=")
    spec_cols = list(spec_oh.columns)
    spec_map = {
        int(v): spec_oh.iloc[i].to_numpy(dtype=np.float32)
        for i, v in enumerate(specs_sel["vehicle_id"].to_numpy())
    }

    # Attach specs as static columns for window builder convenience
    spec_matrix = np.vstack([spec_map[int(v)] for v in ops["vehicle_id"].to_numpy()])
    for j, col in enumerate(spec_cols):
        ops[col] = spec_matrix[:, j]

    units = ops["unit"].unique()
    train_u, val_u, test_u = _unit_splits(units, val_frac=0.15, test_frac=0.20)

    def _sub(u_set):
        return ops.loc[ops["unit"].isin(u_set)].reset_index(drop=True)

    feat_ops = feature_cols
    feat_all = feature_cols + spec_cols
    feat_specs = spec_cols
    # reduced: first 20 op cols + all specs
    feat_reduced = feature_cols[:20] + spec_cols

    # Last readout before repair has RUL ~ 0 for every vehicle -> cut held-out
    # vehicles at random readouts before repair instead.
    def _cut(u_set, salt):
        return truncate_trajectories(
            _sub(u_set), n_cuts=3, min_len=8, seed=SEED + salt
        )

    return {
        "name": "SCANIA-TTE",
        "train_df": _sub(train_u),
        "val_df": _cut(val_u, 1),
        "test_df": _cut(test_u, 2),
        "test_cont_df": _sub(test_u),
        "feature_presets": {
            "selected": feat_all,
            "all": feat_all,
            "sensors_only": feat_ops,
            "ops_only": feat_specs,  # static specs only (ablation: drop time series)
            "reduced": feat_reduced,
        },
        "default_features": "selected",
        "cond_features": [],
        "default_seq_len": 8,
        "stride_train": 1,
        "last_window_eval": True,
        # Train windows are sampled per vehicle (all ~700k windows do not fit in
        # Colab RAM); pass max_windows_per_vehicle=None for every window.
        "max_windows_per_vehicle": max_windows_per_vehicle,
        "note": "TTE regression on repaired train vehicles only; held-out vehicles cut at 3 random readouts before repair.",
    }


def load_all_raw_datasets() -> list[dict]:
    packs = []
    for fd in ("FD001", "FD002", "FD003", "FD004"):
        packs.append(load_raw_cmapss(fd))
    packs.append(load_raw_phm08())
    try:
        packs.append(load_raw_n_cmapss())
    except (FileNotFoundError, OSError) as exc:
        print(f"[skip] N-CMAPSS not available: {exc}")
    packs.append(load_raw_scania_tte())
    return packs
