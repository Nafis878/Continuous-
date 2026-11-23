"""N-CMAPSS cycle-level aggregation (memory-safe). Loading lives in ablation_data.py."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pandas as pd



def _decode_vars(arr) -> list[str]:
    out = []
    for x in np.array(arr):
        if isinstance(x, (bytes, np.bytes_)):
            out.append(x.decode())
        else:
            out.append(str(x))
    return out


def _cycle_aggregate(h5_path: Path, split: str = "dev") -> pd.DataFrame:
    """Mean-aggregate W + X_s per (unit, cycle); RUL is cycle-constant."""
    with h5py.File(h5_path, "r") as hdf:
        A = np.array(hdf[f"A_{split}"])
        Y = np.array(hdf[f"Y_{split}"]).ravel()
        W = np.array(hdf[f"W_{split}"])
        Xs = np.array(hdf[f"X_s_{split}"])
        W_var = _decode_vars(hdf["W_var"])
        Xs_var = _decode_vars(hdf["X_s_var"])

    unit = A[:, 0].astype(np.int32)
    cycle = A[:, 1].astype(np.int32)
    # vectorized group keys
    key = unit.astype(np.int64) * 100_000 + cycle.astype(np.int64)
    order = np.argsort(key, kind="mergesort")
    key_s = key[order]
    breaks = np.flatnonzero(np.r_[True, key_s[1:] != key_s[:-1], True])

    rows = []
    feat_names = W_var + Xs_var
    for i in range(len(breaks) - 1):
        sl = order[breaks[i] : breaks[i + 1]]
        u = int(unit[sl[0]])
        c = int(cycle[sl[0]])
        feats = np.concatenate([W[sl].mean(0), Xs[sl].mean(0)])
        rows.append((u, c, float(Y[sl[0]]), *feats.tolist()))

    cols = ["unit", "cycle", "RUL"] + feat_names
    return pd.DataFrame(rows, columns=cols)
