"""NASA CMAPSS column/feature definitions (loading lives in ablation_data.py)."""

from __future__ import annotations

COLS = (
    ["unit", "cycle", "op1", "op2", "op3"]
    + [f"s{i}" for i in range(1, 22)]
)

# Sensors that are typically constant / low-info on FD001; keep ops + informative sensors
FEATURE_COLS = [
    "op1",
    "op2",
    "op3",
    "s2",
    "s3",
    "s4",
    "s7",
    "s8",
    "s9",
    "s11",
    "s12",
    "s13",
    "s14",
    "s15",
    "s17",
    "s20",
    "s21",
]
