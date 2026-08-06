"""Keep final same-time truth rows during Track 5 estimate calibration."""

from __future__ import annotations

from functools import wraps
from typing import Any, Callable

import numpy as np
import pandas as pd

_PATCH_MARKER = "_raft_uav_prefers_final_track5_estimate_truth"
_ORDER_COLUMN = "_raft_uav_truth_input_order"
_KEY_COLUMN = "_raft_uav_truth_time_key"


def install() -> None:
    """Install final-finite duplicate truth handling for estimate calibration."""

    from raft_uav.mmuad import track5_estimate_calibration as calibration_module

    original: Callable[..., Any] = calibration_module._fit_pairs
    if getattr(original, _PATCH_MARKER, False):
        return

    @wraps(original)
    def fit_pairs(estimates, truth):
        truth_rows = pd.DataFrame(truth).copy()
        required = ("sequence_id", "time_s", "x_m", "y_m", "z_m")
        if truth_rows.empty or any(column not in truth_rows.columns for column in required):
            return original(estimates, truth)

        truth_rows[_ORDER_COLUMN] = np.arange(len(truth_rows), dtype=np.int64)
        truth_rows["sequence_id"] = truth_rows["sequence_id"].astype(str)
        numeric_columns = ("time_s", "x_m", "y_m", "z_m")
        for column in numeric_columns:
            truth_rows[column] = pd.to_numeric(truth_rows[column], errors="coerce")
        finite = np.isfinite(truth_rows[list(numeric_columns)].to_numpy()).all(axis=1)
        usable = truth_rows.loc[finite].copy()
        if usable.empty:
            return original(estimates, truth)

        usable[_KEY_COLUMN] = calibration_module._time_key(usable["time_s"])
        usable = usable.drop_duplicates(
            subset=["sequence_id", _KEY_COLUMN],
            keep="last",
        )
        usable = usable.sort_values(_ORDER_COLUMN, kind="stable")
        canonical = usable.drop(columns=[_ORDER_COLUMN, _KEY_COLUMN])
        return original(estimates, canonical)

    setattr(fit_pairs, _PATCH_MARKER, True)
    calibration_module._fit_pairs = fit_pairs
