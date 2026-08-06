"""Keep final duplicate truth samples in time-offset diagnostics."""

from __future__ import annotations

from importlib import import_module

import numpy as np
import pandas as pd

from raft_uav.numeric import optional_float


_time_offset = import_module("raft_uav.diagnostics.time_offset")
_PATCH_MARKER = "_raft_uav_time_offset_final_duplicate_truth_patch_applied"
_ORIGINAL_TRUTH_POSITIONS_AT_TIMES = _time_offset.truth_positions_at_times


def _truth_with_final_duplicate_samples(truth: pd.DataFrame) -> pd.DataFrame:
    """Keep the final row at each finite numeric truth timestamp."""

    rows = pd.DataFrame(truth).copy()
    if rows.empty or "time_s" not in rows.columns:
        return rows

    time_keys = pd.Series(
        [optional_float(value) for value in rows["time_s"]],
        index=rows.index,
        dtype=float,
    )
    finite = time_keys.notna()
    duplicate = pd.Series(False, index=rows.index, dtype=bool)
    duplicate.loc[finite] = time_keys.loc[finite].duplicated(keep="last")
    return rows.loc[~duplicate].copy()


def truth_positions_at_times(
    truth: pd.DataFrame,
    times_s: np.ndarray,
    *,
    max_delta_s: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate after applying the shared final-sample convention."""

    return _ORIGINAL_TRUTH_POSITIONS_AT_TIMES(
        _truth_with_final_duplicate_samples(truth),
        times_s,
        max_delta_s=max_delta_s,
    )


def install() -> None:
    """Install final-sample duplicate handling once per interpreter."""

    if getattr(_time_offset, _PATCH_MARKER, False):
        return
    _time_offset.truth_positions_at_times = truth_positions_at_times
    legacy = getattr(_time_offset, "_legacy", None)
    if legacy is not None:
        legacy.truth_positions_at_times = truth_positions_at_times
    setattr(_time_offset, _PATCH_MARKER, True)


install()
