"""Keep geometric-median ensemble rows tied to exact template timestamps."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _exact_template_time_matches(values: pd.Series, target: float) -> np.ndarray:
    """Match only the exact template timestamp already used during resampling."""

    numeric = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    return np.isfinite(numeric) & (numeric == float(target))


def install() -> None:
    """Install exact row matching into the active geometric-median implementation."""

    from raft_uav.mmuad import track5_geometric_median_ensemble as module

    module._template_time_matches = _exact_template_time_matches
    implementation = getattr(module, "_IMPL", None)
    if implementation is not None:
        implementation._template_time_matches = _exact_template_time_matches
    module._exact_template_time_match_patch_applied = True
