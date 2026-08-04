"""Order research track-switch diagnostics by normalized numeric timestamps."""

from __future__ import annotations

from functools import wraps
from types import ModuleType
from typing import Callable

import pandas as pd

from raft_uav.numeric import optional_float

_PATCH_MARKER = "_raft_uav_orders_track_switch_times_numerically"


def apply_diagnostics_numeric_time_patch(module: ModuleType) -> None:
    """Patch ``track_switch_metrics`` to normalize timestamps before sorting."""

    original: Callable[..., dict[str, object]] = module.track_switch_metrics
    if getattr(original, _PATCH_MARKER, False):
        return

    @wraps(original)
    def track_switch_metrics(
        selected: pd.DataFrame,
        *,
        long_gap_s: float = 5.0,
    ) -> dict[str, object]:
        normalized = pd.DataFrame(selected).copy()
        if "time_s" in normalized.columns:
            normalized["time_s"] = pd.Series(
                [optional_float(value) for value in normalized["time_s"]],
                index=normalized.index,
                dtype=float,
            )
        return original(normalized, long_gap_s=long_gap_s)

    setattr(track_switch_metrics, _PATCH_MARKER, True)
    module.track_switch_metrics = track_switch_metrics
