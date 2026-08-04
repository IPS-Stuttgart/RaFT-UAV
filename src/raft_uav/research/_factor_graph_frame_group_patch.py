"""Keep reused radar frame counters separate in factor-graph association."""

from __future__ import annotations

from functools import wraps
from types import ModuleType

import numpy as np
import pandas as pd

_PATCH_MARKER = "_raft_uav_groups_factor_graph_frames_by_index_and_time"


def _finite_real_scalar(value: object) -> float | None:
    """Return a finite real scalar without complex-column dtype poisoning."""

    if value is None or np.ma.is_masked(value) or isinstance(value, (bool, np.bool_)):
        return None
    try:
        array = np.asarray(value)
    except (TypeError, ValueError):
        return None
    if array.ndim != 0:
        return None
    scalar = array.item()
    if np.ma.is_masked(scalar) or isinstance(scalar, (bool, np.bool_)):
        return None
    if isinstance(scalar, (complex, np.complexfloating)):
        real = float(np.real(scalar))
        imaginary = float(np.imag(scalar))
        if not np.isfinite(real) or not np.isfinite(imaginary) or imaginary != 0.0:
            return None
        return real
    try:
        number = float(scalar)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if np.isfinite(number) else None


def apply_factor_graph_frame_group_patch(module: ModuleType) -> None:
    """Patch factor-graph radar grouping to disambiguate counter reuse."""

    implementation = getattr(module, "_LEGACY", module)
    original = implementation._radar_frame_groups
    if getattr(original, _PATCH_MARKER, False):
        module._radar_frame_groups = original
        return

    @wraps(original)
    def radar_frame_groups(
        radar: pd.DataFrame,
    ) -> list[tuple[object, pd.DataFrame]]:
        sort_cols = [
            column
            for column in ("time_s", "frame_index", "track_id")
            if column in radar.columns
        ]
        ordered = radar.sort_values(sort_cols).reset_index(drop=True)
        times = pd.Series(
            [_finite_real_scalar(value) for value in ordered["time_s"]],
            index=ordered.index,
            dtype=float,
        )
        if "frame_index" in ordered.columns:
            frame_indices = pd.Series(
                [_finite_real_scalar(value) for value in ordered["frame_index"]],
                index=ordered.index,
                dtype=float,
            )
        else:
            frame_indices = pd.Series(
                np.nan,
                index=ordered.index,
                dtype=float,
            )

        group_keys = pd.Series(
            [
                ("frame_index_time", float(frame_index), float(time_s))
                if np.isfinite(frame_index) and np.isfinite(time_s)
                else ("frame_index", float(frame_index))
                if np.isfinite(frame_index)
                else ("time_s", float(time_s))
                if np.isfinite(time_s)
                else None
                for frame_index, time_s in zip(frame_indices, times, strict=True)
            ],
            index=ordered.index,
            dtype=object,
        )
        usable = group_keys.notna()
        ordered = ordered.loc[usable]
        group_keys = group_keys.loc[usable]
        return [
            (key, group.copy())
            for key, group in ordered.groupby(group_keys, sort=False)
        ]

    setattr(radar_frame_groups, _PATCH_MARKER, True)
    implementation._radar_frame_groups = radar_frame_groups
    module._radar_frame_groups = radar_frame_groups
