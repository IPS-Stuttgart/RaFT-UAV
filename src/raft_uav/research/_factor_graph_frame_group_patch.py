"""Keep reused radar frame counters separate in factor-graph association."""

from __future__ import annotations

from functools import wraps
from types import ModuleType

import numpy as np
import pandas as pd

_PATCH_MARKER = "_raft_uav_groups_factor_graph_frames_by_index_and_time"


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
        times = pd.to_numeric(ordered["time_s"], errors="coerce")
        if "frame_index" in ordered.columns:
            frame_indices = pd.to_numeric(
                ordered["frame_index"],
                errors="coerce",
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
