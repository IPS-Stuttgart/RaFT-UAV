"""Keep radar rows whose frame index is missing in oracle diagnostics."""

from __future__ import annotations

from importlib import import_module

import numpy as np
import pandas as pd


_radar_oracle = import_module("raft_uav.evaluation.radar_oracle_diagnostics")


def _radar_frame_groups(radar: pd.DataFrame) -> list[pd.DataFrame]:
    """Group indexed frames while falling back to time for unindexed rows.

    Pandas drops NA group keys by default.  Using ``frame_index`` merely because
    the column exists therefore used to discard every radar row whose frame
    index was missing.  Imported/external candidate tables can legitimately be
    only partially indexed, so retain those rows by grouping the unindexed
    subset by ``time_s``.
    """

    if radar.empty:
        return []

    sort_columns = [
        column
        for column in ("time_s", "frame_index", "track_id", "track_index")
        if column in radar.columns
    ]
    ordered = radar.sort_values(sort_columns).reset_index(drop=True)

    if "frame_index" not in ordered.columns:
        return [group.copy() for _, group in ordered.groupby("time_s", sort=True)]

    indexed = ordered["frame_index"].notna()
    groups = [
        group.copy()
        for _, group in ordered.loc[indexed].groupby("frame_index", sort=True)
    ]
    if (~indexed).any():
        groups.extend(
            group.copy()
            for _, group in ordered.loc[~indexed].groupby("time_s", sort=True)
        )

    def _time_key(group: pd.DataFrame) -> float:
        times = pd.to_numeric(group["time_s"], errors="coerce").to_numpy(dtype=float)
        finite = times[np.isfinite(times)]
        return float(np.median(finite)) if finite.size else float("inf")

    groups.sort(key=_time_key)
    return groups


def install() -> None:
    """Install the grouping fix on public and legacy module entry points."""

    if getattr(_radar_oracle, "_missing_frame_index_patch_applied", False):
        return
    _radar_oracle._radar_frame_groups = _radar_frame_groups
    implementation = getattr(_radar_oracle, "_IMPL", None)
    if implementation is not None:
        implementation._radar_frame_groups = _radar_frame_groups
    _radar_oracle._missing_frame_index_patch_applied = True


install()
