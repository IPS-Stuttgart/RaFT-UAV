"""Compatibility package preserving radar frames with incomplete indices.

The maintained implementation lives in the sibling ``diagnostics.py`` module.
This package preserves the public import path while making radar-frame grouping
fall back to timestamps whenever frame-index metadata are incomplete.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pandas as pd

_LEGACY_PATH = Path(__file__).resolve().parent.parent / "diagnostics.py"
_SPEC = importlib.util.spec_from_file_location(
    "raft_uav.research._diagnostics_legacy",
    _LEGACY_PATH,
)
if _SPEC is None or _SPEC.loader is None:  # pragma: no cover
    raise ImportError(f"cannot load research diagnostics from {_LEGACY_PATH}")
_LEGACY = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _LEGACY
_SPEC.loader.exec_module(_LEGACY)


def _radar_frame_groups(
    radar: pd.DataFrame,
) -> list[tuple[tuple[str, int | float], pd.DataFrame]]:
    """Group every physical radar frame despite partially missing frame indices."""

    if radar.empty:
        return []
    sort_columns = [
        column
        for column in ("time_s", "frame_index", "track_id")
        if column in radar.columns
    ]
    ordered = radar.sort_values(sort_columns, kind="mergesort").copy()
    frame_indices = None
    if "frame_index" in ordered.columns:
        frame_indices = pd.to_numeric(ordered["frame_index"], errors="coerce")
    use_frame_index = frame_indices is not None and bool(
        np.isfinite(frame_indices.to_numpy(dtype=float)).all()
    )
    if use_frame_index:
        ordered["_research_diagnostic_frame_key"] = frame_indices.to_numpy(
            dtype=float
        )
        key_kind = "frame_index"
    else:
        times = pd.to_numeric(ordered["time_s"], errors="coerce")
        finite = np.isfinite(times.to_numpy(dtype=float))
        ordered = ordered.loc[finite].copy()
        ordered["_research_diagnostic_frame_key"] = times.loc[finite].round(9)
        key_kind = "time_s"

    groups: list[tuple[tuple[str, int | float], pd.DataFrame]] = []
    for key, group in ordered.groupby(
        "_research_diagnostic_frame_key",
        sort=True,
    ):
        event_key = (
            ("frame_index", int(float(key)))
            if key_kind == "frame_index"
            else ("time_s", round(float(group["time_s"].median()), 9))
        )
        groups.append(
            (
                event_key,
                group.drop(columns="_research_diagnostic_frame_key").copy(),
            )
        )
    return groups


_LEGACY._radar_frame_groups = _radar_frame_groups

for _name in dir(_LEGACY):
    if not (_name.startswith("__") and _name.endswith("__")):
        globals()[_name] = getattr(_LEGACY, _name)
globals()["_radar_frame_groups"] = _radar_frame_groups

__doc__ = _LEGACY.__doc__
__all__ = [
    name
    for name in dir(_LEGACY)
    if not (name.startswith("__") and name.endswith("__"))
]
