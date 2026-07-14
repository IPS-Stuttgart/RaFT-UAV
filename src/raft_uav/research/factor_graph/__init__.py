"""Compatibility package for the factor-graph research utilities."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pandas as pd

_LEGACY_PATH = Path(__file__).resolve().parent.parent / "factor_graph.py"
_LEGACY_NAME = f"{__name__.rsplit('.', 1)[0]}._factor_graph_legacy"
_SPEC = importlib.util.spec_from_file_location(_LEGACY_NAME, _LEGACY_PATH)
if _SPEC is None or _SPEC.loader is None:  # pragma: no cover - import machinery failure
    raise ImportError(f"cannot load factor-graph implementation from {_LEGACY_PATH}")
_LEGACY = importlib.util.module_from_spec(_SPEC)
sys.modules[_LEGACY_NAME] = _LEGACY
_SPEC.loader.exec_module(_LEGACY)

for _name in dir(_LEGACY):
    if not _name.startswith("__"):
        globals()[_name] = getattr(_LEGACY, _name)


def _radar_frame_groups(radar: pd.DataFrame) -> list[tuple[object, pd.DataFrame]]:
    """Group physical radar frames without dropping partially indexed rows."""

    sort_cols = [c for c in ("time_s", "frame_index", "track_id") if c in radar.columns]
    ordered = radar.sort_values(sort_cols).reset_index(drop=True)
    frame_indices = None
    if "frame_index" in ordered.columns:
        frame_indices = pd.to_numeric(ordered["frame_index"], errors="coerce")
    if frame_indices is not None and np.isfinite(frame_indices.to_numpy(dtype=float)).all():
        group_keys = frame_indices
    else:
        group_keys = pd.to_numeric(ordered["time_s"], errors="coerce")
    ordered = ordered.assign(_frame_group_key=group_keys)
    finite = np.isfinite(ordered["_frame_group_key"].to_numpy(dtype=float))
    ordered = ordered.loc[finite]
    return [
        (key, group.drop(columns="_frame_group_key").copy())
        for key, group in ordered.groupby("_frame_group_key", sort=True)
    ]


_LEGACY._radar_frame_groups = _radar_frame_groups
