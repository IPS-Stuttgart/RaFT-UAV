"""Compatibility package for the tracklet feature-store diagnostics."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_LEGACY_MODULE_NAME = "raft_uav.diagnostics._tracklet_feature_store_legacy"
_LEGACY_PATH = Path(__file__).resolve().parent.parent / "tracklet_feature_store.py"
_SPEC = importlib.util.spec_from_file_location(_LEGACY_MODULE_NAME, _LEGACY_PATH)
if _SPEC is None or _SPEC.loader is None:  # pragma: no cover - import machinery failure
    raise ImportError(f"unable to load legacy tracklet feature store from {_LEGACY_PATH}")
_legacy = importlib.util.module_from_spec(_SPEC)
sys.modules[_LEGACY_MODULE_NAME] = _legacy
_SPEC.loader.exec_module(_legacy)

for _name in dir(_legacy):
    if not _name.startswith("__"):
        globals()[_name] = getattr(_legacy, _name)


def _append_frame_keys(frame: pd.DataFrame) -> pd.DataFrame:
    """Attach stable physical-frame keys without collapsing missing indices."""

    out = frame.copy()
    frame_indices = None
    if "frame_index" in out.columns:
        frame_indices = pd.to_numeric(out["frame_index"], errors="coerce")
    complete_frame_indices = (
        frame_indices is not None
        and frame_indices.notna().all()
        and np.isfinite(frame_indices.to_numpy(dtype=float)).all()
    )
    if complete_frame_indices:
        out["frame_key_type"] = "frame_index"
        out["frame_key"] = frame_indices.round().astype("Int64").astype(str)
        return out

    if "time_s" not in out.columns:
        raise KeyError("radar candidates are missing required column 'time_s'")
    times = pd.to_numeric(out["time_s"], errors="coerce")
    out["frame_key_type"] = "time_s"
    out["frame_key"] = times.round(9).astype(str)
    return out


_legacy._append_frame_keys = _append_frame_keys
