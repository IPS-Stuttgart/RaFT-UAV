"""Preserve radar frame identity when frame indices are only partially available."""

from __future__ import annotations

from importlib import import_module

import numpy as np
import pandas as pd


_feature_store = import_module("raft_uav.diagnostics.tracklet_feature_store")
_PATCH_MARKER = "_raft_uav_partial_frame_key_patch_applied"


def _append_frame_keys(frame: pd.DataFrame) -> pd.DataFrame:
    """Use finite frame indices and fall back to timestamps row by row."""

    out = frame.copy()
    times = pd.to_numeric(out["time_s"], errors="coerce")
    out["frame_key_type"] = "time_s"
    out["frame_key"] = times.round(9).astype(str)

    if "frame_index" not in out.columns:
        return out

    frame_indices = pd.to_numeric(out["frame_index"], errors="coerce")
    finite = frame_indices.notna() & np.isfinite(frame_indices)
    if not finite.any():
        return out

    frame_keys = frame_indices.where(finite).round().astype("Int64").astype(str)
    out.loc[finite, "frame_key_type"] = "frame_index"
    out.loc[finite, "frame_key"] = frame_keys.loc[finite]
    return out


def install() -> None:
    """Install the partial-frame-index fallback once per interpreter."""

    if getattr(_feature_store, _PATCH_MARKER, False):
        return

    _feature_store._append_frame_keys = _append_frame_keys
    legacy = getattr(_feature_store, "_LEGACY", None)
    if legacy is not None:
        legacy._append_frame_keys = _append_frame_keys
    setattr(_feature_store, _PATCH_MARKER, True)
