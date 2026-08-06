"""Preserve physical radar frame identity in tracklet diagnostics."""

from __future__ import annotations

from importlib import import_module
from types import ModuleType
from typing import Iterator

import numpy as np
import pandas as pd


_feature_store = import_module("raft_uav.diagnostics.tracklet_feature_store")
_PATCH_MARKER = "_raft_uav_partial_frame_key_patch_applied"


def _append_frame_keys(frame: pd.DataFrame) -> pd.DataFrame:
    """Use timestamp-qualified frame indices and row-wise timestamp fallback."""

    out = frame.copy()
    if "time_s" in out.columns:
        times = pd.to_numeric(out["time_s"], errors="coerce")
    else:
        times = pd.Series(np.nan, index=out.index, dtype=float)
    time_keys = times.round(9).astype(str)
    out["frame_key_type"] = "time_s"
    out["frame_key"] = time_keys

    if "frame_index" not in out.columns:
        return out

    frame_indices = pd.to_numeric(out["frame_index"], errors="coerce")
    finite_frame = frame_indices.notna() & np.isfinite(frame_indices)
    if not finite_frame.any():
        return out

    frame_keys = frame_indices.where(finite_frame).round().astype("Int64").astype(str)
    finite_time = times.notna() & np.isfinite(times)
    indexed_with_time = finite_frame & finite_time
    indexed_without_time = finite_frame & ~finite_time

    out.loc[indexed_without_time, "frame_key_type"] = "frame_index"
    out.loc[indexed_without_time, "frame_key"] = frame_keys.loc[indexed_without_time]
    out.loc[indexed_with_time, "frame_key_type"] = "frame_index_time_s"
    out.loc[indexed_with_time, "frame_key"] = (
        frame_keys.loc[indexed_with_time] + "@" + time_keys.loc[indexed_with_time]
    )
    return out


def _patch_targets() -> Iterator[ModuleType]:
    """Yield the public compatibility module and its implementation modules."""

    seen: set[int] = set()
    for target in (
        _feature_store,
        getattr(_feature_store, "_IMPL", None),
        getattr(_feature_store, "_LEGACY", None),
    ):
        if not isinstance(target, ModuleType) or id(target) in seen:
            continue
        seen.add(id(target))
        yield target


def install() -> None:
    """Install the frame-key helper at every active compatibility boundary."""

    for target in _patch_targets():
        target._append_frame_keys = _append_frame_keys
    setattr(_feature_store, _PATCH_MARKER, True)
