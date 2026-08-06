"""Preserve physical radar frame identity in tracklet diagnostics."""

from __future__ import annotations

from importlib import import_module
from types import ModuleType
from typing import Iterator

import numpy as np
import pandas as pd


_feature_store = import_module("raft_uav.diagnostics.tracklet_feature_store")
_ORIGINAL_SELECTION_MASK = _feature_store._selection_mask
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


def _stable_identifier(value: object) -> object | None:
    """Delegate identifier normalization to the active compatibility layer."""

    return _feature_store._stable_identifier(value)


def _frame_index_identity(row: pd.Series) -> object | None:
    """Return a stable raw frame-index identity from normalized or legacy rows."""

    frame_index = _stable_identifier(row.get("frame_index"))
    if frame_index is not None:
        return frame_index

    key_type = str(row.get("frame_key_type", ""))
    frame_key = row.get("frame_key")
    if key_type == "frame_index_time_s":
        frame_key = str(frame_key).partition("@")[0]
    elif key_type != "frame_index":
        return None
    return _stable_identifier(frame_key)


def _identifier_fallback_keys(row: pd.Series) -> tuple[tuple[object, ...], ...]:
    """Return raw-frame keys for the row's available candidate identifiers."""

    frame_index = _frame_index_identity(row)
    if frame_index is None:
        return ()
    keys: list[tuple[object, ...]] = []
    track_id = _stable_identifier(row.get("track_id"))
    track_index = _stable_identifier(row.get("track_index"))
    if track_id is not None:
        keys.append((frame_index, "track_id", track_id))
    if track_index is not None:
        keys.append((frame_index, "track_index", track_index))
    return tuple(keys)


def _selection_mask(
    features: pd.DataFrame,
    selected_radar: pd.DataFrame | None,
) -> np.ndarray:
    """Match timestamped frames exactly and unambiguous frame-only selections."""

    exact = np.asarray(
        _ORIGINAL_SELECTION_MASK(features, selected_radar),
        dtype=bool,
    )
    if selected_radar is None or selected_radar.empty or exact.all():
        return exact

    selected = _append_frame_keys(pd.DataFrame(selected_radar))
    frame_only = selected.loc[selected["frame_key_type"].eq("frame_index")]
    if frame_only.empty:
        return exact

    selected_track_ids: set[tuple[object, ...]] = set()
    selected_track_indices: set[tuple[object, ...]] = set()
    fallback_track_indices: set[tuple[object, ...]] = set()
    for _, row in frame_only.iterrows():
        keys = _identifier_fallback_keys(row)
        track_id = _stable_identifier(row.get("track_id"))
        for key in keys:
            if key[1] == "track_id":
                selected_track_ids.add(key)
            else:
                selected_track_indices.add(key)
                if track_id is None:
                    fallback_track_indices.add(key)

    physical_frames: dict[tuple[object, ...], set[tuple[object, object]]] = {}
    for _, row in pd.DataFrame(features).iterrows():
        physical_key = (row.get("frame_key_type"), row.get("frame_key"))
        for key in _identifier_fallback_keys(row):
            physical_frames.setdefault(key, set()).add(physical_key)

    def is_unambiguous(key: tuple[object, ...]) -> bool:
        return len(physical_frames.get(key, ())) == 1

    result = exact.copy()
    for position, (_, row) in enumerate(pd.DataFrame(features).iterrows()):
        if result[position]:
            continue
        track_id = _stable_identifier(row.get("track_id"))
        track_index = _stable_identifier(row.get("track_index"))
        frame_index = _frame_index_identity(row)
        if frame_index is None:
            continue
        if track_id is not None:
            id_key = (frame_index, "track_id", track_id)
            if id_key in selected_track_ids and is_unambiguous(id_key):
                result[position] = True
                continue
        if track_index is None:
            continue
        index_key = (frame_index, "track_index", track_index)
        allowed = selected_track_indices if track_id is None else fallback_track_indices
        if index_key in allowed and is_unambiguous(index_key):
            result[position] = True
    return result


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
    """Install frame-key and selection helpers at every active boundary."""

    for target in _patch_targets():
        target._append_frame_keys = _append_frame_keys
        target._selection_mask = _selection_mask
    setattr(_feature_store, _PATCH_MARKER, True)
