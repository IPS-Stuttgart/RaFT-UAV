from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as _np
import pandas as _pd

_LEGACY_PATH = Path(__file__).resolve().parent.parent / "time_offset.py"
_SPEC = importlib.util.spec_from_file_location("_raft_uav_time_offset_legacy", _LEGACY_PATH)
if _SPEC is None or _SPEC.loader is None:  # pragma: no cover
    raise ImportError(f"cannot load {_LEGACY_PATH}")
_legacy = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_legacy)
_original_catprob_candidate_pool = _legacy.catprob_candidate_pool
_original_highest_catprob_candidate = _legacy.highest_catprob_candidate
_original_nearest_candidate_to_truth = _legacy.nearest_candidate_to_truth

_POSITION_COLUMNS = ("east_m", "north_m", "up_m")


def _finite_position_candidates(candidates):
    if candidates.empty:
        return candidates

    numeric_positions = candidates.loc[:, _POSITION_COLUMNS].apply(
        _pd.to_numeric,
        errors="coerce",
    )
    finite = _np.isfinite(numeric_positions.to_numpy(dtype=float)).all(axis=1)
    cleaned = candidates.loc[finite].copy()
    for column in _POSITION_COLUMNS:
        cleaned[column] = numeric_positions.loc[finite, column].to_numpy(dtype=float)
    return cleaned


def catprob_candidate_pool(candidates, threshold):
    if threshold is None:
        return candidates
    return _original_catprob_candidate_pool(candidates, threshold)


def highest_catprob_candidate(candidates):
    return _original_highest_catprob_candidate(_finite_position_candidates(candidates))


def nearest_candidate_to_truth(candidates, truth_position):
    return _original_nearest_candidate_to_truth(
        _finite_position_candidates(candidates),
        truth_position,
    )


def radar_frame_groups(radar: _pd.DataFrame) -> list[_pd.DataFrame]:
    """Group every radar row even when ``frame_index`` is partly populated."""

    if radar.empty:
        return []
    sort_columns = [
        column
        for column in ("time_s", "frame_index", "track_id", "track_index")
        if column in radar.columns
    ]
    if not sort_columns:
        raise KeyError("radar must contain time_s or frame_index")
    ordered = radar.sort_values(sort_columns).reset_index(drop=True)
    frame_index_complete = (
        "frame_index" in ordered.columns and ordered["frame_index"].notna().all()
    )
    if frame_index_complete:
        group_column = "frame_index"
    elif "time_s" in ordered.columns:
        group_column = "time_s"
    else:
        raise KeyError("radar must contain time_s when frame_index is incomplete")
    return [
        group.copy()
        for _, group in ordered.groupby(group_column, sort=True, dropna=False)
    ]


_legacy.catprob_candidate_pool = catprob_candidate_pool
_legacy.highest_catprob_candidate = highest_catprob_candidate
_legacy.nearest_candidate_to_truth = nearest_candidate_to_truth
_legacy.radar_frame_groups = radar_frame_groups

for _name, _value in vars(_legacy).items():
    if not _name.startswith("_"):
        globals()[_name] = _value
globals()["catprob_candidate_pool"] = catprob_candidate_pool
globals()["highest_catprob_candidate"] = highest_catprob_candidate
globals()["nearest_candidate_to_truth"] = nearest_candidate_to_truth
globals()["radar_frame_groups"] = radar_frame_groups
__all__ = sorted(_name for _name in globals() if not _name.startswith("_"))
