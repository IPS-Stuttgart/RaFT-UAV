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


_legacy.catprob_candidate_pool = catprob_candidate_pool
_legacy.highest_catprob_candidate = highest_catprob_candidate
_legacy.nearest_candidate_to_truth = nearest_candidate_to_truth

for _name, _value in vars(_legacy).items():
    if not _name.startswith("_"):
        globals()[_name] = _value
globals()["catprob_candidate_pool"] = catprob_candidate_pool
globals()["highest_catprob_candidate"] = highest_catprob_candidate
globals()["nearest_candidate_to_truth"] = nearest_candidate_to_truth
__all__ = sorted(_name for _name in globals() if not _name.startswith("_"))
