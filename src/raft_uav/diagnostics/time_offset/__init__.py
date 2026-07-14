from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

_LEGACY_PATH = Path(__file__).resolve().parent.parent / "time_offset.py"
_SPEC = importlib.util.spec_from_file_location("_raft_uav_time_offset_legacy", _LEGACY_PATH)
if _SPEC is None or _SPEC.loader is None:  # pragma: no cover
    raise ImportError(f"cannot load {_LEGACY_PATH}")
_legacy = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_legacy)
_original_best_offset_row = _legacy.best_offset_row
_original_catprob_candidate_pool = _legacy.catprob_candidate_pool


def best_offset_row(sweep, *, objective):
    """Select the lowest-error offset among candidates with maximum support."""

    if "matched_count" not in sweep.columns:
        return _original_best_offset_row(sweep, objective=objective)

    column = _legacy.OBJECTIVE_COLUMNS[objective]
    objective_values = pd.to_numeric(sweep[column], errors="coerce")
    matched_counts = pd.to_numeric(sweep["matched_count"], errors="coerce")
    eligible = np.isfinite(objective_values.to_numpy(dtype=float)) & np.isfinite(
        matched_counts.to_numpy(dtype=float)
    )
    if not eligible.any():
        return _original_best_offset_row(sweep, objective=objective)

    maximum_support = float(matched_counts.loc[eligible].max())
    eligible &= matched_counts.to_numpy(dtype=float) == maximum_support
    return _original_best_offset_row(sweep.loc[eligible], objective=objective)


def catprob_candidate_pool(candidates, threshold):
    if threshold is None:
        return candidates
    return _original_catprob_candidate_pool(candidates, threshold)


_legacy.best_offset_row = best_offset_row
_legacy.catprob_candidate_pool = catprob_candidate_pool

for _name, _value in vars(_legacy).items():
    if not _name.startswith("_"):
        globals()[_name] = _value
globals()["best_offset_row"] = best_offset_row
globals()["catprob_candidate_pool"] = catprob_candidate_pool
__all__ = sorted(_name for _name in globals() if not _name.startswith("_"))
