"""Compatibility fixes for source-branch reservoir caps and summaries.

The maintained implementation lives in the sibling
``candidate_source_branch_reservoir.py`` module. That implementation builds an
uncapped base reservoir and then merges additional ``source_branch:`` reasons.
Base rows already carry derived reason-count and cap-score columns, while rows
introduced only by the source-branch quota have those columns missing. Passing
that mixed table directly to the final cap can therefore reuse stale counts.

This package preserves the public import path, removes those derived cache
columns before the final cap, and normalizes serialized Boolean selection flags
before summary counts are computed.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from typing import Any

import pandas as pd

from raft_uav.mmuad.candidate_reservoir import _boolean_series

_IMPL_PATH = Path(__file__).resolve().parent.parent / "candidate_source_branch_reservoir.py"
_SPEC = importlib.util.spec_from_file_location(
    "raft_uav.mmuad._candidate_source_branch_reservoir_legacy",
    _IMPL_PATH,
)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(
        "cannot load source-branch reservoir implementation from "
        f"{_IMPL_PATH}"
    )
_IMPL = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _IMPL
_SPEC.loader.exec_module(_IMPL)

_ORIGINAL_APPLY_FRAME_CAP = _IMPL._apply_frame_cap
_ORIGINAL_SOURCE_BRANCH_RESERVOIR_SUMMARY = _IMPL.source_branch_reservoir_summary
_DERIVED_CAP_COLUMNS = (
    "candidate_reservoir_reason_count",
    "candidate_reservoir_cap_score",
)
_BOOLEAN_SUMMARY_COLUMNS = (
    "candidate_source_branch_quota_selected",
    "candidate_source_branch_diversity_selected",
)


def _apply_frame_cap_with_fresh_reason_counts(
    rows: pd.DataFrame,
    **kwargs: Any,
) -> pd.DataFrame:
    """Recompute reason-derived cap columns after quota reasons are merged."""

    refreshed = rows.drop(columns=list(_DERIVED_CAP_COLUMNS), errors="ignore")
    return _ORIGINAL_APPLY_FRAME_CAP(refreshed, **kwargs)


def source_branch_reservoir_summary(
    candidates: Any,
    reservoir: Any,
) -> dict[str, Any]:
    """Build summary counts with serialized source-branch flags normalized."""

    normalized = _IMPL._candidate_rows(reservoir)
    for column in _BOOLEAN_SUMMARY_COLUMNS:
        if column in normalized.columns:
            normalized[column] = _boolean_series(
                normalized[column],
                normalized.index,
            )
    return _ORIGINAL_SOURCE_BRANCH_RESERVOIR_SUMMARY(candidates, normalized)


_IMPL._apply_frame_cap = _apply_frame_cap_with_fresh_reason_counts
_IMPL._cap_per_frame = _apply_frame_cap_with_fresh_reason_counts
_IMPL.source_branch_reservoir_summary = source_branch_reservoir_summary

globals().update(
    {
        name: getattr(_IMPL, name)
        for name in dir(_IMPL)
        if not (name.startswith("__") and name.endswith("__"))
    }
)
globals()["_apply_frame_cap_with_fresh_reason_counts"] = (
    _apply_frame_cap_with_fresh_reason_counts
)
globals()["source_branch_reservoir_summary"] = source_branch_reservoir_summary

__doc__ = _IMPL.__doc__
__all__ = [
    name for name in dir(_IMPL) if not (name.startswith("__") and name.endswith("__"))
]
