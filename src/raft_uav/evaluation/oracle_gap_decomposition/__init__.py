"""Compatibility fix for confidence-diagnostic selected-context merges.

The maintained implementation lives in the sibling
``oracle_gap_decomposition.py`` module. This package preserves the public import
path while keeping estimate columns, row order, and invalid-time rows intact when
selected-radar context is attached.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pandas as pd

_IMPL_PATH = Path(__file__).resolve().parent.parent / "oracle_gap_decomposition.py"
_SPEC = importlib.util.spec_from_file_location(
    "raft_uav.evaluation._oracle_gap_decomposition_legacy",
    _IMPL_PATH,
)
if _SPEC is None or _SPEC.loader is None:  # pragma: no cover
    raise ImportError(
        f"cannot load oracle-gap decomposition implementation from {_IMPL_PATH}"
    )
_IMPL = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _IMPL
_SPEC.loader.exec_module(_IMPL)

_CONTEXT_COLUMNS = (
    "track_id",
    "association_score",
    "association_nis",
    "association_weight_entropy",
    "association_hypothesis_count",
)
_ROW_ORDER_COLUMN = "__raft_uav_confidence_row_order"
_MERGE_TIME_COLUMN = "__raft_uav_confidence_merge_time_s"


def _merge_selected_context(
    estimates: pd.DataFrame,
    selected: pd.DataFrame,
) -> pd.DataFrame:
    """Attach selected-radar context without corrupting estimate columns or order."""

    if "time_s" not in estimates.columns or "time_s" not in selected.columns:
        return estimates

    available_context = [
        column for column in _CONTEXT_COLUMNS if column in selected.columns
    ]
    if not available_context:
        return estimates

    left = estimates.copy()
    original_index = left.index
    left[_ROW_ORDER_COLUMN] = np.arange(len(left), dtype=np.int64)
    left[_MERGE_TIME_COLUMN] = pd.to_numeric(left["time_s"], errors="coerce")

    context = selected[["time_s", *available_context]].copy()
    context[_MERGE_TIME_COLUMN] = pd.to_numeric(
        context["time_s"],
        errors="coerce",
    )
    renamed_context = {
        column: f"selected_context_{column}" for column in available_context
    }
    context = context.drop(columns=["time_s"]).rename(columns=renamed_context)
    context = context.loc[
        np.isfinite(context[_MERGE_TIME_COLUMN].to_numpy(dtype=float))
    ].sort_values(_MERGE_TIME_COLUMN)

    output_columns = list(renamed_context.values())
    valid_left = np.isfinite(left[_MERGE_TIME_COLUMN].to_numpy(dtype=float))
    if context.empty or not valid_left.any():
        for column in output_columns:
            left[column] = np.nan
        merged = left
    else:
        matched = pd.merge_asof(
            left.loc[valid_left].sort_values(_MERGE_TIME_COLUMN),
            context,
            on=_MERGE_TIME_COLUMN,
            direction="nearest",
            tolerance=0.25,
        )
        unmatched = left.loc[~valid_left].copy()
        for column in output_columns:
            unmatched[column] = np.nan
        merged = pd.concat([matched, unmatched], ignore_index=True, sort=False)

    merged = merged.sort_values(_ROW_ORDER_COLUMN).drop(
        columns=[_ROW_ORDER_COLUMN, _MERGE_TIME_COLUMN]
    )
    merged.index = original_index
    return merged


_IMPL._merge_selected_context = _merge_selected_context

globals().update(
    {
        name: getattr(_IMPL, name)
        for name in dir(_IMPL)
        if not (name.startswith("__") and name.endswith("__"))
    }
)
globals()["_merge_selected_context"] = _merge_selected_context

__doc__ = _IMPL.__doc__
__all__ = [
    name for name in dir(_IMPL) if not (name.startswith("__") and name.endswith("__"))
]
