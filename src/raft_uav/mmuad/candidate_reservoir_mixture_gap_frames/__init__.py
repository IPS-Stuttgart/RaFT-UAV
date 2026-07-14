"""Compatibility wrapper for safe exact frame-gap joins.

The maintained implementation lives in the sibling
``candidate_reservoir_mixture_gap_frames.py`` module. This package preserves the
public import path while preventing invalid or ambiguous rounded timestamp keys
from corrupting frame-level gap diagnostics.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from typing import Any

import pandas as pd

_IMPL_PATH = Path(__file__).resolve().parent.parent / (
    "candidate_reservoir_mixture_gap_frames.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "raft_uav.mmuad._candidate_reservoir_mixture_gap_frames_legacy",
    _IMPL_PATH,
)
if _SPEC is None or _SPEC.loader is None:  # pragma: no cover
    raise ImportError(f"cannot load frame-gap implementation from {_IMPL_PATH}")
_IMPL = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _IMPL
_SPEC.loader.exec_module(_IMPL)


def _require_unique_frame_keys(rows: pd.DataFrame, *, label: str) -> None:
    """Reject ambiguous frame keys before a one-to-one diagnostic join."""

    key_columns = ["sequence_id", "_join_time_s"]
    duplicated = rows.duplicated(key_columns, keep=False)
    if not bool(duplicated.any()):
        return
    examples: list[dict[str, Any]] = (
        rows.loc[duplicated, key_columns]
        .drop_duplicates()
        .head(5)
        .to_dict(orient="records")
    )
    raise ValueError(
        f"{label} contain duplicate frame keys after timestamp rounding: {examples}"
    )


def _join_estimates_to_oracle_exact(
    estimate_rows: pd.DataFrame,
    oracle_rows: pd.DataFrame,
    *,
    time_round_decimals: int,
) -> pd.DataFrame:
    """Join only finite, unique per-sequence rounded frame timestamps."""

    estimate_rows = estimate_rows.copy()
    oracle_rows = oracle_rows.copy()
    estimate_rows["_join_time_s"] = _IMPL._rounded_time(
        estimate_rows["time_s"],
        time_round_decimals,
    )
    oracle_rows["_join_time_s"] = _IMPL._rounded_time(
        oracle_rows["time_s"],
        time_round_decimals,
    )
    estimate_rows = estimate_rows.loc[estimate_rows["_join_time_s"].notna()].copy()
    oracle_rows = oracle_rows.loc[oracle_rows["_join_time_s"].notna()].copy()
    _require_unique_frame_keys(estimate_rows, label="estimates")
    _require_unique_frame_keys(oracle_rows, label="oracle frames")
    return oracle_rows.merge(
        estimate_rows,
        on=["sequence_id", "_join_time_s"],
        how="inner",
        suffixes=("_oracle", "_mixture"),
        validate="one_to_one",
    )


_IMPL._join_estimates_to_oracle_exact = _join_estimates_to_oracle_exact

globals().update(
    {
        name: getattr(_IMPL, name)
        for name in dir(_IMPL)
        if not (name.startswith("__") and name.endswith("__"))
    }
)
globals()["_require_unique_frame_keys"] = _require_unique_frame_keys
globals()["_join_estimates_to_oracle_exact"] = _join_estimates_to_oracle_exact

__doc__ = _IMPL.__doc__
__all__ = [
    name for name in dir(_IMPL) if not (name.startswith("__") and name.endswith("__"))
]
