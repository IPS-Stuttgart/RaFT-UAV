"""Compatibility package rejecting ambiguous trajectory-completion inputs.

The maintained implementation lives in the sibling ``completion.py`` module.
This package preserves the public import path while requiring at most one result
row for each ``(sequence_id, timestamp)`` key before interpolation or holding.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pandas as pd

_IMPL_PATH = Path(__file__).resolve().parent.parent / "completion.py"
_SPEC = importlib.util.spec_from_file_location(
    "raft_uav.mmuad._completion_legacy",
    _IMPL_PATH,
)
if _SPEC is None or _SPEC.loader is None:  # pragma: no cover
    raise ImportError(f"cannot load trajectory completion implementation from {_IMPL_PATH}")
_IMPL = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _IMPL
_SPEC.loader.exec_module(_IMPL)

_ORIGINAL_COMPLETION_RESULT_ROWS = _IMPL._completion_result_rows


def _duplicate_completion_keys(rows: pd.DataFrame) -> pd.DataFrame:
    """Return duplicate normalized result keys in deterministic order."""

    frame = pd.DataFrame(rows)
    required = {"sequence_id", "timestamp"}
    if frame.empty or not required.issubset(frame.columns):
        return pd.DataFrame(columns=["sequence_id", "timestamp"])
    duplicate = frame.duplicated(
        subset=["sequence_id", "timestamp"],
        keep=False,
    )
    return (
        frame.loc[duplicate, ["sequence_id", "timestamp"]]
        .drop_duplicates()
        .sort_values(["sequence_id", "timestamp"])
        .reset_index(drop=True)
    )


def _completion_result_rows(results):
    """Normalize completion inputs and reject ambiguous duplicate timestamps."""

    rows = _ORIGINAL_COMPLETION_RESULT_ROWS(results)
    duplicate_keys = _duplicate_completion_keys(rows)
    if duplicate_keys.empty:
        return rows

    sample = ", ".join(
        f"{row.sequence_id}@{float(row.timestamp):g}"
        for row in duplicate_keys.head(5).itertuples(index=False)
    )
    suffix = ", ..." if len(duplicate_keys) > 5 else ""
    raise ValueError(
        "trajectory completion results contain "
        f"{len(duplicate_keys)} duplicate (sequence_id, timestamp) key(s): "
        f"{sample}{suffix}"
    )


_IMPL._completion_result_rows = _completion_result_rows

globals().update(
    {
        name: getattr(_IMPL, name)
        for name in dir(_IMPL)
        if not (name.startswith("__") and name.endswith("__"))
    }
)
globals()["_duplicate_completion_keys"] = _duplicate_completion_keys
globals()["_completion_result_rows"] = _completion_result_rows

__doc__ = _IMPL.__doc__
__all__ = [
    name for name in dir(_IMPL) if not (name.startswith("__") and name.endswith("__"))
]
