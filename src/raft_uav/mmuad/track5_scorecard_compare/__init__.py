"""Compatibility wrapper rejecting duplicate scorecard sequence rows.

The maintained implementation lives in the sibling
``track5_scorecard_compare.py`` module. This package preserves the public
import path while preventing duplicate sequence summaries from being silently
reduced to the first row.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pandas as pd

_IMPL_PATH = Path(__file__).resolve().parent.parent / "track5_scorecard_compare.py"
_SPEC = importlib.util.spec_from_file_location(
    "raft_uav.mmuad._track5_scorecard_compare_legacy",
    _IMPL_PATH,
)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"cannot load Track 5 scorecard comparison from {_IMPL_PATH}")
_IMPL = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _IMPL
_SPEC.loader.exec_module(_IMPL)

_ORIGINAL_NORMALIZE_POSE_BY_SEQUENCE_TABLE = _IMPL._normalize_pose_by_sequence_table


def _duplicate_sequence_ids(rows: pd.DataFrame) -> list[str]:
    """Return duplicate normalized sequence identifiers in input order."""

    frame = pd.DataFrame(rows)
    if frame.empty:
        return []
    sequence_column = "sequence_id" if "sequence_id" in frame.columns else "sequence"
    if sequence_column not in frame.columns:
        return []
    sequence_ids = frame[sequence_column].astype(str)
    duplicate_mask = sequence_ids.duplicated(keep=False)
    return sequence_ids.loc[duplicate_mask].drop_duplicates().tolist()


def _normalize_pose_by_sequence_table(
    rows: pd.DataFrame,
    *,
    label: str,
) -> pd.DataFrame:
    """Normalize one scorecard table without discarding duplicate sequences."""

    duplicate_ids = _duplicate_sequence_ids(rows)
    if duplicate_ids:
        preview = ", ".join(repr(value) for value in duplicate_ids[:5])
        suffix = "" if len(duplicate_ids) <= 5 else f", ... ({len(duplicate_ids)} ids total)"
        raise ValueError(
            f"{label} pose table contains duplicate sequence_id values: "
            f"{preview}{suffix}"
        )
    return _ORIGINAL_NORMALIZE_POSE_BY_SEQUENCE_TABLE(rows, label=label)


_IMPL._normalize_pose_by_sequence_table = _normalize_pose_by_sequence_table

globals().update(
    {
        name: getattr(_IMPL, name)
        for name in dir(_IMPL)
        if not (name.startswith("__") and name.endswith("__"))
    }
)
globals()["_duplicate_sequence_ids"] = _duplicate_sequence_ids
globals()["_normalize_pose_by_sequence_table"] = _normalize_pose_by_sequence_table

__doc__ = _IMPL.__doc__
__all__ = [
    name for name in dir(_IMPL) if not (name.startswith("__") and name.endswith("__"))
]
