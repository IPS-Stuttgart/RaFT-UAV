"""Index-safe compatibility layer for reservoir bottleneck diagnostics.

The maintained implementation lives in the sibling
``candidate_reservoir_bottleneck.py`` module.  This package preserves the
public import path while keeping annotations and summary row selection
positional, independent of caller-provided DataFrame index labels.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

_IMPL_PATH = Path(__file__).resolve().parent.parent / "candidate_reservoir_bottleneck.py"
_SPEC = importlib.util.spec_from_file_location(
    "raft_uav.mmuad._candidate_reservoir_bottleneck_impl",
    _IMPL_PATH,
)
if _SPEC is None or _SPEC.loader is None:  # pragma: no cover - import guard
    raise ImportError(f"could not load reservoir bottleneck implementation from {_IMPL_PATH}")
_IMPL = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _IMPL
_SPEC.loader.exec_module(_IMPL)


def annotate_gap_table(
    gap_rows: pd.DataFrame,
    *,
    config: Any = None,
) -> pd.DataFrame:
    """Append annotations by row position rather than index-label alignment."""

    rows = pd.DataFrame(gap_rows).copy()
    if rows.empty:
        return rows.assign(
            primary_bottleneck=pd.Series(dtype=str),
            recommended_action=pd.Series(dtype=str),
        )
    annotations = pd.DataFrame.from_records(
        [
            _IMPL.classify_gap_row(record, config=config)
            for record in rows.to_dict(orient="records")
        ]
    )
    add_columns = [
        column
        for column in annotations.columns
        if column not in rows.columns or column in _IMPL._OVERWRITE_COLUMNS
    ]
    for column in add_columns:
        rows[column] = annotations[column].to_numpy()
    return rows


def _max_record(rows: pd.DataFrame, column: str) -> dict[str, Any]:
    """Return the maximum-valued row without coercing its index label."""

    if rows.empty or column not in rows.columns:
        return {}
    values = pd.to_numeric(rows[column], errors="coerce")
    if values.dropna().empty:
        return {}
    position = int(np.nanargmax(values.to_numpy(dtype=float)))
    return _IMPL._jsonable(rows.iloc[position].to_dict())


_IMPL.annotate_gap_table = annotate_gap_table
_IMPL._max_record = _max_record

for _name, _value in vars(_IMPL).items():
    if _name not in {
        "__name__",
        "__package__",
        "__loader__",
        "__spec__",
        "__file__",
        "__cached__",
    }:
        globals()[_name] = _value

globals()["annotate_gap_table"] = annotate_gap_table
globals()["_max_record"] = _max_record

__doc__ = _IMPL.__doc__
__all__ = [name for name in vars(_IMPL) if not name.startswith("_")]
