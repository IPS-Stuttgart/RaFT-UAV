"""Compatibility package for robust MMUAD trajectory-completion flag parsing.

The legacy implementation lives in the sibling ``trajectory_completion.py``
module. This wrapper preserves the public import path while replacing only the
estimate-row normalization that interprets ``selected_path_update`` values.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

_IMPL_PATH = Path(__file__).resolve().parent.parent / "trajectory_completion.py"
_SPEC = importlib.util.spec_from_file_location(
    "raft_uav.mmuad._trajectory_completion_legacy",
    _IMPL_PATH,
)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"cannot load legacy MMUAD trajectory completion from {_IMPL_PATH}")
_IMPL = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _IMPL
_SPEC.loader.exec_module(_IMPL)

_TRUE_TEXT = {"1", "true", "t", "yes", "y", "on"}
_FALSE_TEXT = {"0", "false", "f", "no", "n", "off", "", "none", "null", "nan", "<na>", "nat"}


def _parse_selected_path_update(value: Any) -> bool:
    """Parse native and serialized path-selection flags without string truthiness."""

    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if value is None:
        return False
    try:
        if bool(pd.isna(value)):
            return False
    except (TypeError, ValueError):
        pass
    if isinstance(value, str):
        text = value.strip().lower()
        if text in _TRUE_TEXT:
            return True
        if text in _FALSE_TEXT:
            return False
        raise ValueError(f"cannot parse selected_path_update value: {value!r}")
    if isinstance(value, (int, float, np.integer, np.floating)):
        numeric = float(value)
        return bool(numeric) if np.isfinite(numeric) else False
    raise ValueError(f"cannot parse selected_path_update value: {value!r}")


def _estimate_rows(estimates: pd.DataFrame) -> pd.DataFrame:
    rows = pd.DataFrame(estimates).copy()
    if rows.empty:
        return rows
    if "sequence_id" not in rows.columns:
        rows["sequence_id"] = "default"
    for column in ("time_s", "state_x_m", "state_y_m", "state_z_m"):
        if column not in rows.columns:
            raise ValueError(f"estimate rows missing required column {column!r}")
        rows[column] = pd.to_numeric(rows[column], errors="coerce")
    finite = np.isfinite(
        rows[["time_s", "state_x_m", "state_y_m", "state_z_m"]].to_numpy(float)
    ).all(axis=1)
    rows = rows.loc[finite].copy()
    rows["sequence_id"] = rows["sequence_id"].astype(str)
    if "selected_path_update" in rows.columns:
        rows["selected_path_update"] = rows["selected_path_update"].map(
            _parse_selected_path_update
        )
    else:
        rows["selected_path_update"] = True
    return rows.sort_values(_IMPL._sort_columns(rows)).reset_index(drop=True)


_IMPL._estimate_rows = _estimate_rows

for _name in dir(_IMPL):
    if not _name.startswith("__"):
        globals()[_name] = getattr(_IMPL, _name)

globals()["_parse_selected_path_update"] = _parse_selected_path_update
globals()["_estimate_rows"] = _estimate_rows
__doc__ = _IMPL.__doc__
__all__ = [_name for _name in dir(_IMPL) if not _name.startswith("__")]
