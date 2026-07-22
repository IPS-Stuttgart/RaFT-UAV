"""Compatibility fixes for robust paper-table diagnostics.

The maintained implementation lives in the sibling ``paper_table.py`` module.
This package preserves the public import path while excluding malformed radar
anchors before interpolation and reporting invalid reference counts as failed
checks instead of raising conversion errors.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pandas as pd

_IMPL_PATH = Path(__file__).resolve().parent.parent / "paper_table.py"
_SPEC = importlib.util.spec_from_file_location(
    "raft_uav.diagnostics._paper_table_legacy",
    _IMPL_PATH,
)
if _SPEC is None or _SPEC.loader is None:  # pragma: no cover
    raise ImportError(f"cannot load paper-table implementation from {_IMPL_PATH}")
_IMPL = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _IMPL
_SPEC.loader.exec_module(_IMPL)

_ORIGINAL_INTERPOLATE_SELECTED_RADAR = (
    _IMPL._interpolate_selected_radar_to_frame_times
)
_POSITION_COLUMNS = ("east_m", "north_m", "up_m")


def _finite_interpolation_anchors(selected: pd.DataFrame) -> pd.DataFrame:
    """Return anchors with finite numeric timestamps and complete 3D positions."""

    required = ("time_s", *_POSITION_COLUMNS)
    if any(column not in selected.columns for column in required):
        return selected

    anchors = selected.copy()
    for column in required:
        anchors[column] = pd.to_numeric(anchors[column], errors="coerce")
    finite = np.isfinite(
        anchors.loc[:, list(required)].to_numpy(dtype=float)
    ).all(axis=1)
    return anchors.loc[finite].copy()


def _interpolate_selected_radar_to_frame_times(
    radar: pd.DataFrame,
    selected: pd.DataFrame,
    *,
    association_mode: str,
    max_gap_s: float | None = None,
    max_speed_mps: float | None = None,
) -> pd.DataFrame:
    """Interpolate from usable anchors without letting one bad row erase output."""

    return _ORIGINAL_INTERPOLATE_SELECTED_RADAR(
        radar,
        _finite_interpolation_anchors(selected),
        association_mode=association_mode,
        max_gap_s=max_gap_s,
        max_speed_mps=max_speed_mps,
    )


def paper_reference_count_check(
    table: pd.DataFrame,
    *,
    tolerance: int = 0,
) -> dict[str, object]:
    """Compare reference counts while treating invalid values as failed checks."""

    tolerance_value = int(tolerance)
    rows: list[dict[str, object]] = []
    passed = True
    for method, column, expected in _IMPL.PAPER_REFERENCE_COUNT_CHECKS:
        match = (
            table.loc[table.get("method") == method]
            if "method" in table
            else pd.DataFrame()
        )
        if match.empty or column not in match.columns:
            rows.append(
                {
                    "method": method,
                    "column": column,
                    "expected": int(expected),
                    "actual": None,
                    "delta": None,
                    "passed": False,
                }
            )
            passed = False
            continue

        try:
            numeric = float(pd.to_numeric(match.iloc[0][column], errors="coerce"))
        except (TypeError, ValueError):
            numeric = float("nan")
        if not np.isfinite(numeric):
            rows.append(
                {
                    "method": method,
                    "column": column,
                    "expected": int(expected),
                    "actual": None,
                    "delta": None,
                    "passed": False,
                }
            )
            passed = False
            continue

        actual = int(numeric)
        delta = actual - int(expected)
        ok = abs(delta) <= tolerance_value
        rows.append(
            {
                "method": method,
                "column": column,
                "expected": int(expected),
                "actual": actual,
                "delta": int(delta),
                "passed": bool(ok),
            }
        )
        passed &= bool(ok)

    message = (
        "paper reference counts matched"
        if passed
        else f"paper reference count mismatch: {rows}"
    )
    return {
        "passed": bool(passed),
        "tolerance": tolerance_value,
        "checks": rows,
        "message": message,
    }


_IMPL._finite_interpolation_anchors = _finite_interpolation_anchors
_IMPL._interpolate_selected_radar_to_frame_times = (
    _interpolate_selected_radar_to_frame_times
)
_IMPL.paper_reference_count_check = paper_reference_count_check

globals().update(
    {
        name: getattr(_IMPL, name)
        for name in dir(_IMPL)
        if not (name.startswith("__") and name.endswith("__"))
    }
)
globals()["_finite_interpolation_anchors"] = _finite_interpolation_anchors
globals()["_interpolate_selected_radar_to_frame_times"] = (
    _interpolate_selected_radar_to_frame_times
)
globals()["paper_reference_count_check"] = paper_reference_count_check

__doc__ = _IMPL.__doc__
__all__ = [
    name for name in dir(_IMPL) if not (name.startswith("__") and name.endswith("__"))
]
