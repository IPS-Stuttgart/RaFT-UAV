"""Compatibility package for the basic MMUAD tracker."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pandas as pd

_LEGACY_PATH = Path(__file__).resolve().parent.parent / "tracker.py"
_LEGACY_NAME = f"{__name__.rsplit('.', 1)[0]}._tracker_legacy"
_SPEC = importlib.util.spec_from_file_location(_LEGACY_NAME, _LEGACY_PATH)
if _SPEC is None or _SPEC.loader is None:  # pragma: no cover - import machinery failure
    raise ImportError(f"cannot load tracker implementation from {_LEGACY_PATH}")
_LEGACY = importlib.util.module_from_spec(_SPEC)
sys.modules[_LEGACY_NAME] = _LEGACY
_SPEC.loader.exec_module(_LEGACY)

for _name in dir(_LEGACY):
    if not _name.startswith("__"):
        globals()[_name] = getattr(_LEGACY, _name)

_ORIGINAL_CANDIDATE_ROWS_WITH_OPTIONAL_DEFAULTS = (
    _LEGACY._candidate_rows_with_optional_defaults
)
_TRACKER_NUMERIC_COLUMNS = (
    "time_s",
    "x_m",
    "y_m",
    "z_m",
    "std_xy_m",
    "std_z_m",
    "confidence",
)


def _candidate_rows_with_optional_defaults(rows: pd.DataFrame) -> pd.DataFrame:
    """Fill optional columns and normalize values used numerically by the tracker."""

    out = _ORIGINAL_CANDIDATE_ROWS_WITH_OPTIONAL_DEFAULTS(rows)
    for column in _TRACKER_NUMERIC_COLUMNS:
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce")
    return out


_LEGACY._candidate_rows_with_optional_defaults = _candidate_rows_with_optional_defaults
