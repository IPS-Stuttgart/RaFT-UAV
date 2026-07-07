"""CSV readers for MMUAD estimate trajectory tables."""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any

import pandas as pd

_GRID_MODULE = "raft_uav.mmuad.track5_estimate_ensemble_grid"
_ORIGINAL_PANDAS_READ_CSV = pd.read_csv


def read_estimate_csv(path: Path) -> pd.DataFrame:
    """Read estimate CSVs without coercing opaque identifier columns.

    Track 5 sequence identifiers can be numeric-looking strings such as ``001``.
    Read the table as text first so pandas cannot coerce those values before the
    normal schema-specific numeric conversion in downstream loaders.
    """

    rows = _ORIGINAL_PANDAS_READ_CSV(path, dtype=str, keep_default_na=False)
    out = rows.copy()
    out.columns = [str(column).strip() for column in out.columns]
    return out


def _called_from_track5_estimate_grid() -> bool:
    frame = sys._getframe(2)
    while frame is not None:
        if frame.f_globals.get("__name__") == _GRID_MODULE:
            return True
        frame = frame.f_back
    return False


def _read_csv_with_track5_estimate_grid_guard(*args: Any, **kwargs: Any) -> pd.DataFrame:
    if _called_from_track5_estimate_grid() and "dtype" not in kwargs:
        kwargs = dict(kwargs)
        kwargs["dtype"] = str
        kwargs.setdefault("keep_default_na", False)
    return _ORIGINAL_PANDAS_READ_CSV(*args, **kwargs)


def _install_track5_estimate_grid_guard() -> None:
    if pd.read_csv is not _read_csv_with_track5_estimate_grid_guard:
        pd.read_csv = _read_csv_with_track5_estimate_grid_guard


_install_track5_estimate_grid_guard()
