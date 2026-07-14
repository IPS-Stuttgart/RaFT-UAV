"""CSV readers for MMUAD estimate trajectory tables."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pandas as pd

_GRID_MODULE = "raft_uav.mmuad.track5_estimate_ensemble_grid"
_UNCERTAINTY_ENSEMBLE_MODULE = "raft_uav.mmuad.track5_uncertainty_ensemble"
_UNCERTAINTY_ENSEMBLE_READER = "_read_estimate_csv_preserving_sequence_id"
_CANDIDATE_RESERVOIR_MODULE = "raft_uav.mmuad"
_CANDIDATE_RESERVOIR_MAIN_QUALNAME = (
    "_install_candidate_reservoir_topk_guard.<locals>._main"
)
_ORIGINAL_PANDAS_READ_CSV = pd.read_csv


def read_estimate_csv(path: Path) -> pd.DataFrame:
    """Read estimate CSVs without coercing opaque identifier columns.

    Track 5 sequence identifiers can be numeric-looking strings such as ``001``.
    Read the table as text first so pandas cannot coerce those values before the
    normal schema-specific numeric conversion in downstream loaders.
    """

    rows = _ORIGINAL_PANDAS_READ_CSV(path, dtype=str, keep_default_na=False)
    return _strip_column_names(rows)


def _strip_column_names(rows: pd.DataFrame) -> pd.DataFrame:
    out = rows.copy()
    out.columns = [str(column).strip() for column in out.columns]
    return out


def _called_from_track5_estimate_grid() -> bool:
    frame = sys._getframe(2)
    while frame is not None:
        module = frame.f_globals.get("__name__")
        if module == _GRID_MODULE:
            return True
        if (
            module == _UNCERTAINTY_ENSEMBLE_MODULE
            and frame.f_code.co_name == _UNCERTAINTY_ENSEMBLE_READER
        ):
            return True
        frame = frame.f_back
    return False


def _called_from_candidate_reservoir_cli() -> bool:
    frame = sys._getframe(2)
    while frame is not None:
        if (
            frame.f_globals.get("__name__") == _CANDIDATE_RESERVOIR_MODULE
            and frame.f_code.co_qualname == _CANDIDATE_RESERVOIR_MAIN_QUALNAME
        ):
            return True
        frame = frame.f_back
    return False


def _read_csv_with_track5_estimate_grid_guard(*args: Any, **kwargs: Any) -> pd.DataFrame:
    if _called_from_track5_estimate_grid() or _called_from_candidate_reservoir_cli():
        kwargs = dict(kwargs)
        kwargs["dtype"] = str
        kwargs.setdefault("keep_default_na", False)
        return _strip_column_names(_ORIGINAL_PANDAS_READ_CSV(*args, **kwargs))
    return _ORIGINAL_PANDAS_READ_CSV(*args, **kwargs)


def _install_track5_estimate_grid_guard() -> None:
    if pd.read_csv is not _read_csv_with_track5_estimate_grid_guard:
        pd.read_csv = _read_csv_with_track5_estimate_grid_guard


_install_track5_estimate_grid_guard()
