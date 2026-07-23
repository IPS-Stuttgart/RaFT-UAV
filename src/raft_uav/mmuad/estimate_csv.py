"""CSV readers for MMUAD estimate trajectory tables."""

from __future__ import annotations

from collections.abc import Iterable
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
_ORIGINAL_READ_CSV_ATTRIBUTE = "_raft_uav_original_pandas_read_csv"
_ORIGINAL_PANDAS_READ_CSV = getattr(
    pd.read_csv,
    _ORIGINAL_READ_CSV_ATTRIBUTE,
    pd.read_csv,
)
_PHYSICAL_HEADER_READ_CSV_OPTIONS = (
    "sep",
    "delimiter",
    "quotechar",
    "quoting",
    "doublequote",
    "escapechar",
    "comment",
    "encoding",
    "encoding_errors",
    "dialect",
    "skipinitialspace",
    "skiprows",
)


def _normalized_estimate_csv_columns(columns: Iterable[object]) -> list[str]:
    """Return trimmed estimate headers after rejecting ambiguous collisions."""

    normalized = [str(column).strip() for column in columns]
    groups: dict[str, list[str]] = {}
    for column in normalized:
        groups.setdefault(column.casefold(), []).append(column)
    collisions = sorted(
        {
            column
            for group in groups.values()
            if len(group) > 1
            for column in group
        },
        key=str.casefold,
    )
    if collisions:
        raise ValueError(
            "estimate CSV headers are ambiguous after trimming whitespace and "
            f"ignoring case: {collisions}"
        )
    return normalized


def _validate_physical_estimate_csv_header(
    source: Any,
    *,
    read_csv_kwargs: dict[str, Any] | None = None,
) -> None:
    """Reject duplicate path-based CSV headers before pandas mangles their names."""

    if not isinstance(source, (str, Path)):
        return
    options = dict(read_csv_kwargs or {})
    if options.get("header", "infer") not in {"infer", 0} or options.get("names") is not None:
        return
    header_options = {
        key: options[key]
        for key in _PHYSICAL_HEADER_READ_CSV_OPTIONS
        if key in options
    }
    physical_header = _ORIGINAL_PANDAS_READ_CSV(
        source,
        header=None,
        nrows=1,
        dtype=str,
        keep_default_na=False,
        **header_options,
    )
    if physical_header.empty:
        return
    _normalized_estimate_csv_columns(physical_header.iloc[0].tolist())


def read_estimate_csv(path: Path) -> pd.DataFrame:
    """Read estimate CSVs without coercing opaque identifier columns.

    Track 5 sequence identifiers can be numeric-looking strings such as ``001``.
    Read the table as text first so pandas cannot coerce those values before the
    normal schema-specific numeric conversion in downstream loaders.
    """

    _validate_physical_estimate_csv_header(path)
    rows = _ORIGINAL_PANDAS_READ_CSV(path, dtype=str, keep_default_na=False)
    return _strip_column_names(rows)


def _strip_column_names(rows: pd.DataFrame) -> pd.DataFrame:
    out = rows.copy()
    out.columns = _normalized_estimate_csv_columns(out.columns)
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
        source = args[0] if args else kwargs.get("filepath_or_buffer")
        _validate_physical_estimate_csv_header(source, read_csv_kwargs=kwargs)
        kwargs = dict(kwargs)
        kwargs["dtype"] = str
        kwargs.setdefault("keep_default_na", False)
        return _strip_column_names(_ORIGINAL_PANDAS_READ_CSV(*args, **kwargs))
    return _ORIGINAL_PANDAS_READ_CSV(*args, **kwargs)


setattr(
    _read_csv_with_track5_estimate_grid_guard,
    _ORIGINAL_READ_CSV_ATTRIBUTE,
    _ORIGINAL_PANDAS_READ_CSV,
)


def _install_track5_estimate_grid_guard() -> None:
    if pd.read_csv is not _read_csv_with_track5_estimate_grid_guard:
        pd.read_csv = _read_csv_with_track5_estimate_grid_guard


_install_track5_estimate_grid_guard()
