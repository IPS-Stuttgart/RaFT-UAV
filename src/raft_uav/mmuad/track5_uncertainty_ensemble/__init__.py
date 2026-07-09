"""Package wrapper that normalizes Track 5 uncertainty-ensemble sequence IDs.

The legacy implementation lives in the sibling ``track5_uncertainty_ensemble.py``
file. This wrapper preserves public imports while patching the row-normalization
helpers so official Track 5 ``Sequence`` cells are canonicalized consistently with
the template resampler.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

from raft_uav.mmuad.submission import parse_official_sequence_cell

_IMPL_PATH = Path(__file__).resolve().parent.parent / "track5_uncertainty_ensemble.py"
_SPEC = importlib.util.spec_from_file_location(
    "raft_uav.mmuad._track5_uncertainty_ensemble_legacy",
    _IMPL_PATH,
)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"cannot load legacy Track 5 uncertainty ensemble from {_IMPL_PATH}")
_IMPL = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _IMPL
_SPEC.loader.exec_module(_IMPL)


class _PandasCsvProxy:
    """Pandas proxy whose CSV reader preserves opaque sequence-id text."""

    def __init__(self, pandas_module: Any) -> None:
        self._pandas = pandas_module

    def __getattr__(self, name: str) -> Any:
        return getattr(self._pandas, name)

    def read_csv(self, *args: Any, **kwargs: Any) -> pd.DataFrame:
        kwargs["dtype"] = str
        kwargs.setdefault("keep_default_na", False)
        rows = self._pandas.read_csv(*args, **kwargs)
        out = rows.copy()
        out.columns = [str(column).strip() for column in out.columns]
        for column in out.columns:
            if out[column].dtype == object or str(out[column].dtype).startswith("string"):
                out[column] = out[column].map(
                    lambda value: value.strip() if isinstance(value, str) else value
                )
        return out


_IMPL.pd = _PandasCsvProxy(pd)


def _first_present(rows: pd.DataFrame, names: tuple[str, ...]) -> Any | None:
    """Return an existing column whose normalized header matches one of ``names``."""

    by_normalized_name = {str(column).strip().lower(): column for column in rows.columns}
    missing = object()
    for name in names:
        if name in rows.columns:
            return name
        found = by_normalized_name.get(str(name).strip().lower(), missing)
        if found is not missing:
            return found
    return None


def _normalized_sequence_values(values: pd.Series) -> pd.Series:
    return values.map(_sequence_text_or_none)


def _sequence_text_or_none(value: Any) -> str | None:
    try:
        return parse_official_sequence_cell(value)
    except ValueError:
        return None


def _normalize_uncertainty_rows(estimates: pd.DataFrame, *, column: str) -> pd.DataFrame:
    rows = pd.DataFrame(estimates).copy()
    sequence_column = _first_present(rows, _IMPL.SEQUENCE_ALIASES)
    time_column = _first_present(rows, _IMPL.TIME_ALIASES)
    sigma_column = _first_present(rows, (column,))
    if sequence_column is None or time_column is None or sigma_column is None:
        return pd.DataFrame(columns=["sequence_id", "time_s", "sigma_m"])
    out = pd.DataFrame(
        {
            "sequence_id": _normalized_sequence_values(rows[sequence_column]),
            "time_s": pd.to_numeric(rows[time_column], errors="coerce"),
            "sigma_m": pd.to_numeric(rows[sigma_column], errors="coerce"),
        }
    )
    finite = out["sequence_id"].notna()
    finite &= np.isfinite(out[["time_s", "sigma_m"]].to_numpy(float)).all(axis=1)
    out = out.loc[finite & (out["sigma_m"] > 0.0)].copy()
    return out.drop_duplicates(["sequence_id", "time_s"], keep="last").reset_index(drop=True)


def _normalize_template_rows(template: pd.DataFrame) -> pd.DataFrame:
    rows = pd.DataFrame(template).copy()
    sequence_column = _first_present(rows, _IMPL.SEQUENCE_ALIASES)
    time_column = _first_present(rows, _IMPL.TIME_ALIASES)
    if sequence_column is None or time_column is None:
        raise ValueError("template must contain sequence and timestamp columns")
    out = pd.DataFrame(
        {
            "sequence_id": _normalized_sequence_values(rows[sequence_column]),
            "time_s": pd.to_numeric(rows[time_column], errors="coerce"),
        }
    )
    finite = out["sequence_id"].notna()
    finite &= np.isfinite(out["time_s"].to_numpy(float))
    return out.loc[finite].sort_values(["sequence_id", "time_s"]).reset_index(drop=True)


_IMPL._first_present = _first_present
_IMPL._normalized_sequence_values = _normalized_sequence_values
_IMPL._sequence_text_or_none = _sequence_text_or_none
_IMPL._normalize_uncertainty_rows = _normalize_uncertainty_rows
_IMPL._normalize_template_rows = _normalize_template_rows

for _name in dir(_IMPL):
    if not _name.startswith("__"):
        globals()[_name] = getattr(_IMPL, _name)

# Preserve access to the patched private helpers for tests and exploratory imports.
globals()["_first_present"] = _first_present
globals()["_normalized_sequence_values"] = _normalized_sequence_values
globals()["_sequence_text_or_none"] = _sequence_text_or_none
globals()["_normalize_uncertainty_rows"] = _normalize_uncertainty_rows
globals()["_normalize_template_rows"] = _normalize_template_rows
__doc__ = _IMPL.__doc__
__all__ = [_name for _name in dir(_IMPL) if not _name.startswith("__")]
