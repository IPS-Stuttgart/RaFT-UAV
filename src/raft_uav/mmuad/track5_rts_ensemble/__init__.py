"""Compatibility wrapper hardening Track 5 RTS ensemble inputs.

The maintained implementation lives in the sibling ``track5_rts_ensemble.py``
module. This package keeps the public import path while preserving opaque
sequence identifiers in estimate CSVs, canonicalizing template identifiers, and
allowing zero-weight estimate inputs to act as explicit disable switches.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from typing import Any

import pandas as pd

from raft_uav.mmuad.estimate_csv import read_estimate_csv
from raft_uav.mmuad.submission import parse_official_sequence_cell

_IMPL_PATH = Path(__file__).resolve().parent.parent / "track5_rts_ensemble.py"
_SPEC = importlib.util.spec_from_file_location(
    "raft_uav.mmuad._track5_rts_ensemble_legacy",
    _IMPL_PATH,
)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"cannot load Track 5 RTS ensemble implementation from {_IMPL_PATH}")
_IMPL = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _IMPL
_SPEC.loader.exec_module(_IMPL)

_ORIGINAL_POSITIVE_FINITE = _IMPL._positive_finite


class _PandasCsvProxy:
    """Delegate pandas operations while guarding plain estimate CSV reads."""

    def __init__(self, module: Any) -> None:
        self._module = module

    def __getattr__(self, name: str) -> Any:
        return getattr(self._module, name)

    def read_csv(self, path: Any, *args: Any, **kwargs: Any) -> pd.DataFrame:
        if args or kwargs:
            rows = self._module.read_csv(path, *args, **kwargs)
            out = rows.copy()
            out.columns = [str(column).strip() for column in out.columns]
            return out
        return read_estimate_csv(Path(path))


def _first_present(rows: pd.DataFrame, names: tuple[str, ...]) -> Any | None:
    """Return a column whose stripped, case-folded name matches an alias."""

    normalized = {str(column).strip().casefold(): column for column in rows.columns}
    for name in names:
        if name in rows.columns:
            return name
        found = normalized.get(str(name).strip().casefold())
        if found is not None:
            return found
    return None


def _sequence_text_or_none(value: Any) -> str | None:
    try:
        return parse_official_sequence_cell(value)
    except ValueError:
        return None


def _normalize_template_rows(template: pd.DataFrame) -> pd.DataFrame:
    rows = pd.DataFrame(template).copy()
    if rows.empty:
        return pd.DataFrame(columns=["sequence_id", "time_s"])
    sequence_column = _first_present(
        rows,
        ("sequence_id", "Sequence", "sequence", "seq"),
    )
    time_column = _first_present(
        rows,
        ("time_s", "Timestamp", "timestamp", "timestamp_s", "time"),
    )
    if sequence_column is None or time_column is None:
        raise ValueError("template must contain sequence and timestamp columns")
    out = pd.DataFrame(
        {
            "sequence_id": rows[sequence_column].map(_sequence_text_or_none),
            "time_s": pd.to_numeric(rows[time_column], errors="coerce"),
        }
    )
    finite = out["sequence_id"].notna() & _IMPL.np.isfinite(out["time_s"].to_numpy(float))
    return out.loc[finite].sort_values(["sequence_id", "time_s"]).reset_index(drop=True)


def _positive_finite_allowing_disabled_inputs(value: float, name: str) -> float:
    """Allow zero only for estimate weights used as explicit disable switches."""

    if str(name).startswith("weight["):
        return _IMPL._nonnegative_finite(value, name)
    return _ORIGINAL_POSITIVE_FINITE(value, name)


_IMPL.pd = _PandasCsvProxy(pd)
_IMPL._first_present = _first_present
_IMPL._normalize_template_rows = _normalize_template_rows
_IMPL._positive_finite = _positive_finite_allowing_disabled_inputs

globals().update(
    {
        name: getattr(_IMPL, name)
        for name in dir(_IMPL)
        if not (name.startswith("__") and name.endswith("__"))
    }
)

# Keep patched private helpers importable for focused regressions.
globals()["_first_present"] = _first_present
globals()["_sequence_text_or_none"] = _sequence_text_or_none
globals()["_normalize_template_rows"] = _normalize_template_rows
globals()["_positive_finite_allowing_disabled_inputs"] = (
    _positive_finite_allowing_disabled_inputs
)

__doc__ = _IMPL.__doc__
__all__ = [name for name in dir(_IMPL) if not (name.startswith("__") and name.endswith("__"))]
