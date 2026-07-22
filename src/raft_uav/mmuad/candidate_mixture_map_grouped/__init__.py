"""Compatibility wrapper preserving opaque IDs in grouped-mixture CSV inputs.

The maintained implementation lives in the sibling
``candidate_mixture_map_grouped.py`` module. This package keeps the public
import path while making the grouped-mixture CLI read initial trajectories
without coercing numeric-looking sequence identifiers such as ``001``.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from typing import Any

import pandas as pd

from raft_uav.mmuad.estimate_csv import read_estimate_csv

_IMPL_PATH = Path(__file__).resolve().parent.parent / "candidate_mixture_map_grouped.py"
_SPEC = importlib.util.spec_from_file_location(
    "raft_uav.mmuad._candidate_mixture_map_grouped_legacy",
    _IMPL_PATH,
)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"cannot load grouped candidate-mixture implementation from {_IMPL_PATH}")
_IMPL = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _IMPL
_SPEC.loader.exec_module(_IMPL)


class _PandasCsvProxy:
    """Delegate pandas operations while preserving identifiers in plain CSV reads."""

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


def _group_values(
    rows: pd.DataFrame,
    *,
    resolved_group_column: str | None,
    missing_group_policy: str,
) -> pd.Series:
    """Return group labels without colliding synthetic and explicit values."""

    input_rows = rows["mixture_group_input_row"].astype(int)
    if resolved_group_column is None:
        return pd.Series(
            [f"row:{int(index)}" for index in input_rows],
            index=rows.index,
            dtype=str,
        )

    values = rows[resolved_group_column]
    missing = values.isna() | values.astype(str).str.strip().eq("")
    if missing.any() and missing_group_policy == "error":
        raise ValueError(
            f"hypothesis group column {resolved_group_column!r} contains missing values"
        )

    result = values.astype(str).copy()
    occupied = set(result.loc[~missing].astype(str))
    for row_index in rows.index[missing]:
        candidate = f"row:{int(input_rows.loc[row_index])}"
        while candidate in occupied:
            candidate += "#"
        occupied.add(candidate)
        result.loc[row_index] = candidate
    return result.astype(str)


_IMPL.pd = _PandasCsvProxy(pd)
_IMPL._group_values = _group_values

globals().update(
    {
        name: getattr(_IMPL, name)
        for name in dir(_IMPL)
        if not (name.startswith("__") and name.endswith("__"))
    }
)
globals()["_group_values"] = _group_values

__doc__ = _IMPL.__doc__
__all__ = [name for name in dir(_IMPL) if not (name.startswith("__") and name.endswith("__"))]
