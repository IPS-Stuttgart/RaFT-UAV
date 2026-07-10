"""Compatibility wrapper for the pair-state forward-backward implementation.

The maintained implementation lives in the sibling
``candidate_pair_forward_backward.py`` module.  This package keeps the public
import path while preserving opaque sequence identifiers in optional mixture
initialization files, applying score fallbacks row by row, and canonicalizing
numeric tracker identifiers before track-continuation scoring.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

from raft_uav.mmuad.candidate_identity import canonical_track_ids
from raft_uav.mmuad.estimate_csv import read_estimate_csv

_IMPL_PATH = Path(__file__).resolve().parent.parent / "candidate_pair_forward_backward.py"
_SPEC = importlib.util.spec_from_file_location(
    "raft_uav.mmuad._candidate_pair_forward_backward_legacy",
    _IMPL_PATH,
)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"cannot load pair-state implementation from {_IMPL_PATH}")
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


def _candidate_score_with_rowwise_fallback(rows: pd.DataFrame, config: Any) -> pd.Series:
    """Resolve configured score columns independently for every candidate row."""

    resolved = pd.Series(np.nan, index=rows.index, dtype=float)
    for column in (config.score_column, *config.fallback_score_columns):
        if column not in rows.columns:
            continue
        values = pd.to_numeric(rows[column], errors="coerce")
        values = values.where(np.isfinite(values))
        resolved = resolved.where(resolved.notna(), values)

    finite = resolved.dropna()
    fill_value = float(finite.min()) if not finite.empty else 1.0
    return resolved.fillna(fill_value).astype(float)


_ORIGINAL_TRANSITION_LOG_LIKELIHOOD = _IMPL._transition_log_likelihood


def _transition_log_likelihood_with_canonical_track_ids(
    previous: dict[str, Any],
    current: dict[str, Any],
    config: Any,
) -> Any:
    """Apply the existing transition model after canonicalizing track identity."""

    previous_rows = dict(previous)
    current_rows = dict(current)
    previous_rows["track_ids"] = canonical_track_ids(previous.get("track_ids", ()))
    current_rows["track_ids"] = canonical_track_ids(current.get("track_ids", ()))
    return _ORIGINAL_TRANSITION_LOG_LIKELIHOOD(previous_rows, current_rows, config)


_IMPL.pd = _PandasCsvProxy(pd)
_IMPL._candidate_score = _candidate_score_with_rowwise_fallback
_IMPL._transition_log_likelihood = _transition_log_likelihood_with_canonical_track_ids

globals().update(
    {
        name: getattr(_IMPL, name)
        for name in dir(_IMPL)
        if not (name.startswith("__") and name.endswith("__"))
    }
)

__doc__ = _IMPL.__doc__
__all__ = [name for name in dir(_IMPL) if not (name.startswith("__") and name.endswith("__"))]
