"""Compatibility wrapper for the first-order forward-backward implementation.

The maintained implementation lives in the sibling ``candidate_forward_backward.py``
module. This package keeps the public import path while hardening candidate identity,
row-wise score fallback, tied-rank handling, and configuration validation without
duplicating the implementation.
"""

from __future__ import annotations

import importlib.util
from numbers import Real
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

from raft_uav.mmuad.candidate_identity import canonical_track_ids

_IMPL_PATH = Path(__file__).resolve().parent.parent / "candidate_forward_backward.py"
_SPEC = importlib.util.spec_from_file_location(
    "raft_uav.mmuad._candidate_forward_backward_legacy",
    _IMPL_PATH,
)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"cannot load forward-backward implementation from {_IMPL_PATH}")
_IMPL = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _IMPL
_SPEC.loader.exec_module(_IMPL)

_ORIGINAL_TRANSITION_LOG_LIKELIHOOD = _IMPL._transition_log_likelihood
_ORIGINAL_VALIDATE_CONFIG = _IMPL._validate_config
_NUMERIC_CONFIG_FIELDS = (
    "default_sigma_m",
    "sigma_min_m",
    "sigma_max_m",
    "score_weight",
    "sigma_log_weight",
    "transition_distance_std_m",
    "transition_speed_std_mps",
    "max_speed_mps",
    "speed_gate_penalty",
    "source_switch_penalty",
    "branch_switch_penalty",
    "track_continuation_bonus",
    "time_gap_penalty",
)


def _finite_config_scalar(value: Any, *, name: str) -> float:
    """Return a finite real scalar for a forward-backward tuning field."""

    message = f"{name} must be a finite scalar"
    if isinstance(value, np.ndarray):
        if value.ndim != 0:
            raise ValueError(message)
        value = value.item()
    elif isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(message)
    number = float(value)
    if not np.isfinite(number):
        raise ValueError(message)
    return number


def _validate_config_with_finite_controls(config: Any) -> None:
    """Reject non-finite controls before transition or emission construction."""

    for field in _NUMERIC_CONFIG_FIELDS:
        _finite_config_scalar(getattr(config, field), name=field)
    _ORIGINAL_VALIDATE_CONFIG(config)


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


def _candidate_score_with_row_fallback(rows: pd.DataFrame, config: Any) -> pd.Series:
    """Resolve primary and fallback scores independently for every candidate row."""

    score = pd.Series(np.nan, index=rows.index, dtype=float)
    for column in (config.score_column, *config.fallback_score_columns):
        if column not in rows.columns:
            continue
        values = pd.to_numeric(rows[column], errors="coerce").astype(float)
        values = values.where(np.isfinite(values), np.nan)
        score = score.where(score.notna(), values)

    finite = np.isfinite(score.to_numpy(float))
    if not finite.any():
        return pd.Series(1.0, index=rows.index, dtype=float)
    return score.fillna(float(score.loc[finite].min())).astype(float)


def _descending_average_ranks(values: np.ndarray) -> np.ndarray:
    """Return permutation-invariant descending ranks with average ranks for ties."""

    array = np.asarray(values, dtype=float)
    if array.size == 0:
        return np.asarray([], dtype=float)
    finite = np.isfinite(array)
    ranks = np.full(array.shape, np.nan, dtype=float)
    if finite.any():
        ranks[finite] = (
            pd.Series(array[finite])
            .rank(method="average", ascending=False)
            .to_numpy(dtype=float)
        )
        ranks[~finite] = float(np.sum(finite) + 1)
    return ranks


_IMPL._validate_config = _validate_config_with_finite_controls
_IMPL._transition_log_likelihood = _transition_log_likelihood_with_canonical_track_ids
_IMPL._candidate_score = _candidate_score_with_row_fallback
_IMPL._descending_ranks = _descending_average_ranks

globals().update(
    {
        name: getattr(_IMPL, name)
        for name in dir(_IMPL)
        if not (name.startswith("__") and name.endswith("__"))
    }
)
globals()["_finite_config_scalar"] = _finite_config_scalar
globals()["_validate_config"] = _validate_config_with_finite_controls

__doc__ = _IMPL.__doc__
__all__ = [name for name in dir(_IMPL) if not (name.startswith("__") and name.endswith("__"))]
