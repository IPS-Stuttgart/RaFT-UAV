"""Compatibility validation for empirical covariance sequence alignment.

The maintained implementation lives in the sibling ``empirical_covariance.py``
module. This package preserves the public import path while rejecting one-sided
flight/sequence metadata and scoping timestamp alignment by every available
flight boundary before pooled calibration can mix unrelated flights.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pandas as pd

_IMPL_PATH = Path(__file__).resolve().parent.parent / "empirical_covariance.py"
_SPEC = importlib.util.spec_from_file_location(
    "raft_uav.calibration._empirical_covariance_legacy",
    _IMPL_PATH,
)
if _SPEC is None or _SPEC.loader is None:  # pragma: no cover - import guard
    raise ImportError(
        f"cannot load empirical covariance implementation from {_IMPL_PATH}"
    )
_IMPL = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _IMPL
_SPEC.loader.exec_module(_IMPL)

_ORIGINAL_ALIGNED_RESIDUALS = _IMPL.aligned_residuals
_SCOPE_COLUMNS = ("sequence_id", "flight_id")


def _present_scope_columns(frame: pd.DataFrame) -> tuple[str, ...]:
    """Return flight-boundary aliases present on ``frame`` in stable order."""

    return tuple(column for column in _SCOPE_COLUMNS if column in frame.columns)


def _normalized_scope_keys(
    frame: pd.DataFrame,
    scope_columns: tuple[str, ...],
) -> pd.DataFrame:
    """Normalize every scope alias with the legacy sequence-key semantics."""

    keys = pd.DataFrame(index=frame.index)
    for column in scope_columns:
        keys[column] = _IMPL._sequence_keys(frame[column])
    return keys


def _aligned_residuals_by_scope(
    frame: pd.DataFrame,
    truth: pd.DataFrame,
    *,
    source: str,
    max_time_delta_s: float,
    scope_columns: tuple[str, ...],
) -> np.ndarray:
    """Align residuals independently for each complete joint scope key."""

    if source not in _IMPL._SOURCE_COORDS:
        raise ValueError(f"unknown source {source!r}")
    required = ("time_s", *_IMPL._SOURCE_COORDS[source])
    if frame.empty or not all(column in frame.columns for column in required):
        return np.empty((0, len(_IMPL._SOURCE_COORDS[source])), dtype=float)
    if truth.empty or not all(column in truth.columns for column in required):
        return np.empty((0, len(_IMPL._SOURCE_COORDS[source])), dtype=float)

    frame_keys = _normalized_scope_keys(frame, scope_columns)
    truth_keys = _normalized_scope_keys(truth, scope_columns)
    complete_frame = frame_keys.notna().all(axis=1)
    complete_truth = truth_keys.notna().all(axis=1)
    if not bool(complete_frame.any()) or not bool(complete_truth.any()):
        return np.empty((0, len(_IMPL._SOURCE_COORDS[source])), dtype=float)

    residual_blocks: list[np.ndarray] = []
    scope_values = frame_keys.loc[complete_frame, list(scope_columns)].drop_duplicates()
    for _, scope_value in scope_values.iterrows():
        frame_mask = complete_frame.copy()
        truth_mask = complete_truth.copy()
        for column in scope_columns:
            frame_mask &= frame_keys[column].eq(scope_value[column]).fillna(False)
            truth_mask &= truth_keys[column].eq(scope_value[column]).fillna(False)

        scoped_truth = truth.loc[truth_mask]
        if scoped_truth.empty:
            continue
        scoped_frame = frame.loc[frame_mask]
        block = _ORIGINAL_ALIGNED_RESIDUALS(
            scoped_frame.drop(columns=list(_SCOPE_COLUMNS), errors="ignore"),
            scoped_truth.drop(columns=list(_SCOPE_COLUMNS), errors="ignore"),
            source=source,
            max_time_delta_s=max_time_delta_s,
        )
        if block.size:
            residual_blocks.append(block)

    if residual_blocks:
        return np.vstack(residual_blocks)
    return np.empty((0, len(_IMPL._SOURCE_COORDS[source])), dtype=float)


def aligned_residuals(
    frame: pd.DataFrame,
    truth: pd.DataFrame,
    *,
    source: str,
    max_time_delta_s: float,
):
    """Align residuals only within structurally consistent flight scopes."""

    for column in _SCOPE_COLUMNS:
        frame_has_scope = column in frame.columns
        truth_has_scope = column in truth.columns
        if frame_has_scope != truth_has_scope:
            raise ValueError(
                f"frame and truth must either both contain {column} or both omit it"
            )

    scope_columns = _present_scope_columns(frame)
    if not scope_columns:
        return _ORIGINAL_ALIGNED_RESIDUALS(
            frame,
            truth,
            source=source,
            max_time_delta_s=max_time_delta_s,
        )
    return _aligned_residuals_by_scope(
        frame,
        truth,
        source=source,
        max_time_delta_s=max_time_delta_s,
        scope_columns=scope_columns,
    )


_IMPL.aligned_residuals = aligned_residuals

globals().update(
    {
        name: getattr(_IMPL, name)
        for name in dir(_IMPL)
        if not (name.startswith("__") and name.endswith("__"))
    }
)
globals()["_ORIGINAL_ALIGNED_RESIDUALS"] = _ORIGINAL_ALIGNED_RESIDUALS
globals()["_SCOPE_COLUMNS"] = _SCOPE_COLUMNS
globals()["_present_scope_columns"] = _present_scope_columns
globals()["_normalized_scope_keys"] = _normalized_scope_keys
globals()["_aligned_residuals_by_scope"] = _aligned_residuals_by_scope
globals()["aligned_residuals"] = aligned_residuals

__doc__ = _IMPL.__doc__
__all__ = [
    name for name in dir(_IMPL) if not (name.startswith("__") and name.endswith("__"))
]
