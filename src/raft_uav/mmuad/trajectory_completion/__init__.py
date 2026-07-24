"""Compatibility fixes for MMUAD trajectory-completion parsing and grid inference.

The maintained implementation lives in the sibling ``trajectory_completion.py``
module. This package preserves the public import path while parsing serialized
``selected_path_update`` values explicitly instead of relying on string
truthiness, and while avoiding floating-point undercounting when inferring
regular timestamps inside short gaps.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

_IMPL_PATH = Path(__file__).resolve().parent.parent / "trajectory_completion.py"
_SPEC = importlib.util.spec_from_file_location(
    "raft_uav.mmuad._trajectory_completion_legacy",
    _IMPL_PATH,
)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(
        f"cannot load MMUAD trajectory completion implementation from {_IMPL_PATH}"
    )
_IMPL = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _IMPL
_SPEC.loader.exec_module(_IMPL)

_ORIGINAL_ESTIMATE_ROWS = _IMPL._estimate_rows
_ORIGINAL_SELECTED_MEASUREMENTS = _IMPL._selected_measurements
_TRUE_TEXT = frozenset({"1", "true", "t", "yes", "y", "on"})
_FALSE_TEXT = frozenset(
    {"0", "false", "f", "no", "n", "off", "", "none", "null", "nan", "<na>", "nat"}
)


def _parse_selected_path_update(value: Any) -> bool:
    """Return one explicit path-selection flag without string truthiness."""

    scalar = value
    if isinstance(value, np.ndarray):
        if value.ndim != 0:
            raise ValueError(f"cannot parse selected_path_update value: {value!r}")
        scalar = value.item()

    if isinstance(scalar, (bool, np.bool_)):
        return bool(scalar)
    if scalar is None:
        return False

    try:
        missing = pd.isna(scalar)
    except (TypeError, ValueError):
        missing = False
    if isinstance(missing, (bool, np.bool_)) and bool(missing):
        return False

    if isinstance(scalar, str):
        text = scalar.strip().lower()
        if text in _TRUE_TEXT:
            return True
        if text in _FALSE_TEXT:
            return False
        raise ValueError(f"cannot parse selected_path_update value: {scalar!r}")

    if isinstance(scalar, (int, float, np.integer, np.floating)):
        numeric = float(scalar)
        return bool(numeric) if np.isfinite(numeric) else False

    raise ValueError(f"cannot parse selected_path_update value: {scalar!r}")


def _normalized_selected_path_updates(rows: pd.DataFrame) -> pd.DataFrame:
    """Return rows with any selected-path flag column normalized to Boolean."""

    normalized = pd.DataFrame(rows).copy()
    if "selected_path_update" in normalized.columns:
        normalized["selected_path_update"] = normalized[
            "selected_path_update"
        ].map(_parse_selected_path_update)
    return normalized


def _estimate_rows(estimates: pd.DataFrame) -> pd.DataFrame:
    """Normalize trajectory estimates after parsing serialized selection flags."""

    return _ORIGINAL_ESTIMATE_ROWS(_normalized_selected_path_updates(estimates))


def _selected_measurements(source: pd.DataFrame) -> pd.DataFrame:
    """Select smoothing measurements after normalizing serialized flags."""

    return _ORIGINAL_SELECTED_MEASUREMENTS(_normalized_selected_path_updates(source))


def _target_times(
    group: pd.DataFrame,
    truth_rows: pd.DataFrame | None,
    *,
    config: Any,
) -> np.ndarray:
    """Build target times without floor-based floating-point undercounting."""

    original = _IMPL._unique_times(group)
    targets = {float(value) for value in original}
    if (
        config.include_truth_timestamps
        and truth_rows is not None
        and not truth_rows.empty
    ):
        for timestamp in pd.to_numeric(
            truth_rows["time_s"],
            errors="coerce",
        ).to_numpy(float):
            if np.isfinite(timestamp) and _IMPL._time_supported_by_short_gap(
                timestamp,
                original,
                config.max_gap_s,
            ):
                targets.add(float(timestamp))
    elif config.infer_missing_grid:
        step = _IMPL._typical_step_s(original)
        if np.isfinite(step) and step > 0.0:
            for left, right in zip(original[:-1], original[1:], strict=False):
                gap_s = float(right - left)
                if (
                    gap_s <= max(float(config.max_gap_s), step)
                    and gap_s > 1.5 * step
                ):
                    index = 1
                    previous = float(left)
                    while True:
                        value = float(left + index * step)
                        if value <= previous or value >= right - 1.0e-9:
                            break
                        targets.add(value)
                        previous = value
                        index += 1
    return np.asarray(sorted(targets), dtype=float)


_IMPL._parse_selected_path_update = _parse_selected_path_update
_IMPL._normalized_selected_path_updates = _normalized_selected_path_updates
_IMPL._estimate_rows = _estimate_rows
_IMPL._selected_measurements = _selected_measurements
_IMPL._target_times = _target_times

globals().update(
    {
        name: getattr(_IMPL, name)
        for name in dir(_IMPL)
        if not (name.startswith("__") and name.endswith("__"))
    }
)
globals()["_parse_selected_path_update"] = _parse_selected_path_update
globals()["_normalized_selected_path_updates"] = _normalized_selected_path_updates
globals()["_estimate_rows"] = _estimate_rows
globals()["_selected_measurements"] = _selected_measurements
globals()["_target_times"] = _target_times

__doc__ = _IMPL.__doc__
__all__ = [
    name for name in dir(_IMPL) if not (name.startswith("__") and name.endswith("__"))
]
