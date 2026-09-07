"""Compatibility fixes for Track 5 temporal-repair inputs and outputs.

The maintained implementation lives in the sibling ``track5_temporal_repair.py``
module. This package preserves the public import path while rejecting invalid
iteration controls, stabilizing finite-extreme arithmetic, and normalizing
persisted Boolean diagnostics before output summaries are computed.
"""

from __future__ import annotations

import importlib.util
from numbers import Integral
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

from raft_uav.mmuad._track5_temporal_repair_numeric_stability_patch import (
    install as _install_numeric_stability,
)

_IMPL_PATH = Path(__file__).resolve().parent.parent / "track5_temporal_repair.py"
_SPEC = importlib.util.spec_from_file_location(
    "raft_uav.mmuad._track5_temporal_repair_legacy",
    _IMPL_PATH,
)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"cannot load temporal-repair implementation from {_IMPL_PATH}")
_IMPL = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _IMPL
_SPEC.loader.exec_module(_IMPL)
_install_numeric_stability(_IMPL)

_ORIGINAL_REPAIR_TRACK5_TEMPORAL_SPIKES = _IMPL.repair_track5_temporal_spikes
_ORIGINAL_WRITE_TRACK5_TEMPORAL_REPAIR_OUTPUTS = (
    _IMPL.write_track5_temporal_repair_outputs
)

_TRUE_REPAIRED_FLAG_TEXT = frozenset({"true", "t", "yes", "y", "on"})
_FALSE_REPAIRED_FLAG_TEXT = frozenset(
    {
        "",
        "false",
        "f",
        "no",
        "n",
        "off",
        "none",
        "null",
        "nan",
        "na",
        "n/a",
        "<na>",
        "nat",
    }
)


def _validate_iterations(value: Any) -> int:
    """Return an exact positive integer iteration count."""

    message = "iterations must be an exact positive integer"
    if isinstance(value, (bool, np.bool_)) or np.ma.is_masked(value):
        raise ValueError(f"{message}: {value!r}")
    try:
        array = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{message}: {value!r}") from exc
    if array.ndim != 0 or array.dtype.kind in {"b", "c"}:
        raise ValueError(f"{message}: {value!r}")
    scalar = array.item()
    if isinstance(scalar, (bool, np.bool_)):
        raise ValueError(f"{message}: {value!r}")
    if isinstance(scalar, Integral):
        iterations = int(scalar)
    elif isinstance(scalar, (float, np.floating)):
        if not np.isfinite(scalar):
            raise ValueError(f"{message}: {value!r}")
        iterations = int(scalar)
        if scalar != iterations:
            raise ValueError(f"{message}: {value!r}")
    else:
        try:
            numeric = float(scalar)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"{message}: {value!r}") from exc
        if not np.isfinite(numeric) or not numeric.is_integer():
            raise ValueError(f"{message}: {value!r}")
        iterations = int(numeric)
    if iterations <= 0:
        raise ValueError(f"{message}: {value!r}")
    return iterations


def _parse_boolean_flag(value: Any) -> bool:
    """Parse one persisted repair flag without Python truthiness coercion."""

    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if value is None or np.ma.is_masked(value):
        return False
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, str):
        text = value.strip().casefold()
        if text in _TRUE_REPAIRED_FLAG_TEXT:
            return True
        if text in _FALSE_REPAIRED_FLAG_TEXT:
            return False
        try:
            numeric = float(text)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError from exc
        if np.isfinite(numeric) and numeric == 0.0:
            return False
        if np.isfinite(numeric) and numeric == 1.0:
            return True
        raise ValueError

    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        missing = False
    if isinstance(missing, (bool, np.bool_)) and bool(missing):
        return False

    try:
        array = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise ValueError from exc
    if array.ndim != 0 or array.dtype.kind in {"b", "c"}:
        raise ValueError
    try:
        numeric = float(array.item())
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError from exc
    if np.isfinite(numeric) and numeric == 0.0:
        return False
    if np.isfinite(numeric) and numeric == 1.0:
        return True
    raise ValueError


def _boolean_series(values: Any, index: pd.Index) -> pd.Series:
    """Parse Boolean-like repair flags and reject malformed persisted values."""

    series = pd.Series(values, index=index, copy=False)
    if series.empty:
        return pd.Series(False, index=index, dtype=bool)
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)

    parsed: list[bool] = []
    invalid_rows: list[Any] = []
    invalid_values: list[Any] = []
    for position, value in enumerate(series.tolist()):
        try:
            parsed.append(_parse_boolean_flag(value))
        except ValueError:
            parsed.append(False)
            invalid_rows.append(series.index[position])
            invalid_values.append(value)
    if invalid_rows:
        raise ValueError(
            "repaired contains invalid Boolean values at rows "
            f"{invalid_rows}: {invalid_values}; expected booleans, exact numeric "
            "0/1, recognized Boolean text, or missing values"
        )
    return pd.Series(parsed, index=series.index, dtype=bool)


def repair_track5_temporal_spikes(
    submission: Any,
    *,
    max_speed_mps: float = 80.0,
    max_interpolation_residual_m: float = 25.0,
    iterations: Any = 2,
) -> tuple[Any, Any]:
    """Return repaired estimates after validating the requested pass count."""

    return _ORIGINAL_REPAIR_TRACK5_TEMPORAL_SPIKES(
        submission,
        max_speed_mps=max_speed_mps,
        max_interpolation_residual_m=max_interpolation_residual_m,
        iterations=_validate_iterations(iterations),
    )


def write_track5_temporal_repair_outputs(
    *,
    repaired: pd.DataFrame,
    diagnostics: pd.DataFrame,
    output_dir: Path,
    input_submission_path: Path,
    template: pd.DataFrame | None = None,
    manifest: dict[str, Any] | None = None,
    require_leaderboard_ready: bool = False,
) -> dict[str, Path]:
    """Write outputs after normalizing persisted repair flags."""

    normalized_diagnostics = pd.DataFrame(diagnostics).copy()
    if "repaired" in normalized_diagnostics.columns:
        normalized_diagnostics["repaired"] = _boolean_series(
            normalized_diagnostics["repaired"],
            normalized_diagnostics.index,
        )
    return _ORIGINAL_WRITE_TRACK5_TEMPORAL_REPAIR_OUTPUTS(
        repaired=repaired,
        diagnostics=normalized_diagnostics,
        output_dir=output_dir,
        input_submission_path=input_submission_path,
        template=template,
        manifest=manifest,
        require_leaderboard_ready=require_leaderboard_ready,
    )


_IMPL.repair_track5_temporal_spikes = repair_track5_temporal_spikes
_IMPL.write_track5_temporal_repair_outputs = write_track5_temporal_repair_outputs

globals().update(
    {
        name: getattr(_IMPL, name)
        for name in dir(_IMPL)
        if not (name.startswith("__") and name.endswith("__"))
    }
)
globals()["_validate_iterations"] = _validate_iterations
globals()["_parse_boolean_flag"] = _parse_boolean_flag
globals()["_boolean_series"] = _boolean_series
globals()["repair_track5_temporal_spikes"] = repair_track5_temporal_spikes
globals()["write_track5_temporal_repair_outputs"] = (
    write_track5_temporal_repair_outputs
)

__doc__ = _IMPL.__doc__
__all__ = [
    name for name in dir(_IMPL) if not (name.startswith("__") and name.endswith("__"))
]
