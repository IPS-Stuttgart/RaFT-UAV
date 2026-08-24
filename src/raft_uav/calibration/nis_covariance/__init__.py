"""Compatibility package hardening NIS diagnostics and payload validation."""

from __future__ import annotations

from collections.abc import Mapping
import importlib.util
from pathlib import Path
import shlex
import sys

import numpy as np
import pandas as pd

from raft_uav.numeric import optional_float, optional_int

_IMPL_PATH = Path(__file__).resolve().parent.parent / "nis_covariance.py"
_SPEC = importlib.util.spec_from_file_location(
    "raft_uav.calibration._nis_covariance_legacy",
    _IMPL_PATH,
)
if _SPEC is None or _SPEC.loader is None:  # pragma: no cover
    raise ImportError(f"cannot load NIS covariance utilities from {_IMPL_PATH}")
_IMPL = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _IMPL
_SPEC.loader.exec_module(_IMPL)

_LEGACY_NORMALIZED_DIAGNOSTICS_FRAME = _IMPL._normalized_diagnostics_frame


def _truthy(value: object) -> bool:
    """Interpret missing or non-scalar acceptance values as not accepted."""

    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "t", "yes", "y"}
    if value is None:
        return False
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        missing = False
    if isinstance(missing, (bool, np.bool_)):
        if bool(missing):
            return False
    else:
        return False
    try:
        return bool(value)
    except (TypeError, ValueError):
        return False


def _normalized_diagnostics_frame(
    frame: pd.DataFrame,
    *,
    accepted_only: bool,
) -> pd.DataFrame:
    """Normalize diagnostics without inventing source groups or truncating dimensions."""

    required = {"source", "measurement_dim", "nis"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise KeyError(f"diagnostics frame is missing required columns: {missing}")

    work = frame.copy()
    if accepted_only and "accepted" in work.columns:
        work = work.loc[work["accepted"].map(_truthy)].copy()

    _validate_measurement_dimensions(work["measurement_dim"])

    source = work["source"].astype("string").str.strip()
    valid_source = source.notna() & source.ne("").fillna(False)
    work = work.loc[valid_source].copy()
    work["source"] = source.loc[work.index].astype(str)

    return _LEGACY_NORMALIZED_DIAGNOSTICS_FRAME(
        work,
        accepted_only=False,
    )


def _validate_measurement_dimensions(values: pd.Series) -> None:
    """Reject non-real, non-finite, and fractional dimensions before integer casting."""

    raw = pd.Series(values)
    boolean_mask = raw.map(lambda value: isinstance(value, (bool, np.bool_)))
    complex_mask = raw.map(
        lambda value: isinstance(value, (complex, np.complexfloating))
    )
    invalid_scalar_type = boolean_mask | complex_mask
    if invalid_scalar_type.any():
        row_index = int(np.flatnonzero(invalid_scalar_type.to_numpy())[0])
        bad_index = raw.index[row_index]
        bad_value = raw.iloc[row_index]
        raise ValueError(
            "diagnostics measurement_dim values must be real integer dimensions; "
            f"got {bad_value!r} at index {bad_index!r}"
        )

    numbers = pd.to_numeric(raw, errors="coerce")
    numeric = numbers.to_numpy(dtype=float)
    finite = np.isfinite(numeric)
    nonfinite = numbers.notna().to_numpy() & ~finite
    fractional = finite & (numeric != np.rint(numeric))
    invalid = nonfinite | fractional
    if invalid.any():
        row_index = int(np.flatnonzero(invalid)[0])
        bad_index = raw.index[row_index]
        bad_value = raw.iloc[row_index]
        raise ValueError(
            "diagnostics measurement_dim values must be finite integer dimensions; "
            f"got {bad_value!r} at index {bad_index!r}"
        )


def _validate_quantile(value: float) -> float:
    """Return a scalar probability strictly between zero and one."""

    quantile = optional_float(value)
    if quantile is None or not 0.0 < quantile < 1.0:
        raise ValueError("quantile must be in (0, 1)")
    return quantile


def _validate_positive_float(value: float, name: str) -> float:
    """Return a finite positive real scalar without Boolean coercion."""

    number = optional_float(value)
    if number is None or number <= 0.0:
        raise ValueError(f"{name} must be positive and finite")
    return number


def _validate_nonnegative_int(value: int, name: str) -> int:
    """Return an exact non-negative integer without truncating fractions."""

    number = optional_int(value)
    if number is None or number < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return number


def _positive_measurement_dim(value: object, *, name: str) -> int:
    """Return an exact positive measurement dimension."""

    dimension = optional_int(value)
    if dimension is None or dimension <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return dimension


def _enabled_flag(value: object, *, key: object) -> bool:
    """Return a literal Boolean calibration-enable flag."""

    if not isinstance(value, (bool, np.bool_)):
        raise ValueError(f"calibration group {key!r} has invalid enabled flag")
    return bool(value)


def validate_nis_covariance_calibration(payload: Mapping[str, object]) -> None:
    """Validate runtime-relevant NIS calibration metadata without lossy coercion."""

    if not isinstance(payload, Mapping):
        raise ValueError("NIS covariance calibration must be an object")
    if payload.get("schema") != _IMPL.NIS_COVARIANCE_CALIBRATION_SCHEMA:
        raise ValueError("unknown NIS covariance calibration schema")
    groups = payload.get("groups")
    if not isinstance(groups, Mapping):
        raise ValueError("NIS covariance calibration must contain a groups object")
    for key, group in groups.items():
        if not isinstance(group, Mapping):
            raise ValueError(f"calibration group {key!r} must be an object")
        source = str(group.get("source", ""))
        dim = _positive_measurement_dim(
            group.get("measurement_dim"),
            name=f"calibration group {key!r} measurement_dim",
        )
        if key != _IMPL._group_key(source, dim):
            raise ValueError(f"calibration group key {key!r} does not match source/dimension")
        _enabled_flag(group.get("enabled", False), key=key)
        scale = optional_float(group.get("applied_scale", 1.0))
        if scale is None or scale <= 0.0:
            raise ValueError(f"calibration group {key!r} has invalid applied_scale")


def covariance_scale_for_source_dim(
    calibration: Mapping[str, object] | None,
    source: str,
    measurement_dim: int,
) -> float:
    """Return one validated source/dimension covariance multiplier."""

    dimension = _positive_measurement_dim(measurement_dim, name="measurement_dim")
    group = _IMPL._calibration_group(calibration, source, dimension)
    if group is None:
        return 1.0
    key = _IMPL._group_key(source, dimension)
    if not _enabled_flag(group.get("enabled", False), key=key):
        return 1.0
    scale = optional_float(group.get("applied_scale", 1.0))
    if scale is None or scale <= 0.0:
        raise ValueError(f"invalid covariance scale for {source}:{dimension}")
    return scale


def environment_assignment(path: Path | str) -> str:
    """Return a POSIX-shell-safe environment assignment for a calibration file."""

    value = shlex.quote(str(Path(path)))
    return f"{_IMPL.ENV_NIS_COVARIANCE_CALIBRATION_JSON}={value}"


_IMPL._truthy = _truthy
_IMPL._normalized_diagnostics_frame = _normalized_diagnostics_frame
_IMPL._validate_quantile = _validate_quantile
_IMPL._validate_positive_float = _validate_positive_float
_IMPL._validate_nonnegative_int = _validate_nonnegative_int
_IMPL.validate_nis_covariance_calibration = validate_nis_covariance_calibration
_IMPL.covariance_scale_for_source_dim = covariance_scale_for_source_dim
_IMPL.environment_assignment = environment_assignment

globals().update(
    {
        name: getattr(_IMPL, name)
        for name in dir(_IMPL)
        if not (name.startswith("__") and name.endswith("__"))
    }
)
globals()["_truthy"] = _truthy
globals()["_normalized_diagnostics_frame"] = _normalized_diagnostics_frame
globals()["_validate_measurement_dimensions"] = _validate_measurement_dimensions
globals()["_validate_quantile"] = _validate_quantile
globals()["_validate_positive_float"] = _validate_positive_float
globals()["_validate_nonnegative_int"] = _validate_nonnegative_int
globals()["_positive_measurement_dim"] = _positive_measurement_dim
globals()["_enabled_flag"] = _enabled_flag
globals()["validate_nis_covariance_calibration"] = validate_nis_covariance_calibration
globals()["covariance_scale_for_source_dim"] = covariance_scale_for_source_dim
globals()["environment_assignment"] = environment_assignment

__doc__ = _IMPL.__doc__
__all__ = [
    name for name in dir(_IMPL) if not (name.startswith("__") and name.endswith("__"))
]
