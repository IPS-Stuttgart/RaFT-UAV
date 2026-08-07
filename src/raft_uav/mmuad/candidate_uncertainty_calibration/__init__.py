"""Validated public boundary for MMUAD candidate uncertainty calibration.

The maintained implementation lives in the sibling
``candidate_uncertainty_calibration.py`` module.  This package preserves the
public import path while rejecting malformed persisted calibrations before they
can produce non-finite, non-positive, or out-of-contract candidate standard
deviations.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from typing import Any

import numpy as np

_IMPL_PATH = Path(__file__).resolve().parent.parent / "candidate_uncertainty_calibration.py"
_SPEC = importlib.util.spec_from_file_location(
    f"{__name__}._impl",
    _IMPL_PATH,
    submodule_search_locations=[],
)
if _SPEC is None or _SPEC.loader is None:  # pragma: no cover - importlib guard
    raise ImportError(f"could not load candidate uncertainty calibration from {_IMPL_PATH}")
_IMPL = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _IMPL
_SPEC.loader.exec_module(_IMPL)

_ORIGINAL_FIT = _IMPL.fit_candidate_sigma_calibration
_ORIGINAL_SAVE = _IMPL.save_candidate_sigma_calibration
_ORIGINAL_LOAD = _IMPL.load_candidate_sigma_calibration
_ORIGINAL_APPLY = _IMPL.apply_candidate_sigma_calibration


def _finite_float(value: Any, *, field: str) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be a finite number") from exc
    if not np.isfinite(numeric):
        raise ValueError(f"{field} must be a finite number")
    return numeric


def _validate_scale_mapping(
    values: Any,
    *,
    field: str,
    scale_min: float,
    scale_max: float,
) -> None:
    if not isinstance(values, dict):
        raise ValueError(f"{field} must be a mapping")
    for key, value in values.items():
        scale = _finite_float(value, field=f"{field}[{key!r}]")
        if not scale_min <= scale <= scale_max:
            raise ValueError(
                f"{field}[{key!r}] must lie within [{scale_min}, {scale_max}]"
            )


def _validate_candidate_sigma_calibration(calibration: Any) -> Any:
    """Reject calibration objects that can emit invalid standard deviations."""

    if not isinstance(calibration, _IMPL.CandidateSigmaCalibration):
        raise TypeError("calibration must be a CandidateSigmaCalibration")

    target_quantile = _finite_float(
        calibration.target_quantile,
        field="target_quantile",
    )
    if not 0.0 < target_quantile <= 1.0:
        raise ValueError("target_quantile must be in (0, 1]")

    scale_min = _finite_float(calibration.scale_min, field="scale_min")
    scale_max = _finite_float(calibration.scale_max, field="scale_max")
    if not 0.0 < scale_min <= scale_max:
        raise ValueError("scale bounds must satisfy 0 < scale_min <= scale_max")

    global_scale = _finite_float(calibration.global_scale, field="global_scale")
    if not scale_min <= global_scale <= scale_max:
        raise ValueError(f"global_scale must lie within [{scale_min}, {scale_max}]")

    for field in ("source_scales", "branch_scales", "source_branch_scales"):
        _validate_scale_mapping(
            getattr(calibration, field),
            field=field,
            scale_min=scale_min,
            scale_max=scale_max,
        )

    min_group_rows = int(calibration.min_group_rows)
    calibration_row_count = int(calibration.calibration_row_count)
    shrinkage_rows = _finite_float(calibration.shrinkage_rows, field="shrinkage_rows")
    if min_group_rows < 1:
        raise ValueError("min_group_rows must be at least 1")
    if calibration_row_count < 0:
        raise ValueError("calibration_row_count must be non-negative")
    if shrinkage_rows < 0.0:
        raise ValueError("shrinkage_rows must be non-negative")
    return calibration


def fit_candidate_sigma_calibration(*args: Any, **kwargs: Any) -> Any:
    calibration = _ORIGINAL_FIT(*args, **kwargs)
    return _validate_candidate_sigma_calibration(calibration)


def save_candidate_sigma_calibration(calibration: Any, *args: Any, **kwargs: Any) -> Any:
    _validate_candidate_sigma_calibration(calibration)
    return _ORIGINAL_SAVE(calibration, *args, **kwargs)


def load_candidate_sigma_calibration(*args: Any, **kwargs: Any) -> Any:
    calibration = _ORIGINAL_LOAD(*args, **kwargs)
    return _validate_candidate_sigma_calibration(calibration)


def apply_candidate_sigma_calibration(
    candidates: Any,
    calibration: Any,
    *args: Any,
    **kwargs: Any,
) -> Any:
    _validate_candidate_sigma_calibration(calibration)
    if bool(kwargs.get("replace_covariance", False)):
        z_scale = _finite_float(kwargs.get("z_scale", 1.0), field="z_scale")
        if z_scale <= 0.0:
            raise ValueError("z_scale must be positive when replacing covariance")
    return _ORIGINAL_APPLY(candidates, calibration, *args, **kwargs)


_IMPL.fit_candidate_sigma_calibration = fit_candidate_sigma_calibration
_IMPL.save_candidate_sigma_calibration = save_candidate_sigma_calibration
_IMPL.load_candidate_sigma_calibration = load_candidate_sigma_calibration
_IMPL.apply_candidate_sigma_calibration = apply_candidate_sigma_calibration

globals().update(
    {
        name: value
        for name, value in vars(_IMPL).items()
        if not (name.startswith("__") and name.endswith("__"))
    }
)
globals()["_validate_candidate_sigma_calibration"] = _validate_candidate_sigma_calibration
