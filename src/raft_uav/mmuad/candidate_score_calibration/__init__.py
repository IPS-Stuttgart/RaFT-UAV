"""Compatibility fixes for candidate score calibration.

The maintained implementation lives in the sibling
``candidate_score_calibration.py`` module. This package preserves the public
import path while validating persisted logit offsets and retaining candidate
row order through truth matching.
"""

from __future__ import annotations

from collections.abc import Mapping
import importlib.util
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

_IMPL_PATH = Path(__file__).resolve().parent.parent / "candidate_score_calibration.py"
_SPEC = importlib.util.spec_from_file_location(
    "raft_uav.mmuad._candidate_score_calibration_legacy",
    _IMPL_PATH,
)
if _SPEC is None or _SPEC.loader is None:  # pragma: no cover
    raise ImportError(f"cannot load candidate score calibration implementation from {_IMPL_PATH}")
_IMPL = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _IMPL
_SPEC.loader.exec_module(_IMPL)
_ORIGINAL_VALIDATE_MODEL = _IMPL._validate_model
_ORIGINAL_ATTACH_TRUTH_TARGETS = _IMPL._attach_truth_targets


def _finite_logit_offset(value: Any, *, name: str) -> float:
    """Return one finite scalar logit offset without lossy coercion."""

    message = f"{name} must be a finite scalar logit offset"
    if np.ma.is_masked(value) or isinstance(value, (bool, np.bool_)):
        raise ValueError(message)
    try:
        array = np.asarray(value)
        if array.ndim != 0:
            raise TypeError
        scalar = array.item()
        if np.ma.is_masked(scalar) or isinstance(scalar, (bool, np.bool_)):
            raise TypeError
        numeric = float(scalar)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(message) from exc
    if not np.isfinite(numeric):
        raise ValueError(message)
    return numeric


def _validate_offset_mapping(value: Any, *, name: str) -> None:
    """Validate a group/class mapping of finite scalar logit offsets."""

    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping of group/class logit offsets")
    for group, class_map in value.items():
        if not isinstance(class_map, Mapping):
            raise ValueError(f"{name}[{group!r}] must be a mapping of class logit offsets")
        for class_label, offset in class_map.items():
            _finite_logit_offset(
                offset,
                name=f"{name}[{group!r}][{class_label!r}]",
            )


def _validate_model(model: Mapping[str, Any]) -> None:
    """Validate the legacy schema plus every persisted logit offset."""

    _ORIGINAL_VALIDATE_MODEL(model)
    _finite_logit_offset(
        model.get("global_logit_offset", 0.0),
        name="global_logit_offset",
    )
    for key in (
        "branch_class_logit_offsets",
        "source_class_logit_offsets",
        "branch_source_class_logit_offsets",
    ):
        _validate_offset_mapping(model.get(key, {}), name=key)


def _attach_truth_targets(
    candidates: pd.DataFrame,
    truth: pd.DataFrame,
    *,
    good_threshold_m: float,
    max_truth_time_delta_s: float,
) -> pd.DataFrame:
    """Attach truth targets without regrouping the caller's candidate rows."""

    rows = pd.DataFrame(candidates).copy()
    order_column = "__candidate_score_calibration_input_row_order"
    while order_column in rows.columns:
        order_column = f"_{order_column}"
    rows[order_column] = np.arange(len(rows), dtype=int)

    labelled = _ORIGINAL_ATTACH_TRUTH_TARGETS(
        rows,
        truth,
        good_threshold_m=good_threshold_m,
        max_truth_time_delta_s=max_truth_time_delta_s,
    )
    if order_column not in labelled.columns:  # pragma: no cover - defensive
        return labelled
    return (
        labelled.sort_values(order_column, kind="stable")
        .drop(columns=[order_column])
        .reset_index(drop=True)
    )


_IMPL._validate_model = _validate_model
_IMPL._attach_truth_targets = _attach_truth_targets

globals().update(
    {
        name: getattr(_IMPL, name)
        for name in dir(_IMPL)
        if not (name.startswith("__") and name.endswith("__"))
    }
)
globals()["_finite_logit_offset"] = _finite_logit_offset
globals()["_validate_offset_mapping"] = _validate_offset_mapping
globals()["_validate_model"] = _validate_model
globals()["_attach_truth_targets"] = _attach_truth_targets

__doc__ = _IMPL.__doc__
__all__ = [
    name for name in dir(_IMPL) if not (name.startswith("__") and name.endswith("__"))
]
