"""Compatibility wrapper for strict train-selected configuration handling.

The maintained implementation lives in the sibling ``train_selected_config.py``
module. This package preserves the public import path while making alias
selection skip missing values, ensuring component-level fixed updates cannot be
overridden by summary CSV aliases, rejecting malformed numeric controls before
Python or NumPy can silently coerce them, rejecting summary tables without a
usable finite selection metric, and refusing classifier-fusion weights that the
train-to-validation pipeline does not consume.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

_IMPL_PATH = Path(__file__).resolve().parent.parent / "train_selected_config.py"
_SPEC = importlib.util.spec_from_file_location(
    "raft_uav.mmuad._train_selected_config_legacy",
    _IMPL_PATH,
)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"cannot load train-selected config implementation from {_IMPL_PATH}")
_IMPL = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _IMPL
_SPEC.loader.exec_module(_IMPL)

_ORIGINAL_BEST_ROW = _IMPL._best_row
_ORIGINAL_SELECT_COMPONENT = _IMPL._select_component
_ORIGINAL_VALIDATE_TRAIN_SELECTED_CONFIG = _IMPL.validate_train_selected_config


def _first_present(row: pd.Series, columns: tuple[str, ...]) -> Any:
    """Return the first present, non-missing alias value from ``row``."""

    for column in columns:
        if column not in row.index:
            continue
        value = row[column]
        if not _IMPL._is_nan(value):
            return value
    return None


def _best_row(
    csv_path: Path,
    metric_columns: tuple[str, ...],
    *,
    maximize: bool,
) -> tuple[pd.Series, str | None, float | None, int]:
    """Select only from a recognized metric with at least one finite value."""

    row, metric_column, metric_value, row_index = _ORIGINAL_BEST_ROW(
        csv_path,
        metric_columns,
        maximize=maximize,
    )
    if metric_column is not None:
        return row, metric_column, metric_value, row_index

    columns = set(pd.read_csv(csv_path, nrows=0).columns)
    recognized = tuple(column for column in metric_columns if column in columns)
    if not recognized:
        expected = ", ".join(metric_columns)
        raise ValueError(
            f"{csv_path} has no recognized selection metric column; "
            f"expected one of: {expected}"
        )

    present = ", ".join(recognized)
    raise ValueError(
        f"{csv_path} has no finite values in recognized selection metric "
        f"columns: {present}"
    )


def _select_component(
    config: dict[str, Any],
    records: list[dict[str, Any]],
    *,
    component: str,
    csv_path: Path | None,
    mappings: dict[str, tuple[str, ...]],
    metric_columns: tuple[str, ...],
    maximize: bool,
    fixed_updates: dict[str, Any] | None = None,
) -> None:
    """Apply component-fixed values after CSV-derived aliases."""

    _ORIGINAL_SELECT_COMPONENT(
        config,
        records,
        component=component,
        csv_path=csv_path,
        mappings=mappings,
        metric_columns=metric_columns,
        maximize=maximize,
        fixed_updates=fixed_updates,
    )
    if csv_path is None or not fixed_updates:
        return
    config.update(fixed_updates)
    records[-1].update(
        {key: _IMPL._jsonable(value) for key, value in fixed_updates.items()}
    )


def _float(value: Any) -> float:
    """Return a finite scalar float without lossy implicit coercion."""

    message = f"expected finite float, got {value!r}"
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(message)
    if np.ma.isMaskedArray(value):
        if bool(np.ma.getmaskarray(value).any()):
            raise ValueError(message)
        value = np.ma.getdata(value)
    try:
        array = np.asarray(value)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(message) from None
    if array.ndim != 0:
        raise ValueError(message)
    scalar = array.item()
    if isinstance(scalar, (bool, np.bool_, complex, np.complexfloating)):
        raise ValueError(message)
    try:
        number = float(scalar)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(message) from None
    if not np.isfinite(number):
        raise ValueError(message)
    return number


def validate_train_selected_config(config: dict[str, Any]) -> dict[str, Any]:
    """Reject train-selected settings that the validation harness ignores."""

    normalized = _ORIGINAL_VALIDATE_TRAIN_SELECTED_CONFIG(config)
    if normalized["image_nonimage_fusion_weight"] != 0.0:
        raise ValueError(
            "image_nonimage_fusion_weight must be 0.0 because the "
            "train-to-validation pipeline does not consume image-fusion outputs; "
            "use the sequence_classifier_fusion workflow instead"
        )
    return normalized


_IMPL._first_present = _first_present
_IMPL._best_row = _best_row
_IMPL._select_component = _select_component
_IMPL._float = _float
_IMPL.validate_train_selected_config = validate_train_selected_config

globals().update(
    {
        name: getattr(_IMPL, name)
        for name in dir(_IMPL)
        if not (name.startswith("__") and name.endswith("__"))
    }
)
globals()["_first_present"] = _first_present
globals()["_best_row"] = _best_row
globals()["_select_component"] = _select_component
globals()["_float"] = _float
globals()["validate_train_selected_config"] = validate_train_selected_config

__doc__ = _IMPL.__doc__
__all__ = [name for name in dir(_IMPL) if not (name.startswith("__") and name.endswith("__"))]
