"""Compatibility wrapper preventing uncertainty-adapter label collisions.

The maintained implementation lives in the sibling
``track5_uncertainty_column_adapter.py`` module. This package preserves the
public import path while rejecting labels that would overwrite the same
normalized estimate CSV.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from typing import Iterable

_IMPL_PATH = Path(__file__).resolve().parent.parent / "track5_uncertainty_column_adapter.py"
_SPEC = importlib.util.spec_from_file_location(
    "raft_uav.mmuad._track5_uncertainty_column_adapter_legacy",
    _IMPL_PATH,
)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"cannot load uncertainty column adapter implementation from {_IMPL_PATH}")
_IMPL = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _IMPL
_SPEC.loader.exec_module(_IMPL)

_ORIGINAL_NORMALIZE = _IMPL.normalize_uncertainty_estimate_inputs
_ORIGINAL_PARSE_COLUMN_MAP = _IMPL._parse_uncertainty_column_map


def _validate_unique_estimate_labels(estimate_inputs: Iterable[object]) -> list[object]:
    """Materialize inputs and reject normalized output-filename collisions."""

    inputs = list(estimate_inputs)
    original_labels: dict[str, str] = {}
    for item in inputs:
        raw_label = str(item.label)
        normalized_label = _IMPL._safe_label(raw_label)
        if normalized_label in original_labels:
            previous = original_labels[normalized_label]
            raise ValueError(
                "estimate labels must be unique after normalization; "
                f"{previous!r} and {raw_label!r} both map to {normalized_label!r}"
            )
        original_labels[normalized_label] = raw_label
    return inputs


def normalize_uncertainty_estimate_inputs(
    estimate_inputs,
    *,
    output_dir,
    uncertainty_columns=None,
    output_uncertainty_column="predicted_sigma_m",
    fallback_sigma_m=30.0,
    require_uncertainty=False,
):
    """Normalize inputs only after proving their output paths are distinct."""

    inputs = _validate_unique_estimate_labels(estimate_inputs)
    return _ORIGINAL_NORMALIZE(
        inputs,
        output_dir=output_dir,
        uncertainty_columns=uncertainty_columns,
        output_uncertainty_column=output_uncertainty_column,
        fallback_sigma_m=fallback_sigma_m,
        require_uncertainty=require_uncertainty,
    )


def _parse_uncertainty_column_map(values: list[str]) -> dict[str, str]:
    """Reject CLI label aliases that normalize to the same mapping key."""

    mapping: dict[str, str] = {}
    original_labels: dict[str, str] = {}
    for value in values:
        parsed = _ORIGINAL_PARSE_COLUMN_MAP([value])
        normalized_label, column = next(iter(parsed.items()))
        raw_label = value.split("=", 1)[0]
        if normalized_label in mapping:
            previous = original_labels[normalized_label]
            raise ValueError(
                "uncertainty-column labels must be unique after normalization; "
                f"{previous!r} and {raw_label!r} both map to {normalized_label!r}"
            )
        mapping[normalized_label] = column
        original_labels[normalized_label] = raw_label
    return mapping


_IMPL.normalize_uncertainty_estimate_inputs = normalize_uncertainty_estimate_inputs
_IMPL._parse_uncertainty_column_map = _parse_uncertainty_column_map

globals().update(
    {
        name: getattr(_IMPL, name)
        for name in dir(_IMPL)
        if not (name.startswith("__") and name.endswith("__"))
    }
)
globals()["normalize_uncertainty_estimate_inputs"] = normalize_uncertainty_estimate_inputs
globals()["_parse_uncertainty_column_map"] = _parse_uncertainty_column_map

__doc__ = _IMPL.__doc__
__all__ = [
    name for name in dir(_IMPL) if not (name.startswith("__") and name.endswith("__"))
]
