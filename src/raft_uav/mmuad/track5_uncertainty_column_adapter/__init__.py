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

import pandas as pd

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


def _read_physical_header(path: Path) -> list[str]:
    """Read the unmangled CSV header before pandas deduplicates names."""

    header = pd.read_csv(
        path,
        dtype=str,
        keep_default_na=False,
        header=None,
        nrows=1,
    )
    if header.empty:
        return []
    return [str(value) for value in header.iloc[0].tolist()]


def _validate_unambiguous_uncertainty_columns(
    path: Path,
    *,
    label: str,
    requested: str | None,
) -> None:
    """Reject physical uncertainty columns that collapse to one normalized name."""

    groups: dict[str, list[str]] = {}
    for column in _read_physical_header(path):
        groups.setdefault(_IMPL._column_name_key(column), []).append(column)

    candidate_keys = (
        {_IMPL._column_name_key(requested)}
        if requested is not None
        else {_IMPL._column_name_key(column) for column in _IMPL.DEFAULT_UNCERTAINTY_COLUMNS}
    )
    ambiguous_columns = sorted(
        {
            column
            for key in candidate_keys
            for column in groups.get(key, [])
            if len(groups.get(key, [])) > 1
        },
        key=lambda column: (_IMPL._column_name_key(column), column),
    )
    if ambiguous_columns:
        rendered = ", ".join(repr(column) for column in ambiguous_columns)
        raise ValueError(
            f"estimate CSV for {label!r} has ambiguous uncertainty columns after "
            f"trimming whitespace and ignoring case: {rendered}"
        )


def normalize_uncertainty_estimate_inputs(
    estimate_inputs,
    *,
    output_dir,
    uncertainty_columns=None,
    output_uncertainty_column="predicted_sigma_m",
    fallback_sigma_m=30.0,
    require_uncertainty=False,
):
    """Normalize inputs only after proving labels and uncertainty columns are distinct."""

    inputs = _validate_unique_estimate_labels(estimate_inputs)
    column_map = dict(uncertainty_columns or {})
    for item in inputs:
        requested = _IMPL._lookup_requested_uncertainty_column(column_map, item.label)
        _validate_unambiguous_uncertainty_columns(
            Path(item.path),
            label=str(item.label),
            requested=requested,
        )
    return _ORIGINAL_NORMALIZE(
        inputs,
        output_dir=output_dir,
        uncertainty_columns=column_map,
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
globals()["_read_physical_header"] = _read_physical_header
globals()["_validate_unambiguous_uncertainty_columns"] = (
    _validate_unambiguous_uncertainty_columns
)
globals()["normalize_uncertainty_estimate_inputs"] = normalize_uncertainty_estimate_inputs
globals()["_parse_uncertainty_column_map"] = _parse_uncertainty_column_map

__doc__ = _IMPL.__doc__
__all__ = [
    name for name in dir(_IMPL) if not (name.startswith("__") and name.endswith("__"))
]
