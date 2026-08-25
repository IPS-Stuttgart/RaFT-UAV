"""Compatibility wrapper protecting uncertainty-adapter output integrity.

The maintained implementation lives in the sibling
``track5_uncertainty_column_adapter.py`` module. This package preserves the
public import path while rejecting labels, uncertainty columns, and path
configurations that could make normalized outputs ambiguous or overwrite
estimate inputs.
"""

from __future__ import annotations

import importlib.util
import os
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
_ORIGINAL_WRITE_OUTPUTS = _IMPL.write_uncertainty_column_adapter_outputs
_ORIGINAL_PARSE_COLUMN_MAP = _IMPL._parse_uncertainty_column_map


def _validate_unique_estimate_labels(estimate_inputs: Iterable[object]) -> list[object]:
    """Materialize inputs and reject normalized output-filename collisions."""

    inputs = list(estimate_inputs)
    original_labels: dict[str, tuple[str, str]] = {}
    for item in inputs:
        raw_label = str(item.label)
        normalized_label = _IMPL._safe_label(raw_label)
        portable_key = normalized_label.casefold()
        if portable_key in original_labels:
            previous, previous_normalized = original_labels[portable_key]
            raise ValueError(
                "estimate labels must be unique after normalization, including "
                "case-insensitive filenames; "
                f"{previous!r} maps to {previous_normalized!r} and "
                f"{raw_label!r} maps to {normalized_label!r}"
            )
        original_labels[portable_key] = (raw_label, normalized_label)
    return inputs


def _validate_uncertainty_column_mapping(
    inputs: Iterable[object],
    uncertainty_columns: dict[object, str] | None,
) -> dict[object, str]:
    """Reject unknown mapping labels and multiple aliases for one estimate."""

    mapping = dict(uncertainty_columns or {})
    consumed_keys: set[object] = set()
    for item in inputs:
        raw_label = str(item.label)
        safe_label = _IMPL._safe_label(raw_label)
        aliases = {raw_label, safe_label}
        matched_keys = [key for key in mapping if key in aliases]
        if len(matched_keys) > 1:
            rendered = ", ".join(sorted((repr(key) for key in matched_keys)))
            raise ValueError(
                "uncertainty_columns must define at most one mapping per estimate; "
                f"{raw_label!r} is addressed by multiple labels: {rendered}"
            )
        consumed_keys.update(matched_keys)

    unused_keys = [key for key in mapping if key not in consumed_keys]
    if unused_keys:
        rendered = ", ".join(sorted((repr(key) for key in unused_keys)))
        raise ValueError(
            "uncertainty_columns contains labels that do not match estimate inputs: "
            f"{rendered}"
        )
    return mapping


def _canonical_path(path: Path) -> str:
    """Return a normalized path key without requiring the path to exist."""

    candidate = Path(path).expanduser()
    try:
        resolved = candidate.resolve(strict=False)
    except (OSError, RuntimeError):
        resolved = candidate.absolute()
    return os.path.normcase(os.fspath(resolved))


def _paths_alias(left: Path, right: Path) -> bool:
    """Return whether two paths identify the same filesystem object."""

    if _canonical_path(left) == _canonical_path(right):
        return True
    try:
        return left.exists() and right.exists() and os.path.samefile(left, right)
    except OSError:
        return False


def _validate_normalized_output_paths(
    inputs: Iterable[object],
    *,
    output_dir: Path,
) -> None:
    """Reject normalized outputs that would overwrite any estimate input."""

    materialized = list(inputs)
    normalized_dir = Path(output_dir) / _IMPL.NORMALIZED_DIR
    planned_outputs = [
        (
            str(item.label),
            normalized_dir / f"{_IMPL._safe_label(str(item.label))}.csv",
        )
        for item in materialized
    ]
    for input_item in materialized:
        input_path = Path(input_item.path)
        for output_label, output_path in planned_outputs:
            if not _paths_alias(input_path, output_path):
                continue
            raise ValueError(
                "normalized estimate outputs must not overwrite estimate inputs; "
                f"output for {output_label!r} aliases input for "
                f"{str(input_item.label)!r}: {output_path}"
            )


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
    """Normalize inputs only after proving every planned write is safe."""

    inputs = _validate_unique_estimate_labels(estimate_inputs)
    column_map = _validate_uncertainty_column_mapping(inputs, uncertainty_columns)
    _validate_normalized_output_paths(inputs, output_dir=Path(output_dir))
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


def write_uncertainty_column_adapter_outputs(
    *,
    estimate_inputs,
    output_dir,
    uncertainty_columns=None,
    output_uncertainty_column="predicted_sigma_m",
    fallback_sigma_m=30.0,
    require_uncertainty=False,
    template=None,
    class_map=None,
    default_classification=0,
    sigma_min_m=1.0,
    sigma_max_m=100.0,
    max_nearest_time_delta_s=None,
    run_ensemble=False,
):
    """Validate ensemble prerequisites before normalized outputs are written."""

    if run_ensemble and template is None:
        raise ValueError("template is required when run_ensemble=True")
    return _ORIGINAL_WRITE_OUTPUTS(
        estimate_inputs=estimate_inputs,
        output_dir=output_dir,
        uncertainty_columns=uncertainty_columns,
        output_uncertainty_column=output_uncertainty_column,
        fallback_sigma_m=fallback_sigma_m,
        require_uncertainty=require_uncertainty,
        template=template,
        class_map=class_map,
        default_classification=default_classification,
        sigma_min_m=sigma_min_m,
        sigma_max_m=sigma_max_m,
        max_nearest_time_delta_s=max_nearest_time_delta_s,
        run_ensemble=run_ensemble,
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
_IMPL.write_uncertainty_column_adapter_outputs = write_uncertainty_column_adapter_outputs
_IMPL._parse_uncertainty_column_map = _parse_uncertainty_column_map

globals().update(
    {
        name: getattr(_IMPL, name)
        for name in dir(_IMPL)
        if not (name.startswith("__") and name.endswith("__"))
    }
)
globals()["_validate_uncertainty_column_mapping"] = _validate_uncertainty_column_mapping
globals()["_canonical_path"] = _canonical_path
globals()["_paths_alias"] = _paths_alias
globals()["_validate_normalized_output_paths"] = _validate_normalized_output_paths
globals()["_read_physical_header"] = _read_physical_header
globals()["_validate_unambiguous_uncertainty_columns"] = (
    _validate_unambiguous_uncertainty_columns
)
globals()["normalize_uncertainty_estimate_inputs"] = normalize_uncertainty_estimate_inputs
globals()["write_uncertainty_column_adapter_outputs"] = (
    write_uncertainty_column_adapter_outputs
)
globals()["_parse_uncertainty_column_map"] = _parse_uncertainty_column_map

__doc__ = _IMPL.__doc__
__all__ = [
    name for name in dir(_IMPL) if not (name.startswith("__") and name.endswith("__"))
]
