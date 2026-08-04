"""Compatibility wrapper validating Track 5 acceleration-limit controls.

The maintained implementation lives in the sibling ``track5_acceleration_limit.py``
module. This package preserves the public import path while rejecting malformed
scalar controls, missing sequence identifiers, invalid normalized rows, invalid
classification labels, and duplicate fixed-grid keys, and while keeping zero-blend
runs diagnostic-only.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import ModuleType
from typing import Any

import numpy as np
import pandas as pd

from raft_uav.mmuad.submission import parse_official_classification_cell

_IMPL_PATH = Path(__file__).resolve().parent.parent / "track5_acceleration_limit.py"
_SPEC = importlib.util.spec_from_file_location(
    "raft_uav.mmuad._track5_acceleration_limit_legacy",
    _IMPL_PATH,
)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(
        f"cannot load Track 5 acceleration-limit implementation from {_IMPL_PATH}"
    )
_IMPL = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _IMPL
_SPEC.loader.exec_module(_IMPL)

_ORIGINAL_REPAIR = _IMPL.repair_track5_acceleration_kinks
_ORIGINAL_REPAIR_SEQUENCE = _IMPL._repair_sequence
_NUMERIC_COLUMNS = (
    "time_s",
    "state_x_m",
    "state_y_m",
    "state_z_m",
    "Classification",
)


class _Track5AccelerationLimitModule(ModuleType):
    """Module proxy that keeps runtime monkeypatches visible to legacy globals."""

    def __setattr__(self, name: str, value: Any) -> None:
        super().__setattr__(name, value)
        if name == "_IMPL":
            return
        implementation = self.__dict__.get("_IMPL")
        if implementation is not None and hasattr(implementation, name):
            setattr(implementation, name, value)


def _unwrap_numeric_scalar(value: object) -> tuple[object, str | None]:
    """Recursively unwrap one scalar and classify unsafe array payloads."""

    scalar = value
    seen_arrays: set[int] = set()
    while True:
        if np.ma.is_masked(scalar):
            return scalar, "masked"
        if isinstance(scalar, (bool, np.bool_)):
            return scalar, "Boolean"
        if isinstance(scalar, (complex, np.complexfloating)):
            return scalar, "complex"
        if isinstance(scalar, np.ndarray):
            identity = id(scalar)
            if identity in seen_arrays:
                return scalar, "non-scalar"
            seen_arrays.add(identity)
            if isinstance(scalar, np.ma.MaskedArray):
                if bool(np.ma.getmaskarray(scalar).any()):
                    return scalar, "masked"
                if scalar.ndim != 0:
                    return scalar, "non-scalar"
                scalar = scalar.data
                continue
            if scalar.ndim != 0:
                return scalar, "non-scalar"
            scalar = scalar.item()
            continue
        if isinstance(scalar, np.generic):
            scalar = scalar.item()
            continue
        return scalar, None


def _finite_scalar(value: object, *, message: str) -> float:
    """Return a finite non-Boolean scalar float."""

    scalar, kind = _unwrap_numeric_scalar(value)
    if kind is not None:
        raise ValueError(message)
    try:
        numeric = float(scalar)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(message) from exc
    if not np.isfinite(numeric):
        raise ValueError(message)
    return numeric


def _positive_integer(value: object, *, name: str) -> int:
    """Return a positive integer without lossy or Boolean coercion."""

    message = f"{name} must be a positive finite integer"
    numeric = _finite_scalar(value, message=message)
    if numeric <= 0.0 or not numeric.is_integer():
        raise ValueError(message)
    return int(numeric)


def _validated_controls(
    *,
    max_acceleration_mps2: object,
    max_direct_speed_mps: object,
    min_interpolation_residual_m: object,
    iterations: object,
    repair_blend: object,
) -> dict[str, float | int]:
    """Normalize and validate all public acceleration-repair controls."""

    max_acceleration = _finite_scalar(
        max_acceleration_mps2,
        message="max_acceleration_mps2 must be positive and finite",
    )
    if max_acceleration <= 0.0:
        raise ValueError("max_acceleration_mps2 must be positive and finite")

    max_direct_speed = _finite_scalar(
        max_direct_speed_mps,
        message="max_direct_speed_mps must be positive and finite",
    )
    if max_direct_speed <= 0.0:
        raise ValueError("max_direct_speed_mps must be positive and finite")

    min_residual = _finite_scalar(
        min_interpolation_residual_m,
        message="min_interpolation_residual_m must be finite and non-negative",
    )
    if min_residual < 0.0:
        raise ValueError(
            "min_interpolation_residual_m must be finite and non-negative"
        )

    blend = _finite_scalar(
        repair_blend,
        message="repair_blend must be finite and in [0, 1]",
    )
    if not 0.0 <= blend <= 1.0:
        raise ValueError("repair_blend must be finite and in [0, 1]")

    return {
        "max_acceleration_mps2": max_acceleration,
        "max_direct_speed_mps": max_direct_speed,
        "min_interpolation_residual_m": min_residual,
        "iterations": _positive_integer(iterations, name="iterations"),
        "repair_blend": blend,
    }


def _validate_sequence_ids(submission: object) -> None:
    """Reject genuinely missing or blank sequence identifiers before string conversion."""

    rows = pd.DataFrame(submission).copy()
    sequence_columns = [
        column
        for column in rows.columns
        if str(column).strip().casefold() == "sequence_id"
    ]
    if not sequence_columns:
        sequence_columns = [
            column
            for column in rows.columns
            if str(column).strip().casefold() == "sequence"
        ]
    if len(sequence_columns) != 1:
        return

    values = rows[sequence_columns[0]]
    if isinstance(values, pd.DataFrame):
        return
    text = values.astype("string").str.strip()
    invalid = text.isna() | text.eq("").fillna(False)
    if invalid.any():
        row_positions = np.flatnonzero(invalid.to_numpy(dtype=bool)).tolist()[:5]
        raise ValueError(
            "submission contains missing or blank sequence_id values: "
            f"sequence_id rows {row_positions}"
        )


def _validate_numeric_rows(submission: object) -> pd.DataFrame:
    """Return normalized rows after rejecting unsafe numeric cell payloads."""

    rows = pd.DataFrame(submission).copy()
    if any(column not in rows.columns for column in _NUMERIC_COLUMNS):
        return rows

    invalid_by_kind: dict[str, list[str]] = {
        "Boolean": [],
        "complex": [],
        "masked": [],
        "non-scalar": [],
    }
    for column in _NUMERIC_COLUMNS:
        normalized_values: list[object] = []
        invalid_positions: dict[str, list[int]] = {
            kind: [] for kind in invalid_by_kind
        }
        for row_position, value in enumerate(rows[column].tolist()):
            scalar, kind = _unwrap_numeric_scalar(value)
            normalized_values.append(scalar)
            if kind is not None:
                invalid_positions[kind].append(row_position)
        rows[column] = normalized_values
        for kind, positions in invalid_positions.items():
            if positions:
                invalid_by_kind[kind].append(f"{column} rows {positions}")

    for kind in ("Boolean", "complex", "masked", "non-scalar"):
        invalid = invalid_by_kind[kind]
        if invalid:
            raise ValueError(
                f"submission contains {kind} numeric values: " + "; ".join(invalid)
            )

    nonfinite_invalid: list[str] = []
    for column in _NUMERIC_COLUMNS:
        numeric = pd.to_numeric(rows[column], errors="coerce")
        finite = np.isfinite(numeric.to_numpy(dtype=float))
        if not finite.all():
            nonfinite_invalid.append(f"{column} rows {np.flatnonzero(~finite).tolist()}")
    if nonfinite_invalid:
        raise ValueError(
            "submission contains non-finite numeric values: "
            + "; ".join(nonfinite_invalid)
        )
    return rows


def _validate_and_normalize_classifications(rows: pd.DataFrame) -> pd.DataFrame:
    """Return rows with canonical official Track 5 class IDs."""

    normalized = rows.copy()
    parsed: list[int] = []
    invalid: list[str] = []
    for row_index, value in normalized["Classification"].items():
        try:
            parsed.append(parse_official_classification_cell(value))
        except (TypeError, ValueError, OverflowError) as exc:
            invalid.append(f"{row_index}:{value!r} ({exc})")
    if invalid:
        preview = "; ".join(invalid[:5])
        suffix = f"; {len(invalid) - 5} more" if len(invalid) > 5 else ""
        raise ValueError(
            "submission contains invalid Classification values at rows "
            f"{preview}{suffix}"
        )
    normalized["Classification"] = pd.Series(
        parsed,
        index=normalized.index,
        dtype=int,
    )
    return normalized


def _validate_unique_fixed_grid_keys(rows: pd.DataFrame) -> None:
    """Reject duplicate normalized sequence/timestamp keys before repair."""

    normalized_keys = pd.DataFrame(
        {
            "sequence_id": rows["sequence_id"].astype(str),
            "time_s": pd.to_numeric(rows["time_s"], errors="coerce"),
        }
    )
    duplicate_mask = normalized_keys.duplicated(
        subset=["sequence_id", "time_s"], keep=False
    )
    if not bool(duplicate_mask.any()):
        return
    duplicate_keys = (
        normalized_keys.loc[duplicate_mask, ["sequence_id", "time_s"]]
        .drop_duplicates()
        .sort_values(["sequence_id", "time_s"])
        .reset_index(drop=True)
    )
    sample = ", ".join(
        f"{row.sequence_id}@{float(row.time_s):g}"
        for row in duplicate_keys.head(5).itertuples(index=False)
    )
    suffix = ", ..." if len(duplicate_keys) > 5 else ""
    raise ValueError(
        f"submission contains {len(duplicate_keys)} duplicate "
        f"(sequence_id, time_s) key(s): {sample}{suffix}"
    )


def repair_track5_acceleration_kinks(
    submission,
    *,
    max_acceleration_mps2: float = 20.0,
    max_direct_speed_mps: float = 80.0,
    min_interpolation_residual_m: float = 1.0,
    iterations: int = 2,
    repair_blend: float = 1.0,
):
    """Repair acceleration kinks after validating controls and fixed-grid rows."""

    controls = _validated_controls(
        max_acceleration_mps2=max_acceleration_mps2,
        max_direct_speed_mps=max_direct_speed_mps,
        min_interpolation_residual_m=min_interpolation_residual_m,
        iterations=iterations,
        repair_blend=repair_blend,
    )
    _validate_sequence_ids(submission)
    validated_submission = _validate_numeric_rows(submission)
    normalized = _IMPL._normalized_submission(validated_submission)
    normalized = _validate_and_normalize_classifications(normalized)
    _validate_unique_fixed_grid_keys(normalized)
    return _ORIGINAL_REPAIR(normalized, **controls)


def _repair_sequence(group, **kwargs):
    """Validate direct calls and keep zero-blend runs diagnostic-only."""

    controls = _validated_controls(
        max_acceleration_mps2=kwargs["max_acceleration_mps2"],
        max_direct_speed_mps=kwargs["max_direct_speed_mps"],
        min_interpolation_residual_m=kwargs["min_interpolation_residual_m"],
        iterations=kwargs["iterations"],
        repair_blend=kwargs["repair_blend"],
    )
    repaired, diagnostics = _ORIGINAL_REPAIR_SEQUENCE(group, **controls)
    if controls["repair_blend"] != 0.0:
        return repaired, diagnostics

    repaired = repaired.copy()
    diagnostics = diagnostics.copy()
    repaired["acceleration_limit_applied"] = False
    repaired["acceleration_limit_iteration"] = 0
    repaired["acceleration_limit_displacement_m"] = 0.0
    diagnostics["acceleration_limit_applied"] = False
    diagnostics["acceleration_limit_iteration"] = 0
    diagnostics["acceleration_limit_displacement_m"] = 0.0
    return repaired, diagnostics


_IMPL.repair_track5_acceleration_kinks = repair_track5_acceleration_kinks
_IMPL._repair_sequence = _repair_sequence


globals().update(
    {
        name: getattr(_IMPL, name)
        for name in dir(_IMPL)
        if not (name.startswith("__") and name.endswith("__"))
    }
)
globals()["_NUMERIC_COLUMNS"] = _NUMERIC_COLUMNS
globals()["_unwrap_numeric_scalar"] = _unwrap_numeric_scalar
globals()["_finite_scalar"] = _finite_scalar
globals()["_positive_integer"] = _positive_integer
globals()["_validated_controls"] = _validated_controls
globals()["_validate_sequence_ids"] = _validate_sequence_ids
globals()["_validate_numeric_rows"] = _validate_numeric_rows
globals()["_validate_and_normalize_classifications"] = (
    _validate_and_normalize_classifications
)
globals()["_validate_unique_fixed_grid_keys"] = _validate_unique_fixed_grid_keys
globals()["repair_track5_acceleration_kinks"] = repair_track5_acceleration_kinks
globals()["_repair_sequence"] = _repair_sequence
__doc__ = _IMPL.__doc__
__all__ = [
    name for name in dir(_IMPL) if not (name.startswith("__") and name.endswith("__"))
]
sys.modules[__name__].__class__ = _Track5AccelerationLimitModule
