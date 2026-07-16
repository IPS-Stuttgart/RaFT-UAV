"""Compatibility fixes for Track 5 RTS ensemble inputs and identifiers.

The maintained implementation lives in the sibling ``track5_rts_ensemble.py``
module. This package keeps the public import path while preserving opaque
sequence identifiers, canonicalizing template identifiers, and validating
numeric controls before empty-template returns or estimate-file access.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from typing import Any, Iterable

import numpy as np
import pandas as pd

from raft_uav.mmuad.estimate_csv import read_estimate_csv
from raft_uav.mmuad.submission import parse_official_sequence_cell

_IMPL_PATH = Path(__file__).resolve().parent.parent / "track5_rts_ensemble.py"
_SPEC = importlib.util.spec_from_file_location(
    "raft_uav.mmuad._track5_rts_ensemble_legacy",
    _IMPL_PATH,
)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"cannot load Track 5 RTS ensemble implementation from {_IMPL_PATH}")
_IMPL = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _IMPL
_SPEC.loader.exec_module(_IMPL)

EstimateInput = _IMPL.EstimateInput
_LEGACY_BUILD = _IMPL.build_track5_rts_ensemble
_LEGACY_WRITE = _IMPL.write_track5_rts_ensemble_outputs


class _PandasCsvProxy:
    """Delegate pandas operations while guarding plain estimate CSV reads."""

    def __init__(self, module: Any) -> None:
        self._module = module

    def __getattr__(self, name: str) -> Any:
        return getattr(self._module, name)

    def read_csv(self, path: Any, *args: Any, **kwargs: Any) -> pd.DataFrame:
        if args or kwargs:
            rows = self._module.read_csv(path, *args, **kwargs)
            out = rows.copy()
            out.columns = [str(column).strip() for column in out.columns]
            return out
        return read_estimate_csv(Path(path))


def _first_present(rows: pd.DataFrame, names: tuple[str, ...]) -> Any | None:
    """Return a column whose stripped, case-folded name matches an alias."""

    normalized = {str(column).strip().casefold(): column for column in rows.columns}
    for name in names:
        if name in rows.columns:
            return name
        found = normalized.get(str(name).strip().casefold())
        if found is not None:
            return found
    return None


def _sequence_text_or_none(value: Any) -> str | None:
    try:
        return parse_official_sequence_cell(value)
    except ValueError:
        return None


def _normalize_template_rows(template: pd.DataFrame) -> pd.DataFrame:
    rows = pd.DataFrame(template).copy()
    if rows.empty:
        return pd.DataFrame(columns=["sequence_id", "time_s"])
    sequence_column = _first_present(
        rows,
        ("sequence_id", "Sequence", "sequence", "seq"),
    )
    time_column = _first_present(
        rows,
        ("time_s", "Timestamp", "timestamp", "timestamp_s", "time"),
    )
    if sequence_column is None or time_column is None:
        raise ValueError("template must contain sequence and timestamp columns")
    out = pd.DataFrame(
        {
            "sequence_id": rows[sequence_column].map(_sequence_text_or_none),
            "time_s": pd.to_numeric(rows[time_column], errors="coerce"),
        }
    )
    finite = out["sequence_id"].notna() & np.isfinite(out["time_s"].to_numpy(float))
    return out.loc[finite].sort_values(["sequence_id", "time_s"]).reset_index(drop=True)


def _normalize_finite_scalar(
    value: Any,
    *,
    name: str,
    allow_zero: bool,
) -> float:
    requirement = "non-negative" if allow_zero else "positive"
    message = f"{name} must be {requirement} and finite"
    if isinstance(value, np.ndarray):
        if value.ndim != 0:
            raise ValueError(message)
        value = value.item()
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(message)
    if isinstance(value, (complex, np.complexfloating)):
        raise ValueError(message)
    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(message) from exc
    if not np.isfinite(numeric) or numeric < 0.0 or (not allow_zero and numeric == 0.0):
        raise ValueError(message)
    return numeric


def _positive_finite(value: Any, name: str) -> float:
    """Return a finite positive scalar without accepting Boolean pseudo-numbers."""

    return _normalize_finite_scalar(value, name=name, allow_zero=False)


def _nonnegative_finite(value: Any, name: str) -> float:
    """Return a finite non-negative scalar without Boolean coercion."""

    return _normalize_finite_scalar(value, name=name, allow_zero=True)


def _optional_nonnegative_finite(value: Any, name: str) -> float | None:
    if value is None:
        return None
    return _nonnegative_finite(value, name)


def _validated_runtime_inputs(
    estimate_inputs: Iterable[tuple[str, pd.DataFrame, Any]],
) -> list[tuple[str, pd.DataFrame, float]]:
    validated: list[tuple[str, pd.DataFrame, float]] = []
    for label, estimates, weight in estimate_inputs:
        safe_label = _IMPL._safe_label(label)
        validated.append(
            (
                label,
                estimates,
                _positive_finite(weight, f"weight[{safe_label}]"),
            )
        )
    return validated


def _validated_estimate_inputs(
    estimate_inputs: Iterable[EstimateInput],
) -> list[EstimateInput]:
    validated: list[EstimateInput] = []
    for item in estimate_inputs:
        safe_label = _IMPL._safe_label(item.label)
        validated.append(
            EstimateInput(
                label=item.label,
                path=item.path,
                weight=_positive_finite(item.weight, f"weight[{safe_label}]"),
            )
        )
    return validated


def build_track5_rts_ensemble(
    estimate_inputs: Iterable[tuple[str, pd.DataFrame, Any]],
    template: pd.DataFrame,
    *,
    measurement_sigma_m: float = 10.0,
    process_accel_std_mps2: float = 5.0,
    initial_position_std_m: float = 100.0,
    initial_velocity_std_mps: float = 25.0,
    spread_variance_scale: float = 1.0,
    max_nearest_time_delta_s: float | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build an RTS ensemble after validating every scalar and input weight."""

    measurement_sigma_m = _positive_finite(measurement_sigma_m, "measurement_sigma_m")
    process_accel_std_mps2 = _nonnegative_finite(
        process_accel_std_mps2,
        "process_accel_std_mps2",
    )
    initial_position_std_m = _positive_finite(
        initial_position_std_m,
        "initial_position_std_m",
    )
    initial_velocity_std_mps = _positive_finite(
        initial_velocity_std_mps,
        "initial_velocity_std_mps",
    )
    spread_variance_scale = _nonnegative_finite(
        spread_variance_scale,
        "spread_variance_scale",
    )
    max_nearest_time_delta_s = _optional_nonnegative_finite(
        max_nearest_time_delta_s,
        "max_nearest_time_delta_s",
    )
    inputs = _validated_runtime_inputs(estimate_inputs)
    return _LEGACY_BUILD(
        inputs,
        template,
        measurement_sigma_m=measurement_sigma_m,
        process_accel_std_mps2=process_accel_std_mps2,
        initial_position_std_m=initial_position_std_m,
        initial_velocity_std_mps=initial_velocity_std_mps,
        spread_variance_scale=spread_variance_scale,
        max_nearest_time_delta_s=max_nearest_time_delta_s,
    )


def write_track5_rts_ensemble_outputs(
    *,
    estimate_inputs: Iterable[EstimateInput],
    template: pd.DataFrame,
    output_dir: Path,
    class_map: dict[str, str] | None = None,
    default_classification: int | str = 0,
    measurement_sigma_m: float = 10.0,
    process_accel_std_mps2: float = 5.0,
    initial_position_std_m: float = 100.0,
    initial_velocity_std_mps: float = 25.0,
    spread_variance_scale: float = 1.0,
    max_nearest_time_delta_s: float | None = None,
) -> dict[str, Path]:
    """Validate RTS controls and weights before estimate-file or output I/O."""

    measurement_sigma_m = _positive_finite(measurement_sigma_m, "measurement_sigma_m")
    process_accel_std_mps2 = _nonnegative_finite(
        process_accel_std_mps2,
        "process_accel_std_mps2",
    )
    initial_position_std_m = _positive_finite(
        initial_position_std_m,
        "initial_position_std_m",
    )
    initial_velocity_std_mps = _positive_finite(
        initial_velocity_std_mps,
        "initial_velocity_std_mps",
    )
    spread_variance_scale = _nonnegative_finite(
        spread_variance_scale,
        "spread_variance_scale",
    )
    max_nearest_time_delta_s = _optional_nonnegative_finite(
        max_nearest_time_delta_s,
        "max_nearest_time_delta_s",
    )
    inputs = _validated_estimate_inputs(estimate_inputs)
    return _LEGACY_WRITE(
        estimate_inputs=inputs,
        template=template,
        output_dir=output_dir,
        class_map=class_map,
        default_classification=default_classification,
        measurement_sigma_m=measurement_sigma_m,
        process_accel_std_mps2=process_accel_std_mps2,
        initial_position_std_m=initial_position_std_m,
        initial_velocity_std_mps=initial_velocity_std_mps,
        spread_variance_scale=spread_variance_scale,
        max_nearest_time_delta_s=max_nearest_time_delta_s,
    )


_IMPL.pd = _PandasCsvProxy(pd)
_IMPL._first_present = _first_present
_IMPL._normalize_template_rows = _normalize_template_rows
_IMPL._positive_finite = _positive_finite
_IMPL._nonnegative_finite = _nonnegative_finite
_IMPL.build_track5_rts_ensemble = build_track5_rts_ensemble
_IMPL.write_track5_rts_ensemble_outputs = write_track5_rts_ensemble_outputs

globals().update(
    {
        name: getattr(_IMPL, name)
        for name in dir(_IMPL)
        if not (name.startswith("__") and name.endswith("__"))
    }
)

# Keep patched helpers and public functions importable after re-exporting the legacy module.
globals()["_first_present"] = _first_present
globals()["_sequence_text_or_none"] = _sequence_text_or_none
globals()["_normalize_template_rows"] = _normalize_template_rows
globals()["_normalize_finite_scalar"] = _normalize_finite_scalar
globals()["_positive_finite"] = _positive_finite
globals()["_nonnegative_finite"] = _nonnegative_finite
globals()["_optional_nonnegative_finite"] = _optional_nonnegative_finite
globals()["_validated_runtime_inputs"] = _validated_runtime_inputs
globals()["_validated_estimate_inputs"] = _validated_estimate_inputs
globals()["build_track5_rts_ensemble"] = build_track5_rts_ensemble
globals()["write_track5_rts_ensemble_outputs"] = write_track5_rts_ensemble_outputs

__doc__ = _IMPL.__doc__
__all__ = [name for name in dir(_IMPL) if not (name.startswith("__") and name.endswith("__"))]
