"""Compatibility fixes for Track 5 template resampling."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

_IMPL_PATH = Path(__file__).resolve().parent.parent / "track5_template_resample.py"
_SPEC = importlib.util.spec_from_file_location(
    "raft_uav.mmuad._track5_template_resample_legacy",
    _IMPL_PATH,
)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"cannot load Track 5 template-resample implementation from {_IMPL_PATH}")
_IMPL = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _IMPL
_SPEC.loader.exec_module(_IMPL)

_ORIGINAL_RESAMPLE_ESTIMATES = _IMPL.resample_estimates_to_track5_template
_ORIGINAL_WRITE_OUTPUTS = _IMPL.write_track5_template_resample_outputs
_ORIGINAL_RESAMPLED_POSITION = _IMPL._resampled_position
_ORIGINAL_RESAMPLED_CLASSIFICATION = _IMPL._resampled_classification


def _normalize_optional_nonnegative_float(value: Any, *, field: str) -> float | None:
    """Return an optional finite non-negative scalar with a stable error."""

    if value is None:
        return None
    message = f"{field} must be a finite non-negative number"
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(message)
    if isinstance(value, np.ndarray):
        if value.ndim != 0:
            raise ValueError(message)
        value = value.item()
    if isinstance(value, (complex, np.complexfloating)):
        raise ValueError(message)
    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(message) from exc
    if not np.isfinite(numeric) or numeric < 0.0:
        raise ValueError(message)
    return numeric


def resample_estimates_to_track5_template(
    estimates: pd.DataFrame,
    template: pd.DataFrame,
    *,
    max_nearest_time_delta_s: float | None = None,
    resample_method="linear",
    max_interpolation_gap_s: float | None = None,
    classification_policy="sequence-mode",
):
    """Validate time controls before resampling estimates to the template."""

    max_nearest_time_delta_s = _normalize_optional_nonnegative_float(
        max_nearest_time_delta_s,
        field="max_nearest_time_delta_s",
    )
    max_interpolation_gap_s = _normalize_optional_nonnegative_float(
        max_interpolation_gap_s,
        field="max_interpolation_gap_s",
    )
    return _ORIGINAL_RESAMPLE_ESTIMATES(
        estimates,
        template,
        max_nearest_time_delta_s=max_nearest_time_delta_s,
        resample_method=resample_method,
        max_interpolation_gap_s=max_interpolation_gap_s,
        classification_policy=classification_policy,
    )


def write_track5_template_resample_outputs(
    *,
    estimates: pd.DataFrame,
    template: pd.DataFrame,
    output_dir: Path,
    class_map: dict[str, str] | None = None,
    default_classification: int | str = 0,
    max_nearest_time_delta_s: float | None = None,
    resample_method="linear",
    max_interpolation_gap_s: float | None = None,
    classification_policy="sequence-mode",
) -> dict[str, Path]:
    """Validate time controls before creating any resampling artifacts."""

    max_nearest_time_delta_s = _normalize_optional_nonnegative_float(
        max_nearest_time_delta_s,
        field="max_nearest_time_delta_s",
    )
    max_interpolation_gap_s = _normalize_optional_nonnegative_float(
        max_interpolation_gap_s,
        field="max_interpolation_gap_s",
    )
    return _ORIGINAL_WRITE_OUTPUTS(
        estimates=estimates,
        template=template,
        output_dir=output_dir,
        class_map=class_map,
        default_classification=default_classification,
        max_nearest_time_delta_s=max_nearest_time_delta_s,
        resample_method=resample_method,
        max_interpolation_gap_s=max_interpolation_gap_s,
        classification_policy=classification_policy,
    )


def _normalize_estimate_rows(estimates: pd.DataFrame) -> pd.DataFrame:
    """Normalize estimates while preserving input order among equal timestamps."""

    rows = pd.DataFrame(estimates).copy()
    if rows.empty:
        return pd.DataFrame(
            columns=["sequence_id", "time_s", "state_x_m", "state_y_m", "state_z_m"]
        )
    sequence_column = _IMPL._first_present(rows, _IMPL.SEQUENCE_ALIASES)
    time_column = _IMPL._first_present(rows, _IMPL.TIME_ALIASES)
    coord_columns = _IMPL._coordinate_columns(rows)
    classification_column = _IMPL._first_present(rows, _IMPL.CLASSIFICATION_ALIASES)
    if sequence_column is None or time_column is None:
        raise ValueError("estimates must contain sequence and time columns")
    out = pd.DataFrame(
        {
            "sequence_id": _IMPL._normalized_sequence_values(rows[sequence_column]),
            "time_s": pd.to_numeric(rows[time_column], errors="coerce"),
            "state_x_m": pd.to_numeric(rows[coord_columns[0]], errors="coerce"),
            "state_y_m": pd.to_numeric(rows[coord_columns[1]], errors="coerce"),
            "state_z_m": pd.to_numeric(rows[coord_columns[2]], errors="coerce"),
        }
    )
    if classification_column is not None:
        out["classification"] = _IMPL._normalized_classification_values(
            rows[classification_column]
        )
    finite = out["sequence_id"].notna()
    finite &= np.isfinite(
        out[["time_s", "state_x_m", "state_y_m", "state_z_m"]].to_numpy(float)
    ).all(axis=1)
    return (
        out.loc[finite]
        .sort_values(["sequence_id", "time_s"], kind="mergesort")
        .reset_index(drop=True)
    )


def _unique_time_rows(group: pd.DataFrame) -> pd.DataFrame:
    """Apply one deterministic keep-last rule to every timestamp-dependent field."""

    rows = pd.DataFrame(group).copy()
    if rows.empty or "time_s" not in rows.columns:
        return rows
    return (
        rows.sort_values("time_s", kind="mergesort")
        .drop_duplicates("time_s", keep="last")
        .reset_index(drop=True)
    )


def _resampled_position(
    group: pd.DataFrame,
    time_s: float,
    *,
    resample_method,
    max_interpolation_gap_s,
):
    return _ORIGINAL_RESAMPLED_POSITION(
        _unique_time_rows(group),
        time_s,
        resample_method=resample_method,
        max_interpolation_gap_s=max_interpolation_gap_s,
    )


def _resampled_classification(
    group: pd.DataFrame,
    time_s: float,
    *,
    classification_policy,
):
    return _ORIGINAL_RESAMPLED_CLASSIFICATION(
        _unique_time_rows(group),
        time_s,
        classification_policy=classification_policy,
    )


_IMPL.resample_estimates_to_track5_template = resample_estimates_to_track5_template
_IMPL.write_track5_template_resample_outputs = write_track5_template_resample_outputs
_IMPL._normalize_estimate_rows = _normalize_estimate_rows
_IMPL._resampled_position = _resampled_position
_IMPL._resampled_classification = _resampled_classification

globals().update(
    {
        name: getattr(_IMPL, name)
        for name in dir(_IMPL)
        if not (name.startswith("__") and name.endswith("__"))
    }
)
globals()["_normalize_optional_nonnegative_float"] = _normalize_optional_nonnegative_float
globals()["resample_estimates_to_track5_template"] = resample_estimates_to_track5_template
globals()["write_track5_template_resample_outputs"] = write_track5_template_resample_outputs
globals()["_normalize_estimate_rows"] = _normalize_estimate_rows
globals()["_resampled_position"] = _resampled_position
globals()["_resampled_classification"] = _resampled_classification

__doc__ = _IMPL.__doc__
__all__ = [
    name for name in dir(_IMPL) if not (name.startswith("__") and name.endswith("__"))
]
