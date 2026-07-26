"""Compatibility fixes for robust paper-table diagnostics.

The maintained implementation lives in the sibling ``paper_table.py`` module.
This package preserves the public import path while excluding malformed radar
anchors before interpolation, reporting invalid reference counts as failed
checks instead of raising conversion errors, and validating stable-segment
controls before they can silently distort paper metrics.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

_IMPL_PATH = Path(__file__).resolve().parent.parent / "paper_table.py"
_SPEC = importlib.util.spec_from_file_location(
    "raft_uav.diagnostics._paper_table_legacy",
    _IMPL_PATH,
)
if _SPEC is None or _SPEC.loader is None:  # pragma: no cover
    raise ImportError(f"cannot load paper-table implementation from {_IMPL_PATH}")
_IMPL = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _IMPL
_SPEC.loader.exec_module(_IMPL)

_ORIGINAL_INTERPOLATE_SELECTED_RADAR = (
    _IMPL._interpolate_selected_radar_to_frame_times
)
_ORIGINAL_RUN_PAPER_TABLE_DIAGNOSTIC = _IMPL.run_paper_table_diagnostic
_POSITION_COLUMNS = ("east_m", "north_m", "up_m")


def _finite_interpolation_anchors(selected: pd.DataFrame) -> pd.DataFrame:
    """Return anchors with finite numeric timestamps and complete 3D positions."""

    required = ("time_s", *_POSITION_COLUMNS)
    if any(column not in selected.columns for column in required):
        return selected

    anchors = selected.copy()
    for column in required:
        anchors[column] = pd.to_numeric(anchors[column], errors="coerce")
    finite = np.isfinite(
        anchors.loc[:, list(required)].to_numpy(dtype=float)
    ).all(axis=1)
    return anchors.loc[finite].copy()


def _interpolate_selected_radar_to_frame_times(
    radar: pd.DataFrame,
    selected: pd.DataFrame,
    *,
    association_mode: str,
    max_gap_s: float | None = None,
    max_speed_mps: float | None = None,
) -> pd.DataFrame:
    """Interpolate from usable anchors without letting one bad row erase output."""

    return _ORIGINAL_INTERPOLATE_SELECTED_RADAR(
        radar,
        _finite_interpolation_anchors(selected),
        association_mode=association_mode,
        max_gap_s=max_gap_s,
        max_speed_mps=max_speed_mps,
    )


def paper_reference_count_check(
    table: pd.DataFrame,
    *,
    tolerance: int = 0,
) -> dict[str, object]:
    """Compare reference counts while treating invalid values as failed checks."""

    tolerance_value = int(tolerance)
    rows: list[dict[str, object]] = []
    passed = True
    for method, column, expected in _IMPL.PAPER_REFERENCE_COUNT_CHECKS:
        match = (
            table.loc[table.get("method") == method]
            if "method" in table
            else pd.DataFrame()
        )
        if match.empty or column not in match.columns:
            rows.append(
                {
                    "method": method,
                    "column": column,
                    "expected": int(expected),
                    "actual": None,
                    "delta": None,
                    "passed": False,
                }
            )
            passed = False
            continue

        try:
            numeric = float(pd.to_numeric(match.iloc[0][column], errors="coerce"))
        except (TypeError, ValueError):
            numeric = float("nan")
        if not np.isfinite(numeric):
            rows.append(
                {
                    "method": method,
                    "column": column,
                    "expected": int(expected),
                    "actual": None,
                    "delta": None,
                    "passed": False,
                }
            )
            passed = False
            continue

        actual = int(numeric)
        delta = actual - int(expected)
        ok = abs(delta) <= tolerance_value
        rows.append(
            {
                "method": method,
                "column": column,
                "expected": int(expected),
                "actual": actual,
                "delta": int(delta),
                "passed": bool(ok),
            }
        )
        passed &= bool(ok)

    message = (
        "paper reference counts matched"
        if passed
        else f"paper reference count mismatch: {rows}"
    )
    return {
        "passed": bool(passed),
        "tolerance": tolerance_value,
        "checks": rows,
        "message": message,
    }


def _validated_positive_integer(value: Any, *, name: str) -> int:
    """Return one finite positive integer scalar, excluding Booleans."""

    message = f"{name} must be a positive integer scalar"
    if isinstance(value, (bool, np.bool_)) or np.ma.is_masked(value):
        raise ValueError(message)
    try:
        scalar = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(message) from exc
    if scalar.ndim != 0 or np.iscomplexobj(scalar) or scalar.dtype.kind == "b":
        raise ValueError(message)
    try:
        number = float(scalar.item())
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(message) from exc
    if not np.isfinite(number) or number < 1.0 or not number.is_integer():
        raise ValueError(message)
    return int(number)


def _validated_positive_real(value: Any, *, name: str) -> float:
    """Return one finite positive real scalar, excluding Booleans."""

    message = f"{name} must be a finite positive real scalar"
    if isinstance(value, (bool, np.bool_)) or np.ma.is_masked(value):
        raise ValueError(message)
    try:
        scalar = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(message) from exc
    if scalar.ndim != 0 or np.iscomplexobj(scalar) or scalar.dtype.kind == "b":
        raise ValueError(message)
    try:
        number = float(scalar.item())
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(message) from exc
    if not np.isfinite(number) or number <= 0.0:
        raise ValueError(message)
    return number


def run_paper_table_diagnostic(
    *,
    dataset_root: Path,
    flight_name: str,
    output_dir: Path = Path("outputs/paper-table"),
    radar_catprob_threshold: float = 0.4,
    radar_range_gate_m: float | None = 800.0,
    radar_interpolation_max_gap_s: float | None = None,
    radar_interpolation_max_speed_mps: float | None = None,
    stable_segment_min_frames: int = 100,
    stable_segment_max_transition_speed_mps: float = 65.0,
    empirical_covariance: bool = False,
    empirical_covariance_min_variance_m2: float = 1.0,
    assert_reference_counts: bool = False,
    reference_count_tolerance: int = 0,
    enu_origin_lla: tuple[float, float, float] | None = None,
    radar_selections: tuple[str, ...] = _IMPL.RADAR_SELECTIONS,
    fusion_nis_gate_prob: float = _IMPL.PAPER_NIS_GATE_PROBABILITY,
    rf_nis_gate_prob: float = _IMPL.PAPER_NIS_GATE_PROBABILITY,
    truth_time_gate_s: float = 2.0,
    acceleration_std_mps2: float = 4.0,
    smoother_lag_s: float = 20.0,
    include_smoothed_fusion: bool = False,
    include_fusion: bool = True,
    disable_radar_catprob_threshold: bool = False,
    fusion_associations: tuple[str, ...] = _IMPL.FUSION_ASSOCIATIONS,
) -> dict[str, Any]:
    """Build a paper table after validating stable-segment controls."""

    min_frames = _validated_positive_integer(
        stable_segment_min_frames,
        name="stable_segment_min_frames",
    )
    max_transition_speed_mps = _validated_positive_real(
        stable_segment_max_transition_speed_mps,
        name="stable_segment_max_transition_speed_mps",
    )
    return _ORIGINAL_RUN_PAPER_TABLE_DIAGNOSTIC(
        dataset_root=dataset_root,
        flight_name=flight_name,
        output_dir=output_dir,
        radar_catprob_threshold=radar_catprob_threshold,
        radar_range_gate_m=radar_range_gate_m,
        radar_interpolation_max_gap_s=radar_interpolation_max_gap_s,
        radar_interpolation_max_speed_mps=radar_interpolation_max_speed_mps,
        stable_segment_min_frames=min_frames,
        stable_segment_max_transition_speed_mps=max_transition_speed_mps,
        empirical_covariance=empirical_covariance,
        empirical_covariance_min_variance_m2=empirical_covariance_min_variance_m2,
        assert_reference_counts=assert_reference_counts,
        reference_count_tolerance=reference_count_tolerance,
        enu_origin_lla=enu_origin_lla,
        radar_selections=radar_selections,
        fusion_nis_gate_prob=fusion_nis_gate_prob,
        rf_nis_gate_prob=rf_nis_gate_prob,
        truth_time_gate_s=truth_time_gate_s,
        acceleration_std_mps2=acceleration_std_mps2,
        smoother_lag_s=smoother_lag_s,
        include_smoothed_fusion=include_smoothed_fusion,
        include_fusion=include_fusion,
        disable_radar_catprob_threshold=disable_radar_catprob_threshold,
        fusion_associations=fusion_associations,
    )


_IMPL._finite_interpolation_anchors = _finite_interpolation_anchors
_IMPL._interpolate_selected_radar_to_frame_times = (
    _interpolate_selected_radar_to_frame_times
)
_IMPL.paper_reference_count_check = paper_reference_count_check
_IMPL._validated_positive_integer = _validated_positive_integer
_IMPL._validated_positive_real = _validated_positive_real
_IMPL.run_paper_table_diagnostic = run_paper_table_diagnostic

globals().update(
    {
        name: getattr(_IMPL, name)
        for name in dir(_IMPL)
        if not (name.startswith("__") and name.endswith("__"))
    }
)
globals()["_finite_interpolation_anchors"] = _finite_interpolation_anchors
globals()["_interpolate_selected_radar_to_frame_times"] = (
    _interpolate_selected_radar_to_frame_times
)
globals()["paper_reference_count_check"] = paper_reference_count_check
globals()["_validated_positive_integer"] = _validated_positive_integer
globals()["_validated_positive_real"] = _validated_positive_real
globals()["run_paper_table_diagnostic"] = run_paper_table_diagnostic

__doc__ = _IMPL.__doc__
__all__ = [
    name for name in dir(_IMPL) if not (name.startswith("__") and name.endswith("__"))
]
