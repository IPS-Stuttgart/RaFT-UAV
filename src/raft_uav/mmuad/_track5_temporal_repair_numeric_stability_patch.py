"""Keep Track 5 temporal-repair arithmetic stable for finite coordinates."""

from __future__ import annotations

import math
from functools import wraps
from typing import Any, Callable

import numpy as np
import pandas as pd

_FLOAT_MAX = float(np.finfo(float).max)
_PATCH_MARKER = "_raft_uav_track5_temporal_repair_numeric_stability"


def _positive_product_ratio(
    numerator_a: float,
    numerator_b: float,
    denominator_a: float,
    denominator_b: float = 1.0,
) -> float:
    """Return ``(a * b) / (c * d)`` without overflowing intermediates."""

    numerator_a_mantissa, numerator_a_exponent = math.frexp(float(numerator_a))
    numerator_b_mantissa, numerator_b_exponent = math.frexp(float(numerator_b))
    denominator_a_mantissa, denominator_a_exponent = math.frexp(
        float(denominator_a)
    )
    denominator_b_mantissa, denominator_b_exponent = math.frexp(
        float(denominator_b)
    )
    mantissa = (numerator_a_mantissa * numerator_b_mantissa) / (
        denominator_a_mantissa * denominator_b_mantissa
    )
    exponent = (
        numerator_a_exponent
        + numerator_b_exponent
        - denominator_a_exponent
        - denominator_b_exponent
    )
    try:
        return math.ldexp(mantissa, exponent)
    except OverflowError:
        return math.inf


def _scaled_displacement(
    point: np.ndarray,
    anchor: np.ndarray,
) -> tuple[float, float]:
    """Return a shared coordinate scale and displacement norm in that scale."""

    point_array = np.asarray(point, dtype=float)
    anchor_array = np.asarray(anchor, dtype=float)
    scale = float(max(np.max(np.abs(point_array)), np.max(np.abs(anchor_array))))
    if scale <= 0.0:
        return 0.0, 0.0
    with np.errstate(over="ignore", invalid="ignore"):
        scaled_delta = point_array / scale - anchor_array / scale
        scaled_distance = float(np.hypot.reduce(np.abs(scaled_delta).reshape(-1)))
    return scale, scaled_distance


def _stable_distance(point: np.ndarray, anchor: np.ndarray) -> float:
    """Return a finite Euclidean distance for finite coordinate vectors."""

    point_array = np.asarray(point, dtype=float)
    anchor_array = np.asarray(anchor, dtype=float)
    with np.errstate(over="ignore", invalid="ignore"):
        distance = float(np.linalg.norm(point_array - anchor_array))
    if np.isfinite(distance):
        return distance
    if not (
        bool(np.isfinite(point_array).all())
        and bool(np.isfinite(anchor_array).all())
    ):
        return distance
    scale, scaled_distance = _scaled_displacement(point_array, anchor_array)
    if scale <= 0.0 or scaled_distance <= 0.0:
        return 0.0
    repaired = _positive_product_ratio(scale, scaled_distance, 1.0)
    return min(repaired, _FLOAT_MAX)


def _stable_speed(
    point: np.ndarray,
    anchor: np.ndarray,
    dt_s: float,
) -> float:
    """Return a finite displacement rate without overflowing before division."""

    point_array = np.asarray(point, dtype=float)
    anchor_array = np.asarray(anchor, dtype=float)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        displacement = point_array - anchor_array
        distance = float(np.linalg.norm(displacement))
        speed = float(distance / dt_s)
    if (
        bool(np.isfinite(displacement).all())
        and np.isfinite(distance)
        and np.isfinite(speed)
    ):
        return speed
    if not (
        bool(np.isfinite(point_array).all())
        and bool(np.isfinite(anchor_array).all())
        and np.isfinite(dt_s)
        and dt_s > 0.0
    ):
        return speed
    scale, scaled_distance = _scaled_displacement(point_array, anchor_array)
    if scale <= 0.0 or scaled_distance <= 0.0:
        return 0.0
    repaired = _positive_product_ratio(scale, scaled_distance, dt_s)
    return min(repaired, _FLOAT_MAX)


def _stable_interpolation(
    left: np.ndarray,
    right: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """Preserve ordinary interpolation and repair overflowed finite endpoints."""

    left_array = np.asarray(left, dtype=float)
    right_array = np.asarray(right, dtype=float)
    with np.errstate(over="ignore", invalid="ignore"):
        direct = left_array + float(alpha) * (right_array - left_array)
    if bool(np.isfinite(direct).all()):
        return direct
    if not (
        bool(np.isfinite(left_array).all())
        and bool(np.isfinite(right_array).all())
        and np.isfinite(alpha)
    ):
        return direct
    with np.errstate(over="ignore", invalid="ignore"):
        convex = (1.0 - float(alpha)) * left_array + float(alpha) * right_array
    if bool(np.isfinite(convex).all()):
        return convex
    scale = float(max(np.max(np.abs(left_array)), np.max(np.abs(right_array))))
    if scale <= 0.0:
        return np.zeros_like(left_array)
    with np.errstate(over="ignore", invalid="ignore"):
        scaled = (
            (1.0 - float(alpha)) * (left_array / scale)
            + float(alpha) * (right_array / scale)
        )
        repaired = scale * scaled
    return repaired


def _sequence_diagnostics(group: pd.DataFrame, *, iteration: int) -> pd.DataFrame:
    """Build temporal-repair diagnostics without losing finite extreme rows."""

    n = len(group)
    times = group["time_s"].to_numpy(float)
    xyz = group[["state_x_m", "state_y_m", "state_z_m"]].to_numpy(float)
    records: list[dict[str, Any]] = []
    for i in range(n):
        record = {
            "sequence_id": str(group.loc[i, "sequence_id"]),
            "time_s": float(times[i]),
            "iteration": int(iteration),
            "incoming_speed_mps": np.nan,
            "outgoing_speed_mps": np.nan,
            "neighbor_direct_speed_mps": np.nan,
            "interpolation_residual_m": np.nan,
            "interp_x_m": xyz[i, 0],
            "interp_y_m": xyz[i, 1],
            "interp_z_m": xyz[i, 2],
            "repair_candidate": False,
        }
        if 0 < i < n - 1:
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                dt_in = float(times[i] - times[i - 1])
                dt_out = float(times[i + 1] - times[i])
                dt_direct = float(times[i + 1] - times[i - 1])
            if (
                np.isfinite(dt_in)
                and np.isfinite(dt_out)
                and np.isfinite(dt_direct)
                and dt_in > 0.0
                and dt_out > 0.0
                and dt_direct > 0.0
            ):
                alpha = dt_in / dt_direct
                interpolated = _stable_interpolation(
                    xyz[i - 1],
                    xyz[i + 1],
                    alpha,
                )
                record.update(
                    {
                        "incoming_speed_mps": _stable_speed(
                            xyz[i],
                            xyz[i - 1],
                            dt_in,
                        ),
                        "outgoing_speed_mps": _stable_speed(
                            xyz[i + 1],
                            xyz[i],
                            dt_out,
                        ),
                        "neighbor_direct_speed_mps": _stable_speed(
                            xyz[i + 1],
                            xyz[i - 1],
                            dt_direct,
                        ),
                        "interpolation_residual_m": _stable_distance(
                            xyz[i],
                            interpolated,
                        ),
                        "interp_x_m": float(interpolated[0]),
                        "interp_y_m": float(interpolated[1]),
                        "interp_z_m": float(interpolated[2]),
                    }
                )
        records.append(record)
    return pd.DataFrame.from_records(records)


class _StableLinalgProxy:
    """Delegate linalg calls while repairing finite-input norm overflow."""

    def __init__(self, original: Any) -> None:
        self._original = original

    def __getattr__(self, name: str) -> Any:
        return getattr(self._original, name)

    def norm(
        self,
        values: Any,
        ord: Any = None,
        axis: Any = None,
        keepdims: bool = False,
    ) -> Any:
        """Preserve ordinary norms and repair one-vector finite overflow."""

        with np.errstate(over="ignore", invalid="ignore"):
            direct = self._original.norm(
                values,
                ord=ord,
                axis=axis,
                keepdims=keepdims,
            )
        if ord is not None or axis is not None:
            return direct
        array = np.asarray(values, dtype=float)
        if bool(np.isfinite(np.asarray(direct, dtype=float)).all()):
            return direct
        if array.ndim != 1 or not bool(np.isfinite(array).all()):
            return direct
        repaired = _stable_distance(array, np.zeros_like(array))
        if keepdims:
            return np.asarray(repaired).reshape((1,) * array.ndim)
        return repaired


class _StableNumpyProxy:
    """Proxy NumPy only inside the legacy temporal-repair module."""

    def __init__(self, original: Any) -> None:
        self._original = original
        self.linalg = _StableLinalgProxy(original.linalg)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._original, name)


def _guard_numeric_arithmetic(function: Callable[..., Any]) -> Callable[..., Any]:
    """Prevent caller error-state policy from escaping legacy repair arithmetic."""

    @wraps(function)
    def guarded(*args: Any, **kwargs: Any) -> Any:
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            return function(*args, **kwargs)

    return guarded


def install(implementation: Any) -> None:
    """Install finite-extreme guards on the legacy temporal-repair module."""

    if getattr(implementation, _PATCH_MARKER, False):
        return
    original_numpy = implementation.np
    implementation.np = _StableNumpyProxy(original_numpy)
    implementation._sequence_diagnostics = _sequence_diagnostics
    implementation.repair_track5_temporal_spikes = _guard_numeric_arithmetic(
        implementation.repair_track5_temporal_spikes
    )
    setattr(implementation, _PATCH_MARKER, True)
