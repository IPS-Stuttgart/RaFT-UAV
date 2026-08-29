"""Keep research latency diagnostics stable for finite extreme coordinates."""

from __future__ import annotations

from collections.abc import Mapping
from types import ModuleType
from typing import Any

import numpy as np
import pandas as pd

_PATCH_MARKER = "_raft_uav_research_latency_distance_stability"


def _stable_root_mean_square(values: np.ndarray) -> float:
    """Return RMS without overflowing intermediate squares."""

    array = np.asarray(values, dtype=float)
    if array.size == 0:
        return float("nan")
    scale = float(np.max(np.abs(array)))
    if scale == 0.0 or not np.isfinite(scale):
        return scale
    normalized = array / scale
    return float(scale * np.sqrt(np.mean(normalized * normalized)))


def apply_diagnostics_latency_stability_patch(module: ModuleType) -> None:
    """Patch latency RMSE so finite representable distances stay finite."""

    implementation = getattr(module, "_LEGACY", module)
    if getattr(module, _PATCH_MARKER, False):
        return

    original = module.latency_curve

    def latency_curve(
        estimates_by_latency: Mapping[float | str, pd.DataFrame],
        truth: pd.DataFrame,
        *,
        max_time_delta_s: float = 2.0,
    ) -> pd.DataFrame:
        if not estimates_by_latency:
            return original(
                estimates_by_latency,
                truth,
                max_time_delta_s=max_time_delta_s,
            )

        normalizer = getattr(module, "_nonnegative_finite_scalar", None)
        if normalizer is None:  # pragma: no cover - compatibility fallback
            normalized_gate = float(max_time_delta_s)
            if not np.isfinite(normalized_gate) or normalized_gate < 0.0:
                raise ValueError(
                    "max_time_delta_s must be a finite non-negative scalar"
                )
        else:
            normalized_gate = normalizer(
                max_time_delta_s,
                name="max_time_delta_s",
            )

        implementation._require_columns(
            truth,
            {"time_s", *implementation.PositionColumns},
            "truth",
        )
        truth_times = truth["time_s"].to_numpy(dtype=float)
        truth_xyz = truth.loc[:, implementation.PositionColumns].to_numpy(
            dtype=float
        )
        rows: list[dict[str, Any]] = []

        for latency, estimates in estimates_by_latency.items():
            if estimates.empty:
                errors = np.empty(0)
                covered = 0
            else:
                implementation._require_columns(
                    estimates,
                    {"time_s", *implementation.PositionColumns},
                    "estimates",
                )
                estimate_times = estimates["time_s"].to_numpy(dtype=float)
                estimate_xyz = estimates.loc[
                    :, implementation.PositionColumns
                ].to_numpy(dtype=float)
                nearest = implementation._nearest_time_indices(
                    estimate_times,
                    truth_times,
                )
                with np.errstate(over="ignore", invalid="ignore"):
                    dt_s = np.abs(estimate_times[nearest] - truth_times)
                keep = dt_s <= normalized_gate
                covered = int(np.count_nonzero(keep))
                with np.errstate(over="ignore", invalid="ignore"):
                    residuals = (
                        estimate_xyz[nearest][keep] - truth_xyz[keep]
                    )
                    errors = np.hypot.reduce(np.abs(residuals), axis=1)

            rows.append(
                {
                    "latency_s": float(latency),
                    "truth_rows": int(len(truth)),
                    "covered_truth_rows": covered,
                    "truth_coverage_rate": (
                        float(covered / len(truth))
                        if len(truth)
                        else float("nan")
                    ),
                    "error_3d_count": int(errors.size),
                    "error_3d_rmse_m": _stable_root_mean_square(errors),
                    "error_3d_p95_m": (
                        float(np.percentile(errors, 95))
                        if errors.size
                        else float("nan")
                    ),
                }
            )

        return (
            pd.DataFrame.from_records(rows)
            .sort_values("latency_s")
            .reset_index(drop=True)
        )

    setattr(latency_curve, _PATCH_MARKER, True)
    module.latency_curve = latency_curve
    implementation.latency_curve = latency_curve
    setattr(module, _PATCH_MARKER, True)
    setattr(implementation, _PATCH_MARKER, True)
