"""Compatibility fixes for the Track 5 geometric-median ensemble.

The maintained implementation lives in the sibling
``track5_geometric_median_ensemble.py`` module. This package preserves the
public import path while using the modified Weiszfeld update when an iterate
coincides with an input point and validating ensemble inputs before early
returns, file access, or artifact creation.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from typing import Iterable

import numpy as np
import pandas as pd

from raft_uav.numeric import optional_float, optional_int

_IMPL_PATH = Path(__file__).resolve().parent.parent / "track5_geometric_median_ensemble.py"
_SPEC = importlib.util.spec_from_file_location(
    "raft_uav.mmuad._track5_geometric_median_ensemble_legacy",
    _IMPL_PATH,
)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(
        "cannot load Track 5 geometric-median implementation "
        f"from {_IMPL_PATH}"
    )
_IMPL = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _IMPL
_SPEC.loader.exec_module(_IMPL)

EstimateInput = _IMPL.EstimateInput
_ORIGINAL_BUILD_TRACK5_GEOMETRIC_MEDIAN_ENSEMBLE = (
    _IMPL.build_track5_geometric_median_ensemble
)
_ORIGINAL_WRITE_TRACK5_GEOMETRIC_MEDIAN_OUTPUTS = (
    _IMPL.write_track5_geometric_median_outputs
)


def _validate_solver_controls(
    max_iterations: object,
    tolerance_m: object,
) -> tuple[int, float]:
    iteration_limit = optional_int(max_iterations)
    if iteration_limit is None or iteration_limit < 1:
        raise ValueError("max_iterations must be a positive integer")

    convergence_tolerance_m = optional_float(tolerance_m)
    if convergence_tolerance_m is None or convergence_tolerance_m < 0.0:
        raise ValueError("tolerance_m must be a finite non-negative scalar")
    return iteration_limit, convergence_tolerance_m


def _validate_estimate_weight(value: object, *, label: str) -> float:
    """Return a finite non-negative scalar estimate weight."""

    weight = optional_float(value)
    if weight is None or weight < 0.0:
        raise ValueError(
            "estimate weight must be finite and non-negative "
            f"for {label}: {value!r}"
        )
    return weight


def _validated_runtime_inputs(
    estimate_inputs: Iterable[tuple[str, pd.DataFrame, object]],
) -> list[tuple[str, pd.DataFrame, float]]:
    """Materialize runtime inputs after validating every weight."""

    validated: list[tuple[str, pd.DataFrame, float]] = []
    for label, estimates, weight in estimate_inputs:
        safe_label = _IMPL._safe_label(label)
        validated.append(
            (
                label,
                estimates,
                _validate_estimate_weight(weight, label=safe_label),
            )
        )
    return validated


def _validated_estimate_input_objects(
    estimate_inputs: Iterable[EstimateInput],
) -> list[EstimateInput]:
    """Materialize writer inputs after validating every weight."""

    validated: list[EstimateInput] = []
    for item in estimate_inputs:
        safe_label = _IMPL._safe_label(item.label)
        validated.append(
            EstimateInput(
                label=item.label,
                path=item.path,
                weight=_validate_estimate_weight(item.weight, label=safe_label),
            )
        )
    return validated


def _validated_solver_samples(
    xyz: object,
    weights: object,
) -> tuple[np.ndarray, np.ndarray]:
    """Return solver samples with deterministic shape and count validation."""

    try:
        points = np.asarray(xyz, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            "xyz must be a numeric 2D array with shape (n, 3)"
        ) from exc
    try:
        point_weights = np.asarray(weights, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("weights must be a numeric 1D array") from exc

    # A plain empty sequence is the natural representation for no candidates and
    # should reach the solver's documented empty-result branch.
    if points.ndim == 1 and points.size == 0:
        points = np.empty((0, 3), dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("xyz must be a numeric 2D array with shape (n, 3)")
    if point_weights.ndim != 1:
        raise ValueError("weights must be a numeric 1D array")
    if point_weights.shape[0] != points.shape[0]:
        raise ValueError("xyz and weights must have the same row count")
    return points, point_weights


def weighted_geometric_median(
    xyz: np.ndarray,
    weights: np.ndarray,
    *,
    max_iterations: int = 64,
    tolerance_m: float = 1.0e-4,
) -> tuple[np.ndarray, int, float]:
    """Compute a weighted geometric median with a singularity-safe update.

    Plain Weiszfeld iterations are undefined when the current iterate coincides
    with an input point. Replacing the zero distance by a small epsilon can make
    that point dominate the next update even when it is not a minimizer. The
    modified update checks the geometric-median subgradient condition and, when
    necessary, moves away from the coincident point.
    """

    iteration_limit, convergence_tolerance_m = _validate_solver_controls(
        max_iterations,
        tolerance_m,
    )
    points, point_weights = _validated_solver_samples(xyz, weights)
    finite = (
        np.isfinite(points).all(axis=1)
        & np.isfinite(point_weights)
        & (point_weights > 0.0)
    )
    points = points[finite]
    point_weights = point_weights[finite]
    if len(points) == 0:
        return np.asarray([np.nan, np.nan, np.nan], dtype=float), 0, np.nan
    if len(points) == 1:
        return points[0].astype(float), 0, 0.0

    center = np.sum(point_weights[:, None] * points, axis=0) / float(
        np.sum(point_weights)
    )
    last_displacement = np.inf
    coincidence_tolerance_m = 1.0e-12
    optimality_slack = np.finfo(float).eps * max(
        1.0,
        float(np.sum(point_weights)),
    )

    for iteration in range(1, iteration_limit + 1):
        offsets = points - center[None, :]
        distances = np.linalg.norm(offsets, axis=1)
        coincident = distances <= coincidence_tolerance_m

        if coincident.any():
            coincident_weight = float(np.sum(point_weights[coincident]))
            noncoincident = ~coincident
            if not noncoincident.any():
                return center.astype(float), iteration, 0.0

            residual = np.sum(
                point_weights[noncoincident, None]
                * offsets[noncoincident]
                / distances[noncoincident, None],
                axis=0,
            )
            residual_norm = float(np.linalg.norm(residual))
            if residual_norm <= coincident_weight + optimality_slack:
                return center.astype(float), iteration, 0.0

            inverse_distance_weights = (
                point_weights[noncoincident] / distances[noncoincident]
            )
            weiszfeld_center = np.sum(
                inverse_distance_weights[:, None] * points[noncoincident],
                axis=0,
            ) / float(np.sum(inverse_distance_weights))
            interpolation = coincident_weight / residual_norm
            updated = (
                interpolation * center
                + (1.0 - interpolation) * weiszfeld_center
            )
        else:
            inverse_distance_weights = point_weights / distances
            updated = np.sum(
                inverse_distance_weights[:, None] * points,
                axis=0,
            ) / float(np.sum(inverse_distance_weights))

        last_displacement = float(np.linalg.norm(updated - center))
        center = updated
        if last_displacement <= convergence_tolerance_m:
            return center.astype(float), iteration, last_displacement

    return center.astype(float), iteration_limit, last_displacement


def build_track5_geometric_median_ensemble(
    estimate_inputs: Iterable[tuple[str, pd.DataFrame, object]],
    template: pd.DataFrame,
    *,
    max_nearest_time_delta_s: float | None = None,
    max_iterations: int = 64,
    tolerance_m: float = 1.0e-4,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return geometric-median estimates after validating public inputs."""

    iteration_limit, convergence_tolerance_m = _validate_solver_controls(
        max_iterations,
        tolerance_m,
    )
    inputs = _validated_runtime_inputs(estimate_inputs)
    return _ORIGINAL_BUILD_TRACK5_GEOMETRIC_MEDIAN_ENSEMBLE(
        inputs,
        template,
        max_nearest_time_delta_s=max_nearest_time_delta_s,
        max_iterations=iteration_limit,
        tolerance_m=convergence_tolerance_m,
    )


def write_track5_geometric_median_outputs(
    *,
    estimate_inputs: Iterable[EstimateInput],
    template: pd.DataFrame,
    output_dir: Path,
    class_map: dict[str, str] | None = None,
    default_classification: int | str = 0,
    max_nearest_time_delta_s: float | None = None,
    max_iterations: int = 64,
    tolerance_m: float = 1.0e-4,
) -> dict[str, Path]:
    """Validate weights and solver controls before file or artifact access."""

    iteration_limit, convergence_tolerance_m = _validate_solver_controls(
        max_iterations,
        tolerance_m,
    )
    inputs = _validated_estimate_input_objects(estimate_inputs)
    return _ORIGINAL_WRITE_TRACK5_GEOMETRIC_MEDIAN_OUTPUTS(
        estimate_inputs=inputs,
        template=template,
        output_dir=output_dir,
        class_map=class_map,
        default_classification=default_classification,
        max_nearest_time_delta_s=max_nearest_time_delta_s,
        max_iterations=iteration_limit,
        tolerance_m=convergence_tolerance_m,
    )


_IMPL._validate_weight = _validate_estimate_weight
_IMPL.weighted_geometric_median = weighted_geometric_median
_IMPL.build_track5_geometric_median_ensemble = build_track5_geometric_median_ensemble
_IMPL.write_track5_geometric_median_outputs = write_track5_geometric_median_outputs

globals().update(
    {
        name: getattr(_IMPL, name)
        for name in dir(_IMPL)
        if not (name.startswith("__") and name.endswith("__"))
    }
)
globals()["EstimateInput"] = EstimateInput
globals()["_validate_estimate_weight"] = _validate_estimate_weight
globals()["_validated_runtime_inputs"] = _validated_runtime_inputs
globals()["_validated_estimate_input_objects"] = _validated_estimate_input_objects
globals()["_validated_solver_samples"] = _validated_solver_samples
globals()["weighted_geometric_median"] = weighted_geometric_median
globals()["build_track5_geometric_median_ensemble"] = (
    build_track5_geometric_median_ensemble
)
globals()["write_track5_geometric_median_outputs"] = (
    write_track5_geometric_median_outputs
)

__doc__ = _IMPL.__doc__
__all__ = [
    name for name in dir(_IMPL) if not (name.startswith("__") and name.endswith("__"))
]
