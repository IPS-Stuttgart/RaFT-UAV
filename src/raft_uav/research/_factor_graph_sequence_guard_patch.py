"""Reject ambiguous factor-graph inputs that could fuse separate physical flights."""

from __future__ import annotations

from functools import wraps
from types import ModuleType

import numpy as np
import pandas as pd

_PATCH_MARKER = "_raft_uav_rejects_pooled_factor_graph_sequences"
_SCOPE_ALIASES = ("sequence_id", "flight_id")


def _normalized_scope_id(value: object) -> str | None:
    """Return a stable non-empty scope identifier or ``None`` for missing values."""

    if value is None or np.ma.is_masked(value):
        return None
    if isinstance(value, np.ndarray):
        if value.ndim != 0:
            return repr(value)
        value = value.item()
        if value is None or np.ma.is_masked(value):
            return None
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        missing = False
    if isinstance(missing, (bool, np.bool_)) and missing:
        return None
    text = str(value).strip()
    if not text or text.casefold() in {"<na>", "nan", "nat"}:
        return None
    return text


def _single_scope_values(
    frame: pd.DataFrame | None,
    *,
    name: str,
) -> dict[str, str]:
    """Return one complete value per available physical-flight scope alias."""

    if frame is None:
        return {}

    scope: dict[str, str] = {}
    for alias in _SCOPE_ALIASES:
        if alias not in frame.columns:
            continue

        normalized = [_normalized_scope_id(value) for value in frame[alias]]
        identifiers = {
            identifier
            for identifier in normalized
            if identifier is not None
        }
        if len(identifiers) > 1:
            raise ValueError(
                f"{name} contains multiple {alias} values; factor-graph "
                "smoothing must be run separately for each physical flight"
            )

        missing_positions = [
            position
            for position, identifier in enumerate(normalized)
            if identifier is None
        ]
        if identifiers and missing_positions:
            preview = ", ".join(str(position) for position in missing_positions[:8])
            if len(missing_positions) > 8:
                preview = f"{preview}, ..."
            raise ValueError(
                f"{name} contains partially missing {alias} values at row "
                f"positions [{preview}]; factor-graph smoothing requires "
                f"{alias} metadata to be complete or absent"
            )
        if identifiers:
            scope[alias] = next(iter(identifiers))
    return scope


def _require_matching_scope_values(
    left: dict[str, str],
    right: dict[str, str],
    *,
    left_name: str,
    right_name: str,
) -> None:
    """Reject inputs whose shared physical-flight aliases explicitly disagree."""

    for alias in _SCOPE_ALIASES:
        if alias not in left or alias not in right:
            continue
        if left[alias] != right[alias]:
            raise ValueError(
                f"{left_name} and {right_name} {alias} values do not match: "
                f"{left[alias]!r} != {right[alias]!r}"
            )


def apply_factor_graph_sequence_guard_patch(module: ModuleType) -> None:
    """Patch public factor-graph APIs to reject cross-flight fusion."""

    original_smooth = module.smooth_position_trajectory
    original_coordinate_descent = module.coordinate_descent_association_and_smoothing
    if getattr(original_smooth, _PATCH_MARKER, False):
        return

    @wraps(original_smooth)
    def smooth_position_trajectory(
        measurements: pd.DataFrame,
        *,
        initial: pd.DataFrame | None = None,
        config: object | None = None,
    ):
        measurement_scope = _single_scope_values(measurements, name="measurements")
        initial_scope = _single_scope_values(initial, name="initial")
        _require_matching_scope_values(
            measurement_scope,
            initial_scope,
            left_name="measurements",
            right_name="initial",
        )
        return original_smooth(measurements, initial=initial, config=config)

    @wraps(original_coordinate_descent)
    def coordinate_descent_association_and_smoothing(
        radar: pd.DataFrame,
        rf: pd.DataFrame | None = None,
        *,
        iterations: int = 3,
        candidate_gate_m: float = 250.0,
        config: object | None = None,
    ):
        radar_scope = _single_scope_values(radar, name="radar")
        rf_scope = _single_scope_values(rf, name="rf")
        _require_matching_scope_values(
            radar_scope,
            rf_scope,
            left_name="radar",
            right_name="rf",
        )
        return original_coordinate_descent(
            radar,
            rf,
            iterations=iterations,
            candidate_gate_m=candidate_gate_m,
            config=config,
        )

    setattr(smooth_position_trajectory, _PATCH_MARKER, True)
    setattr(coordinate_descent_association_and_smoothing, _PATCH_MARKER, True)
    module.smooth_position_trajectory = smooth_position_trajectory
    module.coordinate_descent_association_and_smoothing = (
        coordinate_descent_association_and_smoothing
    )

    implementation = getattr(module, "_LEGACY", None)
    if implementation is not None:
        implementation.smooth_position_trajectory = smooth_position_trajectory
        implementation.coordinate_descent_association_and_smoothing = (
            coordinate_descent_association_and_smoothing
        )
