"""Reject ambiguous factor-graph inputs that could fuse separate sequences."""

from __future__ import annotations

from functools import wraps
from types import ModuleType

import numpy as np
import pandas as pd

_PATCH_MARKER = "_raft_uav_rejects_pooled_factor_graph_sequences"


def _normalized_sequence_id(value: object) -> str | None:
    """Return a stable non-empty sequence identifier or ``None`` for missing values."""

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


def _single_sequence_id(frame: pd.DataFrame | None, *, name: str) -> str | None:
    """Return one complete sequence identifier or reject ambiguous metadata."""

    if frame is None or "sequence_id" not in frame.columns:
        return None

    normalized = [
        _normalized_sequence_id(value)
        for value in frame["sequence_id"]
    ]
    identifiers = {
        identifier
        for identifier in normalized
        if identifier is not None
    }
    if len(identifiers) > 1:
        raise ValueError(
            f"{name} contains multiple sequence_id values; "
            "factor-graph smoothing must be run separately for each sequence"
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
            f"{name} contains partially missing sequence_id values at row "
            f"positions [{preview}]; factor-graph smoothing requires "
            "sequence_id metadata to be complete or absent"
        )
    return next(iter(identifiers), None)


def _require_matching_sequence_ids(
    left: str | None,
    right: str | None,
    *,
    left_name: str,
    right_name: str,
) -> None:
    """Reject explicitly labeled inputs from different physical sequences."""

    if left is not None and right is not None and left != right:
        raise ValueError(
            f"{left_name} and {right_name} sequence_id values do not match: "
            f"{left!r} != {right!r}"
        )


def apply_factor_graph_sequence_guard_patch(module: ModuleType) -> None:
    """Patch public factor-graph APIs to reject cross-sequence fusion."""

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
        measurement_sequence = _single_sequence_id(measurements, name="measurements")
        initial_sequence = _single_sequence_id(initial, name="initial")
        _require_matching_sequence_ids(
            measurement_sequence,
            initial_sequence,
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
        radar_sequence = _single_sequence_id(radar, name="radar")
        rf_sequence = _single_sequence_id(rf, name="rf")
        _require_matching_sequence_ids(
            radar_sequence,
            rf_sequence,
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
