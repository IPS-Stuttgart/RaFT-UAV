"""Reject ambiguous one-sided sequence metadata in uncertainty calibration."""

from __future__ import annotations

from functools import wraps
from typing import Any, Callable

_PATCH_MARKER = "_raft_uav_guards_one_sided_uncertainty_sequences"
_ERROR = (
    "uncertainty residual alignment cannot use one-sided sequence_id metadata "
    "when the labeled input contains multiple or missing sequence identifiers"
)


def _validate_one_sided_sequence_metadata(
    uncertainty_module: Any,
    frame: Any,
    truth: Any,
) -> None:
    """Reject pooled or partially labeled inputs that cannot be aligned safely."""

    frame_has_sequence = "sequence_id" in frame.columns
    truth_has_sequence = "sequence_id" in truth.columns
    if frame_has_sequence == truth_has_sequence:
        return

    labeled = frame if frame_has_sequence else truth
    keys = uncertainty_module._sequence_keys(labeled["sequence_id"])
    known = keys.dropna()
    sequence_count = int(known.nunique(dropna=True))
    has_partial_labels = sequence_count > 0 and bool(keys.isna().any())
    if sequence_count > 1 or has_partial_labels:
        raise ValueError(_ERROR)


def install() -> None:
    """Install a guard around the uncertainty residual-alignment boundary."""

    from raft_uav import uncertainty as uncertainty_module

    implementation_module = getattr(
        uncertainty_module,
        "_legacy",
        uncertainty_module,
    )
    original: Callable[..., Any] = implementation_module._aligned_residuals
    if getattr(original, _PATCH_MARKER, False):
        uncertainty_module._aligned_residuals = original
        return

    @wraps(original)
    def aligned_residuals(frame, truth, *, max_time_delta_s):
        _validate_one_sided_sequence_metadata(
            uncertainty_module,
            frame,
            truth,
        )
        return original(
            frame,
            truth,
            max_time_delta_s=max_time_delta_s,
        )

    setattr(aligned_residuals, _PATCH_MARKER, True)
    implementation_module._aligned_residuals = aligned_residuals
    uncertainty_module._aligned_residuals = aligned_residuals
