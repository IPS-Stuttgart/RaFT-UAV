"""Reject ambiguous one-sided scope metadata in uncertainty calibration."""

from __future__ import annotations

from functools import wraps
from typing import Any, Callable

import numpy as np
import pandas as pd

_PATCH_MARKER = "_raft_uav_guards_one_sided_uncertainty_sequences"
_SCOPE_FIELDS = ("sequence_id", "flight_id")


def _scope_keys(uncertainty_module: Any, values: pd.Series) -> pd.Series:
    """Normalize one physical-scope identifier with the uncertainty convention."""

    return uncertainty_module._sequence_keys(values)


def _raise_ambiguous_scope(field: str) -> None:
    raise ValueError(
        "uncertainty residual alignment cannot use one-sided "
        f"{field} metadata when the labeled input is ambiguous"
    )


def _validate_one_sided_scope_metadata(
    uncertainty_module: Any,
    frame: Any,
    truth: Any,
) -> None:
    """Reject one-sided scope metadata that shared aliases cannot disambiguate."""

    for field in _SCOPE_FIELDS:
        frame_has_field = field in frame.columns
        truth_has_field = field in truth.columns
        if frame_has_field == truth_has_field:
            continue

        labeled = frame if frame_has_field else truth
        keys = _scope_keys(uncertainty_module, labeled[field])
        known = keys.dropna()
        if known.empty:
            continue
        if bool(keys.isna().any()):
            _raise_ambiguous_scope(field)

        shared_fields = tuple(
            other
            for other in _SCOPE_FIELDS
            if other != field and other in frame.columns and other in truth.columns
        )
        if not shared_fields:
            if int(known.nunique(dropna=True)) > 1:
                _raise_ambiguous_scope(field)
            continue

        scope = pd.DataFrame(index=labeled.index)
        scope[field] = keys
        for shared_field in shared_fields:
            scope[shared_field] = _scope_keys(
                uncertainty_module,
                labeled[shared_field],
            )
        complete = scope[[field, *shared_fields]].notna().all(axis=1)
        if not bool(complete.any()):
            continue

        counts = (
            scope.loc[complete]
            .groupby(list(shared_fields), dropna=False)[field]
            .nunique(dropna=True)
        )
        if bool((counts > 1).any()):
            _raise_ambiguous_scope(field)


def _align_with_flight_scope(
    uncertainty_module: Any,
    original: Callable[..., Any],
    frame: pd.DataFrame,
    truth: pd.DataFrame,
    *,
    max_time_delta_s: float,
) -> pd.DataFrame:
    """Keep residual matching inside physical flights when both sides identify them."""

    if "flight_id" not in frame.columns or "flight_id" not in truth.columns:
        return original(
            frame,
            truth,
            max_time_delta_s=max_time_delta_s,
        )

    frame_keys = _scope_keys(uncertainty_module, frame["flight_id"])
    truth_keys = _scope_keys(uncertainty_module, truth["flight_id"])
    order_column = "__raft_uav_uncertainty_flight_alignment_order__"
    while order_column in frame.columns:
        order_column += "_"

    positioned = frame.copy()
    positioned[order_column] = np.arange(len(positioned), dtype=int)
    blocks: list[pd.DataFrame] = []
    for flight_id in pd.unique(frame_keys.dropna()):
        frame_mask = frame_keys.eq(flight_id).fillna(False)
        truth_mask = truth_keys.eq(flight_id).fillna(False)
        flight_truth = truth.loc[truth_mask]
        if flight_truth.empty:
            continue
        block = original(
            positioned.loc[frame_mask],
            flight_truth,
            max_time_delta_s=max_time_delta_s,
        )
        if not block.empty:
            blocks.append(block)

    if not blocks:
        return frame.iloc[0:0].copy()

    aligned = pd.concat(blocks, ignore_index=False)
    aligned = aligned.sort_values(order_column, kind="mergesort")
    return aligned.drop(columns=order_column).reset_index(drop=True)


def install() -> None:
    """Install scope guards around the uncertainty residual-alignment boundary."""

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
        _validate_one_sided_scope_metadata(
            uncertainty_module,
            frame,
            truth,
        )
        return _align_with_flight_scope(
            uncertainty_module,
            original,
            frame,
            truth,
            max_time_delta_s=max_time_delta_s,
        )

    setattr(aligned_residuals, _PATCH_MARKER, True)
    implementation_module._aligned_residuals = aligned_residuals
    uncertainty_module._aligned_residuals = aligned_residuals
