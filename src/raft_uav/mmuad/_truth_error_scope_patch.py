"""Keep MM-UAD truth interpolation inside the active physical scope."""

from __future__ import annotations

from collections.abc import Callable
from functools import wraps

import pandas as pd

from raft_uav.mmuad import _mot_config_validation_patch as scope_patch
from raft_uav.mmuad import _mot_scope_compat_patch as scope_compat


_ROW_POSITION = "__raft_uav_truth_error_row_position"


def _shared_scope_columns(
    estimates: pd.DataFrame,
    truth: pd.DataFrame,
) -> tuple[str, ...]:
    return tuple(
        column
        for column in ("sequence_id", "flight_id")
        if column in estimates.columns and column in truth.columns
    )


def _helper_column(*frames: pd.DataFrame) -> str:
    """Return a temporary row-position column that cannot overwrite user data."""

    column = _ROW_POSITION
    while any(column in frame.columns for frame in frames):
        column += "_"
    return column


def _scoped_truth_errors(
    original: Callable[[pd.DataFrame, pd.DataFrame], pd.DataFrame],
    estimates: pd.DataFrame,
    truth: pd.DataFrame,
) -> pd.DataFrame:
    if estimates.empty or truth.empty:
        return original(estimates, truth)

    scope_compat._validate_two_sided_flight_scope(
        estimates,
        truth,
        left_name="tracker estimates",
        right_name="ground truth",
    )

    scope_columns = _shared_scope_columns(estimates, truth)
    if not scope_columns:
        return original(estimates, truth)

    estimate_tokens = scope_patch._scope_tokens(
        estimates,
        scope_columns=scope_columns,
    )
    truth_tokens = scope_patch._scope_tokens(
        truth,
        scope_columns=scope_columns,
    )
    ordered_tokens = list(dict.fromkeys(str(token) for token in estimate_tokens))
    row_position = _helper_column(estimates, truth)

    scoped_parts: list[pd.DataFrame] = []
    for token in ordered_tokens:
        estimate_mask = (estimate_tokens == token).to_numpy()
        truth_mask = (truth_tokens == token).to_numpy()
        estimate_positions = estimate_mask.nonzero()[0]
        truth_positions = truth_mask.nonzero()[0]

        scoped_estimates = estimates.iloc[estimate_positions].copy()
        scoped_estimates[row_position] = estimate_positions
        scoped_parts.append(
            original(
                scoped_estimates,
                truth.iloc[truth_positions].copy(),
            )
        )

    combined = (
        pd.concat(scoped_parts, ignore_index=True, sort=False)
        .sort_values(row_position, kind="stable")
        .drop(columns=row_position)
    )
    combined.index = estimates.index.copy()
    return combined


def install() -> None:
    """Wrap both public and legacy tracker helpers with scoped interpolation."""

    from raft_uav.mmuad import tracker

    previous = tracker.add_truth_errors
    if getattr(previous, "_raft_uav_truth_scope", False):
        tracker._LEGACY.add_truth_errors = previous
        return

    @wraps(previous)
    def scoped_add_truth_errors(
        estimates: pd.DataFrame,
        truth: pd.DataFrame,
    ) -> pd.DataFrame:
        return _scoped_truth_errors(previous, estimates, truth)

    setattr(scoped_add_truth_errors, "_raft_uav_truth_scope", True)
    setattr(scoped_add_truth_errors, "_raft_uav_previous", previous)
    tracker.add_truth_errors = scoped_add_truth_errors
    tracker._LEGACY.add_truth_errors = scoped_add_truth_errors
