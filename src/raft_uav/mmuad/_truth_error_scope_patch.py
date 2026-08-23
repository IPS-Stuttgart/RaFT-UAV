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


def _normalized_scope_values(values: pd.Series) -> list[object]:
    return [scope_patch._scope_scalar(value) for value in values.tolist()]


def _has_ambiguous_sequences(
    richer: pd.DataFrame,
    poorer: pd.DataFrame,
) -> bool:
    """Return whether missing sequence metadata can change physical pairing."""

    if "sequence_id" not in richer.columns or richer.empty or poorer.empty:
        return False

    if "flight_id" not in richer.columns or "flight_id" not in poorer.columns:
        return len(set(_normalized_scope_values(richer["sequence_id"]))) > 1

    poorer_flights = set(_normalized_scope_values(poorer["flight_id"]))
    sequences_by_flight: dict[object, set[object]] = {}
    sequence_values = _normalized_scope_values(richer["sequence_id"])
    flight_values = _normalized_scope_values(richer["flight_id"])
    for sequence_id, flight_id in zip(sequence_values, flight_values, strict=True):
        if flight_id not in poorer_flights:
            continue
        sequences_by_flight.setdefault(flight_id, set()).add(sequence_id)
    return any(len(sequences) > 1 for sequences in sequences_by_flight.values())


def _validate_two_sided_sequence_scope(
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    left_name: str,
    right_name: str,
) -> None:
    """Reject asymmetric sequence metadata only when pairing is ambiguous."""

    left_has_sequence = "sequence_id" in left.columns
    right_has_sequence = "sequence_id" in right.columns
    if left_has_sequence == right_has_sequence:
        return

    richer, poorer = (left, right) if left_has_sequence else (right, left)
    if _has_ambiguous_sequences(richer, poorer):
        raise ValueError(
            f"{left_name} and {right_name} have ambiguous sequence_id metadata; "
            "both sides must carry sequence_id when the remaining scope aliases "
            "do not uniquely identify a sequence"
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
    _validate_two_sided_sequence_scope(
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
