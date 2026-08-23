"""Compatibility layer for physical-scope MOT validation.

The primary MOT scope patch intentionally fails closed when flight metadata is
ambiguous.  Legacy callers, however, may carry ``flight_id`` on only one side
of a comparison even though ``sequence_id`` already identifies exactly one
flight.  Those calls are unambiguous and should remain valid.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from raft_uav.mmuad import _mot_config_validation_patch as scope_patch


def _normalized_values(values: pd.Series) -> list[Any]:
    return [scope_patch._scope_scalar(value) for value in values.tolist()]


def _has_ambiguous_flights(
    richer: pd.DataFrame,
    poorer: pd.DataFrame,
) -> bool:
    """Return whether missing flight metadata can change physical pairing."""

    if "flight_id" not in richer.columns or richer.empty or poorer.empty:
        return False

    if "sequence_id" not in richer.columns or "sequence_id" not in poorer.columns:
        return len(set(_normalized_values(richer["flight_id"]))) > 1

    poorer_sequences = set(_normalized_values(poorer["sequence_id"]))
    flights_by_sequence: dict[Any, set[Any]] = {}
    sequence_values = _normalized_values(richer["sequence_id"])
    flight_values = _normalized_values(richer["flight_id"])
    for sequence_id, flight_id in zip(sequence_values, flight_values):
        if sequence_id not in poorer_sequences:
            continue
        flights_by_sequence.setdefault(sequence_id, set()).add(flight_id)
    return any(len(flights) > 1 for flights in flights_by_sequence.values())


def _validate_two_sided_flight_scope(
    left: pd.DataFrame,
    right: pd.DataFrame | None,
    *,
    left_name: str,
    right_name: str,
) -> None:
    """Reject asymmetric flight metadata only when it is actually ambiguous."""

    if not scope_patch._nonempty_frame(left) or not scope_patch._nonempty_frame(right):
        return
    assert right is not None
    left_has_flight = "flight_id" in left.columns
    right_has_flight = "flight_id" in right.columns
    if left_has_flight == right_has_flight:
        return

    richer, poorer = (left, right) if left_has_flight else (right, left)
    if _has_ambiguous_flights(richer, poorer):
        raise ValueError(
            f"{left_name} and {right_name} have ambiguous flight_id metadata; "
            "both sides must carry flight_id when one sequence contains multiple flights"
        )


def _tracker_scope_columns(
    candidate_rows: pd.DataFrame,
    truth_rows: pd.DataFrame | None,
) -> tuple[str, ...]:
    """Partition tracker state by flight only when candidates define that scope."""

    if not scope_patch._nonempty_frame(candidate_rows):
        return ()
    if "flight_id" not in candidate_rows.columns:
        return ()

    truth_nonempty = scope_patch._nonempty_frame(truth_rows)
    if truth_nonempty:
        assert truth_rows is not None
        if "flight_id" not in truth_rows.columns:
            # The validator above guarantees that sequence_id already makes this
            # asymmetric legacy case unambiguous, so the legacy sequence partition
            # is sufficient and preserves compatibility.
            return ()

    columns: list[str] = []
    if "sequence_id" in candidate_rows.columns and (
        not truth_nonempty or (truth_rows is not None and "sequence_id" in truth_rows.columns)
    ):
        columns.append("sequence_id")
    columns.append("flight_id")
    return tuple(columns)


def install() -> None:
    """Narrow the MOT scope guard without replacing the already-installed wrappers."""

    scope_patch._validate_two_sided_flight_scope = _validate_two_sided_flight_scope
    scope_patch._tracker_scope_columns = _tracker_scope_columns
