"""Scope MMUAD submission evaluation by physical flight identifiers."""

from __future__ import annotations

from importlib import import_module

import pandas as pd
from pandas.api.types import is_scalar


_evaluate = import_module("raft_uav.mmuad.evaluate")
_ORIGINAL_AUTHORITATIVE_TRUTH_ROWS = _evaluate._authoritative_truth_rows
_ORIGINAL_MATCH_SUBMISSION_TO_TRUTH = _evaluate.match_submission_to_truth
_ORIGINAL_MEAN_FINAL_ERROR = _evaluate._mean_final_error
_SCOPE_TOKEN_PREFIX = "__raft_uav_submission_evaluation_scope_"
_MISSING_SCOPE_TEXT = frozenset({"", "nan", "none", "null", "<na>", "nat"})


def _canonical_flight_id(value: object, *, field: str) -> str | None:
    """Return one normalized scalar flight identifier or ``None`` when missing."""

    if not is_scalar(value):
        raise ValueError(f"{field} values must be scalar")
    if value is None:
        return None
    try:
        missing = bool(pd.isna(value))
    except (TypeError, ValueError):
        missing = False
    if missing:
        return None
    text = str(value).strip()
    return None if text.casefold() in _MISSING_SCOPE_TEXT else text


def _sequence_ids(frame: pd.DataFrame) -> pd.Series:
    """Return normalized sequence IDs without mutating the caller's frame."""

    if "sequence_id" not in frame.columns:
        values = pd.Series("default", index=frame.index, dtype=object)
    else:
        values = frame["sequence_id"]
        if isinstance(values, pd.DataFrame):
            raise ValueError("frame has duplicate 'sequence_id' columns")
    return _evaluate._normalize_submission_sequence_ids(values)


def _flight_ids(frame: pd.DataFrame, *, name: str) -> pd.Series | None:
    """Return normalized flight IDs when the frame contains that scope field."""

    if "flight_id" not in frame.columns:
        return None
    values = frame["flight_id"]
    if isinstance(values, pd.DataFrame):
        raise ValueError(f"{name} has duplicate 'flight_id' columns")
    return pd.Series(
        [
            _canonical_flight_id(value, field=f"{name}.flight_id")
            for value in values.tolist()
        ],
        index=frame.index,
        dtype=object,
    )


def _has_usable_flight_ids(values: pd.Series | None) -> bool:
    """Return whether at least one non-missing flight identifier is present."""

    return values is not None and bool(values.notna().any())


def _validate_one_sided_flight_metadata(
    frame: pd.DataFrame,
    flight_ids: pd.Series,
    *,
    name: str,
    other_name: str,
) -> None:
    """Allow one-sided flight metadata only for one complete flight per sequence."""

    present = flight_ids.notna()
    if not bool(present.all()):
        raise ValueError(
            f"{name} flight_id metadata is partially missing; {other_name} has no "
            "usable flight_id metadata"
        )
    scopes = pd.DataFrame(
        {
            "sequence_id": _sequence_ids(frame).to_numpy(dtype=object),
            "flight_id": flight_ids.to_numpy(dtype=object),
        }
    )
    flight_counts = scopes.groupby("sequence_id", dropna=False)["flight_id"].nunique()
    if bool((flight_counts > 1).any()):
        raise ValueError(
            f"cannot align pooled {name} against {other_name} without matching "
            "flight_id metadata"
        )


def _joint_flight_scope(
    submission: pd.DataFrame,
    truth: pd.DataFrame,
) -> tuple[pd.Series | None, pd.Series | None, bool]:
    """Return normalized flight IDs and whether both sides require joint scope."""

    submission_flights = _flight_ids(submission, name="submission")
    truth_flights = _flight_ids(truth, name="truth")
    submission_usable = _has_usable_flight_ids(submission_flights)
    truth_usable = _has_usable_flight_ids(truth_flights)
    if submission_usable and truth_usable:
        assert submission_flights is not None
        assert truth_flights is not None
        if not bool(submission_flights.notna().all()) or not bool(
            truth_flights.notna().all()
        ):
            raise ValueError(
                "flight_id metadata is partially missing; provide labels for every "
                "submission and truth row or omit them from both tables"
            )
        return submission_flights, truth_flights, True
    if submission_usable:
        assert submission_flights is not None
        _validate_one_sided_flight_metadata(
            submission,
            submission_flights,
            name="submission",
            other_name="truth",
        )
    if truth_usable:
        assert truth_flights is not None
        _validate_one_sided_flight_metadata(
            truth,
            truth_flights,
            name="truth",
            other_name="submission",
        )
    return submission_flights, truth_flights, False


def _scope_key_rows(
    frame: pd.DataFrame,
    flight_ids: pd.Series,
) -> list[tuple[str, str | None]]:
    """Build collision-free physical-scope keys for each frame row."""

    sequences = _sequence_ids(frame).tolist()
    flights = flight_ids.tolist()
    return [
        (str(sequence_id), flight_id)
        for sequence_id, flight_id in zip(sequences, flights, strict=True)
    ]


def _scope_frames_by_flight(
    submission: pd.DataFrame,
    truth: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    dict[str, tuple[str, str | None]] | None,
]:
    """Replace sequence IDs with joint sequence/flight tokens when available."""

    submission_rows = pd.DataFrame(submission).copy()
    truth_rows = pd.DataFrame(truth).copy()
    submission_flights, truth_flights, use_joint_scope = _joint_flight_scope(
        submission_rows,
        truth_rows,
    )
    if not use_joint_scope:
        return submission_rows, truth_rows, None

    assert submission_flights is not None
    assert truth_flights is not None
    tokens: dict[tuple[str, str | None], str] = {}
    metadata: dict[str, tuple[str, str | None]] = {}

    def token_for(key: tuple[str, str | None]) -> str:
        token = tokens.get(key)
        if token is None:
            token = f"{_SCOPE_TOKEN_PREFIX}{len(tokens)}"
            tokens[key] = token
            metadata[token] = key
        return token

    submission_rows["sequence_id"] = [
        token_for(key) for key in _scope_key_rows(submission_rows, submission_flights)
    ]
    truth_rows["sequence_id"] = [
        token_for(key) for key in _scope_key_rows(truth_rows, truth_flights)
    ]
    submission_rows = submission_rows.drop(columns=["flight_id"], errors="ignore")
    truth_rows = truth_rows.drop(columns=["flight_id"], errors="ignore")
    return submission_rows, truth_rows, metadata


def _scope_single_frame_by_flight(
    frame: pd.DataFrame,
    *,
    name: str,
) -> tuple[pd.DataFrame, dict[str, tuple[str, str | None]] | None]:
    """Replace one frame's sequence IDs with physical-flight scope tokens."""

    rows = pd.DataFrame(frame).copy()
    flight_ids = _flight_ids(rows, name=name)
    if not _has_usable_flight_ids(flight_ids):
        return rows, None
    assert flight_ids is not None

    tokens: dict[tuple[str, str | None], str] = {}
    metadata: dict[str, tuple[str, str | None]] = {}
    scoped_ids: list[str] = []
    for key in _scope_key_rows(rows, flight_ids):
        token = tokens.get(key)
        if token is None:
            token = f"{_SCOPE_TOKEN_PREFIX}{len(tokens)}"
            tokens[key] = token
            metadata[token] = key
        scoped_ids.append(token)
    rows["sequence_id"] = scoped_ids
    rows = rows.drop(columns=["flight_id"], errors="ignore")
    return rows, metadata


def _restore_scope_columns(
    frame: pd.DataFrame,
    metadata: dict[str, tuple[str, str | None]] | None,
) -> pd.DataFrame:
    """Restore public sequence and flight identifiers after scoped evaluation."""

    rows = pd.DataFrame(frame).copy()
    if metadata is None or rows.empty or "sequence_id" not in rows.columns:
        return rows

    sequences: list[str] = []
    flights: list[object] = []
    for value in rows["sequence_id"].astype(str).tolist():
        scope = metadata.get(value)
        if scope is None:  # pragma: no cover - internal contract guard
            raise RuntimeError(
                f"submission evaluation returned unknown internal scope token {value!r}"
            )
        sequence_id, flight_id = scope
        sequences.append(sequence_id)
        flights.append(pd.NA if flight_id is None else flight_id)
    rows["sequence_id"] = sequences
    rows["flight_id"] = flights
    return rows


def _authoritative_truth_rows(
    truth: pd.DataFrame,
    *,
    require_positions: bool = True,
) -> pd.DataFrame:
    """Retain final truth snapshots independently for each physical flight."""

    scoped_truth, metadata = _scope_single_frame_by_flight(truth, name="truth")
    rows = _ORIGINAL_AUTHORITATIVE_TRUTH_ROWS(
        scoped_truth,
        require_positions=require_positions,
    )
    return _restore_scope_columns(rows, metadata)


def match_submission_to_truth(
    submission: pd.DataFrame,
    truth: pd.DataFrame,
    *,
    max_time_delta_s: float = 0.5,
) -> pd.DataFrame:
    """Match predictions and truth only within complete physical-flight scopes."""

    scoped_submission, scoped_truth, metadata = _scope_frames_by_flight(
        submission,
        truth,
    )
    matches = _ORIGINAL_MATCH_SUBMISSION_TO_TRUTH(
        scoped_submission,
        scoped_truth,
        max_time_delta_s=max_time_delta_s,
    )
    return _restore_scope_columns(matches, metadata)


def _mean_final_error(frame: pd.DataFrame, column: str) -> float | None:
    """Average trajectory endpoints independently across physical flights."""

    scoped, _ = _scope_single_frame_by_flight(frame, name="matches")
    return _ORIGINAL_MEAN_FINAL_ERROR(scoped, column)


def install() -> None:
    """Install physical-flight scoping on public and legacy evaluator paths."""

    if getattr(_evaluate, "_flight_scope_patch_applied", False):
        return

    _evaluate._authoritative_truth_rows = _authoritative_truth_rows
    _evaluate.match_submission_to_truth = match_submission_to_truth
    _evaluate._mean_final_error = _mean_final_error
    implementation = getattr(_evaluate, "_IMPL", None)
    if implementation is not None:
        implementation._authoritative_truth_rows = _authoritative_truth_rows
        implementation.match_submission_to_truth = match_submission_to_truth
        implementation._mean_final_error = _mean_final_error
    _evaluate._flight_scope_patch_applied = True


install()
