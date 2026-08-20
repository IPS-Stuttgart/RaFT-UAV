"""Keep compact diagnostic computations within sequence boundaries."""

from __future__ import annotations

from collections import Counter
from importlib import import_module

import pandas as pd

from raft_uav.numeric import optional_int

_diagnostics = import_module("raft_uav.evaluation.diagnostics")
_ORIGINAL_POSITION_ERROR_FRAME = _diagnostics._position_error_frame
_SEQUENCE_ALIASES = ("sequence_id", "flight_id")
_MISSING_SEQUENCE_STRINGS = {"", "nan", "none", "<na>", "nat"}


def _canonical_sequence_column(
    frame: pd.DataFrame,
    *,
    name: str,
) -> pd.Series | None:
    """Return canonical sequence labels or ``None`` when no alias is present."""

    available: list[pd.Series] = []
    for alias in _SEQUENCE_ALIASES:
        matching = [
            column
            for column in frame.columns
            if str(column).strip().casefold() == alias
        ]
        if not matching:
            continue
        if len(matching) != 1:
            raise ValueError(f"{name} has duplicate {alias!r} columns")
        values = frame.loc[:, matching[0]]
        if isinstance(values, pd.DataFrame):
            raise ValueError(f"{name} has duplicate {alias!r} columns")
        missing = values.isna()
        text = values.where(~missing, "").astype(str).str.strip()
        missing = missing | text.str.casefold().isin(_MISSING_SEQUENCE_STRINGS)
        available.append(text.where(~missing, pd.NA))

    if not available:
        return None

    combined = available[0].copy()
    for values in available[1:]:
        conflict = combined.notna() & values.notna() & (combined != values)
        if bool(conflict.any()):
            raise ValueError(
                f"{name} has conflicting sequence_id and flight_id values"
            )
        combined = combined.where(combined.notna(), values)
    return combined


def _sequence_state(
    values: pd.Series | None,
) -> tuple[str, tuple[str, ...]]:
    """Describe whether sequence metadata is absent, missing, partial, or complete."""

    if values is None:
        return "absent", ()
    present = values.notna()
    if not bool(present.any()):
        return "missing", ()
    unique = tuple(dict.fromkeys(values.loc[present].astype(str).tolist()))
    if not bool(present.all()):
        return "partial", unique
    return "complete", unique


def _validate_one_sided_sequence_metadata(
    *,
    labeled_state: str,
    labeled_ids: tuple[str, ...],
    labeled_name: str,
    unlabeled_name: str,
) -> None:
    """Allow one-sided metadata only when it identifies one complete sequence."""

    if labeled_state == "partial":
        raise ValueError(
            f"{labeled_name} sequence metadata is partially missing; "
            f"{unlabeled_name} has no usable sequence metadata"
        )
    if labeled_state == "complete" and len(labeled_ids) > 1:
        raise ValueError(
            f"cannot align pooled {labeled_name} against {unlabeled_name} "
            "without matching sequence metadata"
        )


def _track_switch_rows(frame: pd.DataFrame, *, top_n: int) -> dict[str, object]:
    """Summarize one chronological scope with deterministic tie ordering."""

    if frame.empty or "track_id" not in frame.columns:
        return _diagnostics._empty_track_switch_summary()

    work = frame.copy()
    if "time_s" in work.columns:
        work["_time_s"] = pd.to_numeric(work["time_s"], errors="coerce")
        work = work.sort_values("_time_s", kind="mergesort")

    parsed_track_ids: list[tuple[int, int]] = []
    for position, value in enumerate(work["track_id"].tolist()):
        track_id = optional_int(value)
        if track_id is not None:
            parsed_track_ids.append((position, track_id))
    if not parsed_track_ids:
        return _diagnostics._empty_track_switch_summary()

    events: list[dict[str, object]] = []
    transitions: Counter[tuple[int, int]] = Counter()
    previous: int | None = None
    for position, current_id in parsed_track_ids:
        if previous is not None and current_id != previous:
            transitions[(previous, current_id)] += 1
            event: dict[str, object] = {
                "from_track_id": previous,
                "to_track_id": current_id,
            }
            if "time_s" in work.columns:
                event["time_s"] = _diagnostics._json_value(
                    work.iloc[position]["time_s"]
                )
            events.append(event)
        previous = current_id

    track_ids = [track_id for _, track_id in parsed_track_ids]
    return {
        "count": int(sum(transitions.values())),
        "updates_with_track_id": int(len(track_ids)),
        "unique_track_ids": int(len(set(track_ids))),
        "first_track_id": track_ids[0],
        "last_track_id": track_ids[-1],
        "top_transitions": [
            {"from_track_id": int(src), "to_track_id": int(dst), "count": int(count)}
            for (src, dst), count in transitions.most_common(top_n)
        ],
        "events": events[:top_n],
    }


def _track_switch_summary(frame: pd.DataFrame, *, top_n: int) -> dict[str, object]:
    """Count switches within each complete sequence instead of across pooled runs."""

    sequence_ids = _canonical_sequence_column(frame, name="track switch frame")
    sequence_state, sequence_unique = _sequence_state(sequence_ids)
    if sequence_state != "complete" or len(sequence_unique) <= 1:
        return _track_switch_rows(frame, top_n=top_n)

    assert sequence_ids is not None
    transition_counts: Counter[tuple[int, int]] = Counter()
    events: list[dict[str, object]] = []
    all_track_ids: list[int] = []
    first_track_id: int | None = None
    last_track_id: int | None = None

    for sequence_id in sequence_unique:
        sequence_frame = frame.loc[sequence_ids == sequence_id]
        sequence_top_n = max(top_n, len(sequence_frame))
        summary = _track_switch_rows(sequence_frame, top_n=sequence_top_n)

        if "track_id" in sequence_frame.columns:
            for value in sequence_frame["track_id"].tolist():
                track_id = optional_int(value)
                if track_id is not None:
                    all_track_ids.append(track_id)

        if first_track_id is None and summary["first_track_id"] is not None:
            first_track_id = int(summary["first_track_id"])
        if summary["last_track_id"] is not None:
            last_track_id = int(summary["last_track_id"])

        for transition in summary["top_transitions"]:
            key = (
                int(transition["from_track_id"]),
                int(transition["to_track_id"]),
            )
            transition_counts[key] += int(transition["count"])
        events.extend(summary["events"])

    return {
        "count": int(sum(transition_counts.values())),
        "updates_with_track_id": int(len(all_track_ids)),
        "unique_track_ids": int(len(set(all_track_ids))),
        "first_track_id": first_track_id,
        "last_track_id": last_track_id,
        "top_transitions": [
            {"from_track_id": int(src), "to_track_id": int(dst), "count": int(count)}
            for (src, dst), count in transition_counts.most_common(top_n)
        ],
        "events": events[:top_n],
    }


def _position_error_frame(
    *,
    estimate_frame: pd.DataFrame,
    truth: pd.DataFrame,
    max_eval_time_delta_s: float | None,
) -> pd.DataFrame:
    """Align diagnostic errors within sequence boundaries when metadata exists."""

    estimate_ids = _canonical_sequence_column(
        estimate_frame,
        name="estimate_frame",
    )
    truth_ids = _canonical_sequence_column(truth, name="truth")
    estimate_state, estimate_unique = _sequence_state(estimate_ids)
    truth_state, truth_unique = _sequence_state(truth_ids)

    estimate_usable = estimate_state in {"partial", "complete"}
    truth_usable = truth_state in {"partial", "complete"}
    if not estimate_usable and not truth_usable:
        return _ORIGINAL_POSITION_ERROR_FRAME(
            estimate_frame=estimate_frame,
            truth=truth,
            max_eval_time_delta_s=max_eval_time_delta_s,
        )

    if estimate_usable and not truth_usable:
        _validate_one_sided_sequence_metadata(
            labeled_state=estimate_state,
            labeled_ids=estimate_unique,
            labeled_name="estimate_frame",
            unlabeled_name="truth",
        )
        return _ORIGINAL_POSITION_ERROR_FRAME(
            estimate_frame=estimate_frame,
            truth=truth,
            max_eval_time_delta_s=max_eval_time_delta_s,
        )

    if truth_usable and not estimate_usable:
        _validate_one_sided_sequence_metadata(
            labeled_state=truth_state,
            labeled_ids=truth_unique,
            labeled_name="truth",
            unlabeled_name="estimate_frame",
        )
        return _ORIGINAL_POSITION_ERROR_FRAME(
            estimate_frame=estimate_frame,
            truth=truth,
            max_eval_time_delta_s=max_eval_time_delta_s,
        )

    if estimate_state == "partial" or truth_state == "partial":
        raise ValueError(
            "diagnostic sequence metadata is partially missing; provide labels "
            "for every estimate and truth row or omit them from both tables"
        )

    assert estimate_ids is not None
    assert truth_ids is not None
    order_column = "__raft_uav_diagnostic_row_order__"
    while order_column in estimate_frame.columns:
        order_column = f"_{order_column}"

    estimate_work = estimate_frame.copy()
    estimate_work[order_column] = range(len(estimate_work))
    outputs: list[pd.DataFrame] = []
    for sequence_id in estimate_unique:
        estimate_mask = estimate_ids == sequence_id
        truth_mask = truth_ids == sequence_id
        if not bool(truth_mask.any()):
            continue
        aligned = _ORIGINAL_POSITION_ERROR_FRAME(
            estimate_frame=estimate_work.loc[estimate_mask],
            truth=truth.loc[truth_mask],
            max_eval_time_delta_s=max_eval_time_delta_s,
        )
        if not aligned.empty:
            outputs.append(aligned)

    if not outputs:
        return pd.DataFrame()
    combined = pd.concat(outputs, axis=0)
    return combined.sort_values(order_column, kind="mergesort").drop(
        columns=[order_column]
    )


def install() -> None:
    """Install sequence-local alignment and switch diagnostics."""

    if getattr(_diagnostics, "_sequence_scope_patch_applied", False):
        return
    _diagnostics._position_error_frame = _position_error_frame
    _diagnostics._track_switch_summary = _track_switch_summary
    implementation = getattr(_diagnostics, "_IMPL", None)
    if implementation is not None:
        implementation._position_error_frame = _position_error_frame
        implementation._track_switch_summary = _track_switch_summary
    _diagnostics._sequence_scope_patch_applied = True


install()
