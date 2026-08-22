"""Keep MMUAD multi-object tracking isolated by physical flight scope.

``sequence_id`` is not guaranteed to be globally unique across exported flights.
This guard encodes the joint ``(sequence_id, flight_id)`` scope before invoking
legacy MOT code and restores readable metadata on the result.  It also applies
the same isolation to direct pooled metric calls.
"""

from __future__ import annotations

from functools import wraps
import json
from typing import Any, Callable

import numpy as np
import pandas as pd

_METRICS_MARKER = "_raft_uav_scopes_mot_metrics_by_physical_flight"
_TRACKER_MARKER = "_raft_uav_scopes_mot_state_by_physical_flight"
_SCOPE_TOKEN_PREFIX = "__raft_uav_mot_scope__"
_MISSING_SCOPE_TEXT = {"", "nan", "none", "<na>", "nat"}


def _nonempty_frame(frame: object) -> bool:
    return isinstance(frame, pd.DataFrame) and not frame.empty


def _scope_scalar(value: object) -> str | None:
    """Normalize one physical-scope value without conflating missing text."""

    if value is None or np.ma.is_masked(value):
        return None
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        missing = False
    if isinstance(missing, (bool, np.bool_)) and bool(missing):
        return None
    text = str(value).strip()
    if text.casefold() in _MISSING_SCOPE_TEXT:
        return None
    return text


def _normalized_column(frame: pd.DataFrame, column: str) -> list[str | None]:
    if column not in frame.columns:
        return [None] * len(frame)
    return [_scope_scalar(value) for value in frame[column].tolist()]


def _validate_present_flight_ids(frame: pd.DataFrame, *, label: str) -> None:
    if not _nonempty_frame(frame) or "flight_id" not in frame.columns:
        return
    if any(value is None for value in _normalized_column(frame, "flight_id")):
        raise ValueError(f"{label} flight_id values must be non-missing")


def _validate_sequence_scope(
    left: pd.DataFrame,
    right: pd.DataFrame | None,
    *,
    left_name: str,
    right_name: str,
) -> None:
    """Preserve the maintained MOT sequence-metadata validation contract."""

    for frame, label in ((left, left_name), (right, right_name)):
        if not _nonempty_frame(frame) or "sequence_id" not in frame.columns:
            continue
        if any(value is None for value in _normalized_column(frame, "sequence_id")):
            short_label = "estimates" if label == "MOT estimates" else label
            raise ValueError(f"{short_label} sequence_id values must be non-missing")
    if not _nonempty_frame(left) or not _nonempty_frame(right):
        return
    assert right is not None
    if ("sequence_id" in left.columns) != ("sequence_id" in right.columns):
        if left_name == "MOT estimates" and right_name == "truth":
            raise ValueError(
                "estimates and truth must either both contain sequence_id or both omit it"
            )
        raise ValueError(
            f"{left_name} and {right_name} must either both contain sequence_id or both omit it"
        )


def _flights_by_sequence(frame: pd.DataFrame) -> dict[str | None, set[str]]:
    flights_by_sequence: dict[str | None, set[str]] = {}
    if not _nonempty_frame(frame) or "flight_id" not in frame.columns:
        return flights_by_sequence
    sequences = _normalized_column(frame, "sequence_id")
    flights = _normalized_column(frame, "flight_id")
    for sequence_id, flight_id in zip(sequences, flights):
        if flight_id is None:
            continue
        flights_by_sequence.setdefault(sequence_id, set()).add(flight_id)
    return flights_by_sequence


def _asymmetric_flight_scope_is_ambiguous(
    richer: pd.DataFrame,
    poorer: pd.DataFrame,
) -> bool:
    """Return whether flight IDs cannot be inferred from poorer-side metadata."""

    flights_by_sequence = _flights_by_sequence(richer)
    if "sequence_id" in richer.columns and "sequence_id" in poorer.columns:
        poorer_sequences = set(_normalized_column(poorer, "sequence_id"))
        return any(
            len(flights) > 1
            for sequence_id, flights in flights_by_sequence.items()
            if sequence_id in poorer_sequences
        )
    all_flights = {flight for flights in flights_by_sequence.values() for flight in flights}
    return len(all_flights) > 1


def _validate_two_sided_flight_scope(
    left: pd.DataFrame,
    right: pd.DataFrame | None,
    *,
    left_name: str,
    right_name: str,
) -> None:
    """Reject only asymmetric flight metadata that permits cross-flight pairing."""

    _validate_present_flight_ids(left, label=left_name)
    if isinstance(right, pd.DataFrame):
        _validate_present_flight_ids(right, label=right_name)
    if not _nonempty_frame(left) or not _nonempty_frame(right):
        return
    assert right is not None
    left_has_flight = "flight_id" in left.columns
    right_has_flight = "flight_id" in right.columns
    if left_has_flight == right_has_flight:
        return
    richer, poorer = (left, right) if left_has_flight else (right, left)
    if _asymmetric_flight_scope_is_ambiguous(richer, poorer):
        raise ValueError(
            f"{left_name} and {right_name} have ambiguous flight_id metadata; "
            "both sides must carry flight_id when one sequence contains multiple flights"
        )


def _single_flight_by_sequence(frame: pd.DataFrame) -> dict[str | None, str]:
    mapping: dict[str | None, str] = {}
    for sequence_id, flights in _flights_by_sequence(frame).items():
        if len(flights) == 1:
            mapping[sequence_id] = next(iter(flights))
    return mapping


def _infer_flight_ids(richer: pd.DataFrame, poorer: pd.DataFrame) -> pd.DataFrame:
    """Add flight IDs to an unambiguous poorer-side frame for internal scoping."""

    inferred = poorer.copy()
    if inferred.empty:
        inferred["flight_id"] = pd.Series(index=inferred.index, dtype=object)
        return inferred
    if "sequence_id" in richer.columns and "sequence_id" in inferred.columns:
        mapping = _single_flight_by_sequence(richer)
        inferred["flight_id"] = [
            mapping.get(sequence_id)
            for sequence_id in _normalized_column(inferred, "sequence_id")
        ]
        return inferred

    flights = {flight for values in _flights_by_sequence(richer).values() for flight in values}
    inferred["flight_id"] = next(iter(flights)) if len(flights) == 1 else None
    return inferred


def _scope_token(sequence_id: str | None, flight_id: str | None) -> str:
    payload = json.dumps(
        [sequence_id, flight_id],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return f"{_SCOPE_TOKEN_PREFIX}{payload}"


def _encode_physical_scope(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, tuple[str | None, str | None]]]:
    encoded = frame.copy()
    mapping: dict[str, tuple[str | None, str | None]] = {}
    sequences = _normalized_column(encoded, "sequence_id")
    flights = _normalized_column(encoded, "flight_id")
    tokens: list[str] = []
    for sequence_id, flight_id in zip(sequences, flights):
        token = _scope_token(sequence_id, flight_id)
        mapping[token] = (sequence_id, flight_id)
        tokens.append(token)
    encoded["sequence_id"] = pd.Series(tokens, index=encoded.index, dtype=object)
    return encoded, mapping


def _merge_scope_mappings(
    destination: dict[str, tuple[str | None, str | None]],
    source: dict[str, tuple[str | None, str | None]],
) -> None:
    for token, scope in source.items():
        previous = destination.get(token)
        if previous is not None and previous != scope:
            raise ValueError("MOT physical scope token collision")
        destination[token] = scope


def _scoped_frames(
    left: pd.DataFrame,
    right: pd.DataFrame | None,
    *,
    left_name: str,
    right_name: str,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame | None,
    dict[str, tuple[str | None, str | None]],
    bool,
]:
    """Return internally tokenized frames and a token restoration mapping."""

    left_scoped = left.copy()
    right_scoped = right.copy() if right is not None else None
    _validate_sequence_scope(
        left_scoped,
        right_scoped,
        left_name=left_name,
        right_name=right_name,
    )
    _validate_two_sided_flight_scope(
        left_scoped,
        right_scoped,
        left_name=left_name,
        right_name=right_name,
    )
    left_has_flight = _nonempty_frame(left_scoped) and "flight_id" in left_scoped.columns
    right_has_flight = _nonempty_frame(right_scoped) and "flight_id" in right_scoped.columns
    if not left_has_flight and not right_has_flight:
        return left_scoped, right_scoped, {}, False

    if left_has_flight and isinstance(right_scoped, pd.DataFrame) and not right_has_flight:
        right_scoped = _infer_flight_ids(left_scoped, right_scoped)
    elif right_has_flight and not left_has_flight:
        assert isinstance(right_scoped, pd.DataFrame)
        left_scoped = _infer_flight_ids(right_scoped, left_scoped)

    mapping: dict[str, tuple[str | None, str | None]] = {}
    left_scoped, left_mapping = _encode_physical_scope(left_scoped)
    _merge_scope_mappings(mapping, left_mapping)
    if isinstance(right_scoped, pd.DataFrame):
        right_scoped, right_mapping = _encode_physical_scope(right_scoped)
        _merge_scope_mappings(mapping, right_mapping)
    return left_scoped, right_scoped, mapping, True


def _namespace_metric_ids(
    frame: pd.DataFrame,
    *,
    id_column: str,
) -> pd.DataFrame:
    """Namespace present metric identities by the encoded physical scope."""

    if frame.empty or id_column not in frame.columns or "sequence_id" not in frame.columns:
        return frame
    namespaced = frame.copy()
    identifiers = namespaced[id_column].astype(object).to_numpy(copy=True)
    tokens = namespaced["sequence_id"].astype(str).to_numpy(dtype=object)
    for position, value in enumerate(identifiers):
        normalized = _scope_scalar(value)
        if normalized is None:
            continue
        identifiers[position] = json.dumps(
            [str(tokens[position]), normalized],
            ensure_ascii=False,
            separators=(",", ":"),
        )
    namespaced[id_column] = pd.Series(
        identifiers,
        index=namespaced.index,
        dtype=object,
    )
    return namespaced


def _namespace_metric_inputs(
    estimates: pd.DataFrame,
    truth: pd.DataFrame | None,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    scoped_estimates = estimates
    if "output_track_id" in scoped_estimates.columns:
        scoped_estimates = _namespace_metric_ids(
            scoped_estimates,
            id_column="output_track_id",
        )
    elif "track_id" in scoped_estimates.columns:
        scoped_estimates = _namespace_metric_ids(scoped_estimates, id_column="track_id")
    scoped_truth = truth
    if scoped_truth is not None and "track_id" in scoped_truth.columns:
        scoped_truth = _namespace_metric_ids(scoped_truth, id_column="track_id")
    return scoped_estimates, scoped_truth


def _restore_scope_frame(
    frame: pd.DataFrame,
    mapping: dict[str, tuple[str | None, str | None]],
) -> pd.DataFrame:
    if frame.empty or "sequence_id" not in frame.columns:
        return frame.copy()
    restored = frame.copy()
    scopes: list[tuple[str | None, str | None]] = []
    for value in restored["sequence_id"].tolist():
        token = str(value)
        if token not in mapping:
            raise ValueError("MOT tracker returned an unknown physical scope token")
        scopes.append(mapping[token])
    restored["sequence_id"] = [scope[0] for scope in scopes]
    restored["flight_id"] = [scope[1] for scope in scopes]
    return restored


def _display_scope(scope: tuple[str | None, str | None]) -> str:
    return json.dumps(
        {"flight_id": scope[1], "sequence_id": scope[0]},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _restore_sequence_metrics(
    metrics: dict[str, Any],
    mapping: dict[str, tuple[str | None, str | None]],
) -> dict[str, Any]:
    sequences = metrics.get("sequences")
    if not isinstance(sequences, dict):
        return metrics
    restored = dict(metrics)
    remapped: dict[str, Any] = {}
    for raw_token, value in sequences.items():
        token = str(raw_token)
        if token not in mapping:
            raise ValueError("MOT metrics returned an unknown physical scope token")
        remapped[_display_scope(mapping[token])] = value
    restored["sequences"] = remapped
    return restored


def _install_metric_guard(mot: Any) -> None:
    original: Callable[..., Any] = mot.compute_multi_object_metrics
    if getattr(original, _METRICS_MARKER, False):
        return

    @wraps(original)
    def scoped_metrics(
        estimates: pd.DataFrame,
        truth: pd.DataFrame | None,
        *,
        match_distance_m: float = 25.0,
    ) -> dict[str, Any]:
        scoped_estimates, scoped_truth, _, active = _scoped_frames(
            estimates,
            truth,
            left_name="MOT estimates",
            right_name="truth",
        )
        if active:
            scoped_estimates, scoped_truth = _namespace_metric_inputs(
                scoped_estimates,
                scoped_truth,
            )
        return original(
            scoped_estimates,
            scoped_truth,
            match_distance_m=match_distance_m,
        )

    setattr(scoped_metrics, _METRICS_MARKER, True)
    setattr(mot, "compute_multi_object_metrics", scoped_metrics)


def _install_tracker_guard(mot: Any, mmuad: Any) -> None:
    original: Callable[..., Any] = mot.run_mmuad_multi_object_tracker
    if getattr(original, _TRACKER_MARKER, False):
        setattr(mmuad, "run_mmuad_multi_object_tracker", original)
        return

    @wraps(original)
    def scoped_tracker(
        candidates: Any,
        truth: Any = None,
        *,
        config: Any = None,
    ) -> Any:
        candidate_rows = getattr(candidates, "rows", None)
        truth_rows = getattr(truth, "rows", None) if truth is not None else None
        if not isinstance(candidate_rows, pd.DataFrame):
            return original(candidates, truth, config=config)
        if truth_rows is not None and not isinstance(truth_rows, pd.DataFrame):
            return original(candidates, truth, config=config)

        scoped_candidates, scoped_truth, mapping, active = _scoped_frames(
            candidate_rows,
            truth_rows,
            left_name="MOT candidates",
            right_name="truth",
        )
        if not active:
            return original(candidates, truth, config=config)

        candidate_frame = mot.CandidateFrame(scoped_candidates)
        truth_frame = mot.TruthFrame(scoped_truth) if truth is not None else None
        result = original(candidate_frame, truth_frame, config=config)
        estimates = _restore_scope_frame(result.estimates, mapping)
        selected = _restore_scope_frame(result.selected_tracklets, mapping)
        metrics = _restore_sequence_metrics(result.metrics, mapping)
        metrics = dict(metrics)
        metrics["pooled"] = mot.compute_multi_object_metrics(estimates, truth_rows)
        return mot.TrackerOutput(estimates, metrics, selected)

    setattr(scoped_tracker, _TRACKER_MARKER, True)
    setattr(mot, "run_mmuad_multi_object_tracker", scoped_tracker)
    setattr(mmuad, "run_mmuad_multi_object_tracker", scoped_tracker)


def install(*, mot: Any | None = None, mmuad: Any | None = None) -> None:
    """Install physical-flight scope guards at public MOT boundaries."""

    if mot is None or mmuad is None:
        import raft_uav.mmuad as imported_mmuad
        from raft_uav.mmuad import mot as imported_mot

        mot = imported_mot
        mmuad = imported_mmuad
    _install_metric_guard(mot)
    _install_tracker_guard(mot, mmuad)
