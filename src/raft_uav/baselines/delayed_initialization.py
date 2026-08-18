"""Delayed and multi-hypothesis initialization helpers."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np
import pandas as pd

from raft_uav.numeric import optional_float as _optional_float
from raft_uav.numeric import optional_int as _optional_int

_POSITION_COLUMNS = ("east_m", "north_m", "up_m")
_SEQUENCE_ID_FIELDS = ("sequence_id", "flight_id")
_SERIALIZED_MISSING_SEQUENCE_IDS = frozenset({"nan", "none", "<na>", "nat"})


@dataclass(frozen=True)
class InitialHypothesis:
    """One candidate initial 6D state."""

    time_s: float
    state: np.ndarray
    covariance: np.ndarray
    score: float
    source: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        state = np.asarray(self.state, dtype=float).reshape(6)
        covariance = np.asarray(self.covariance, dtype=float).reshape(6, 6)
        object.__setattr__(self, "time_s", float(self.time_s))
        object.__setattr__(self, "state", state)
        object.__setattr__(self, "covariance", covariance)
        object.__setattr__(self, "score", float(self.score))


def build_delayed_initial_hypotheses(
    *,
    rf_measurements: Iterable[Any],
    radar: pd.DataFrame,
    window_s: float = 5.0,
    max_hypotheses: int = 8,
    initial_position_std_m: float = 50.0,
    initial_velocity_std_mps: float = 15.0,
) -> list[InitialHypothesis]:
    """Build initial-state candidates from the first short RF/radar window."""

    window_s = _require_nonnegative_float(window_s, name="window_s")
    max_hypotheses = _require_nonnegative_int(
        max_hypotheses,
        name="max_hypotheses",
    )
    initial_position_std_m = _require_nonnegative_float(
        initial_position_std_m,
        name="initial_position_std_m",
    )
    initial_velocity_std_mps = _require_nonnegative_float(
        initial_velocity_std_mps,
        name="initial_velocity_std_mps",
    )

    measurement_list = list(rf_measurements)
    _require_single_sequence_inputs(
        rf_measurements=measurement_list,
        radar=radar,
    )
    rf_window = _first_rf_window(measurement_list, window_s=window_s)
    radar_window = _first_radar_window(radar, window_s=window_s)
    hypotheses: list[InitialHypothesis] = []
    for time_s, vector in rf_window:
        position_dimension = min(vector.size, len(_POSITION_COLUMNS))
        state = np.zeros(6)
        if vector.size == 6:
            state[:] = vector
        else:
            state[:position_dimension] = vector[:position_dimension]
        hypotheses.append(
            InitialHypothesis(
                time_s=time_s,
                state=state,
                covariance=_initial_covariance(
                    initial_position_std_m,
                    initial_velocity_std_mps,
                ),
                score=_rf_support_score(
                    time_s,
                    state[:position_dimension],
                    radar_window,
                ),
                source="rf",
                metadata={"rf_dimension": int(vector.size)},
            )
        )
    for _, row in radar_window.iterrows():
        state = _radar_row_state(row, radar_window)
        if state is None:
            continue
        catprob = _optional_float(row.get("cat_prob_uav"))
        catprob_penalty = (
            0.0
            if catprob is None
            else float(-np.log(np.clip(catprob, 1e-6, 1.0)))
        )
        support = _track_support_score(row, radar_window)
        hypotheses.append(
            InitialHypothesis(
                time_s=float(row["time_s"]),
                state=state,
                covariance=_initial_covariance(
                    initial_position_std_m,
                    initial_velocity_std_mps,
                ),
                score=float(catprob_penalty + support),
                source="radar",
                metadata={
                    "track_id": _optional_int(row.get("track_id")),
                    "cat_prob_uav": catprob,
                    "support_score": support,
                },
            )
        )
    return sorted(hypotheses, key=lambda item: item.score)[:max_hypotheses]


def _require_single_sequence_inputs(
    *,
    rf_measurements: Iterable[Any],
    radar: pd.DataFrame,
) -> None:
    """Reject pooled inputs that would initialize one track from several flights."""

    radar_scope = _radar_scope_ids(radar)
    rf_scope = _rf_scope_ids(rf_measurements)
    _reject_multiple_scope_values(source="radar", scope=radar_scope)
    _reject_multiple_scope_values(source="RF measurements", scope=rf_scope)

    common_fields = set(radar_scope) & set(rf_scope)
    mismatches = {
        field: (
            next(iter(radar_scope[field])),
            next(iter(rf_scope[field])),
        )
        for field in sorted(common_fields)
        if radar_scope[field] != rf_scope[field]
    }
    if mismatches:
        raise ValueError(
            "Delayed initialization requires inputs from one sequence; "
            f"radar and RF sequence/flight metadata disagree {mismatches!r}. "
            "Filter the RF and radar data to one sequence before initialization."
        )

    if common_fields or not radar_scope or not rf_scope:
        return

    # Backward compatibility for legacy inputs that expose only one of the two
    # historical scope fields per modality.  Once both fields are present they
    # remain independent dimensions of a joint (sequence_id, flight_id) scope.
    if len(radar_scope) == 1 and len(rf_scope) == 1:
        radar_id = next(iter(next(iter(radar_scope.values()))))
        rf_id = next(iter(next(iter(rf_scope.values()))))
        if radar_id == rf_id:
            return
        raise ValueError(
            "Delayed initialization requires inputs from one sequence; "
            f"radar and RF legacy sequence aliases disagree ({radar_id!r} != {rf_id!r}). "
            "Filter the RF and radar data to one sequence before initialization."
        )


def _radar_scope_ids(radar: pd.DataFrame) -> dict[str, set[str]]:
    if radar.empty:
        return {}

    scope: dict[str, set[str]] = {}
    for field in _SEQUENCE_ID_FIELDS:
        if field not in radar.columns:
            continue
        values, missing_count = _canonical_scope_values(radar[field])
        _reject_partial_sequence_metadata(
            source="radar",
            field=field,
            sequence_ids=values,
            missing_count=missing_count,
            total_count=len(radar),
        )
        if values:
            scope[field] = values
    return scope


def _rf_scope_ids(rf_measurements: Iterable[Any]) -> dict[str, set[str]]:
    measurements = list(rf_measurements)
    if not measurements:
        return {}

    raw_values: dict[str, list[object]] = {
        field: [] for field in _SEQUENCE_ID_FIELDS
    }
    for measurement in measurements:
        aliases = _measurement_sequence_aliases(measurement)
        for field in _SEQUENCE_ID_FIELDS:
            raw_values[field].append(aliases.get(field))

    scope: dict[str, set[str]] = {}
    for field, values in raw_values.items():
        sequence_ids, missing_count = _canonical_scope_values(values)
        _reject_partial_sequence_metadata(
            source="RF measurements",
            field=field,
            sequence_ids=sequence_ids,
            missing_count=missing_count,
            total_count=len(measurements),
        )
        if sequence_ids:
            scope[field] = sequence_ids
    return scope


def _measurement_sequence_aliases(measurement: Any) -> dict[str, object]:
    if isinstance(measurement, Mapping):
        return {
            field: measurement.get(field)
            for field in _SEQUENCE_ID_FIELDS
            if field in measurement
        }
    return {
        field: getattr(measurement, field)
        for field in _SEQUENCE_ID_FIELDS
        if hasattr(measurement, field)
    }


def _canonical_scope_values(values: Iterable[object]) -> tuple[set[str], int]:
    sequence_ids: set[str] = set()
    missing_count = 0
    for value in values:
        sequence_id = _canonical_sequence_id(value)
        if sequence_id is None:
            missing_count += 1
        else:
            sequence_ids.add(sequence_id)
    return sequence_ids, missing_count


def _reject_multiple_scope_values(
    *,
    source: str,
    scope: Mapping[str, set[str]],
) -> None:
    multiple = {
        field: sorted(values)
        for field, values in scope.items()
        if len(values) > 1
    }
    if not multiple:
        return
    raise ValueError(
        "Delayed initialization requires inputs from one sequence; "
        f"{source} spans multiple sequence/flight scopes {multiple!r}. "
        "Filter the RF and radar data to one sequence before initialization."
    )


def _reject_partial_sequence_metadata(
    *,
    source: str,
    field: str,
    sequence_ids: set[str],
    missing_count: int,
    total_count: int,
) -> None:
    if not sequence_ids or missing_count == 0:
        return
    raise ValueError(
        "Delayed initialization requires complete sequence metadata within "
        f"each input; {source} has {missing_count} unlabeled {field} values "
        f"of {total_count} rows."
    )


def _canonical_sequence_id(value: object) -> str | None:
    if value is None:
        return None
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        missing = False
    if not isinstance(missing, (bool, np.bool_)):
        raise ValueError("sequence identifiers must be scalar")
    if bool(missing):
        return None
    text = str(value).strip()
    if not text or text.casefold() in _SERIALIZED_MISSING_SEQUENCE_IDS:
        return None
    return text


def _require_nonnegative_float(value: object, *, name: str) -> float:
    number = _optional_float(value)
    if number is None or number < 0.0:
        raise ValueError(f"{name} must be a finite non-negative real scalar")
    return number


def _require_nonnegative_int(value: object, *, name: str) -> int:
    number = _optional_int(value)
    if number is None or number < 0:
        raise ValueError(f"{name} must be a non-negative integer scalar")
    return number


def best_initial_hypothesis(
    hypotheses: Iterable[InitialHypothesis],
) -> InitialHypothesis | None:
    """Return the lowest-score initial hypothesis."""

    items = list(hypotheses)
    return min(items, key=lambda item: item.score) if items else None


def _measurement_field(
    measurement: Any,
    name: str,
    default: object = None,
) -> object:
    if isinstance(measurement, Mapping):
        return measurement.get(name, default)
    return getattr(measurement, name, default)


def _first_rf_window(
    rf_measurements: Iterable[Any],
    *,
    window_s: float,
) -> list[tuple[float, np.ndarray]]:
    """Return valid RF measurements from the earliest RF initialization window."""

    valid_measurements: list[tuple[float, np.ndarray]] = []
    for measurement in rf_measurements:
        try:
            vector = np.asarray(
                _measurement_field(measurement, "vector", []),
                dtype=float,
            ).reshape(-1)
            time_s = float(_measurement_field(measurement, "time_s"))
        except (TypeError, ValueError, OverflowError):
            continue
        if vector.size < 2 or not np.isfinite(time_s) or not np.isfinite(vector).all():
            continue
        valid_measurements.append((time_s, vector))

    if not valid_measurements:
        return []
    start = min(time_s for time_s, _ in valid_measurements)
    return [
        (time_s, vector)
        for time_s, vector in valid_measurements
        if time_s <= start + window_s
    ]


def _first_radar_window(radar: pd.DataFrame, *, window_s: float) -> pd.DataFrame:
    if radar.empty or "time_s" not in radar.columns:
        return radar.iloc[0:0].copy()

    work = radar.copy()
    work["time_s"] = pd.to_numeric(work["time_s"], errors="coerce")
    finite_time = np.isfinite(
        work["time_s"].to_numpy(dtype=float, na_value=np.nan)
    )
    work = work.loc[finite_time].copy()
    if work.empty:
        return work

    anchor_rows = work
    if set(_POSITION_COLUMNS).issubset(work.columns):
        positions = work.loc[:, _POSITION_COLUMNS].apply(
            pd.to_numeric,
            errors="coerce",
        )
        finite_positions = np.isfinite(
            positions.to_numpy(dtype=float, na_value=np.nan)
        ).all(axis=1)
        anchor_rows = work.loc[finite_positions]
        if anchor_rows.empty:
            return work.iloc[0:0].copy()

    start = float(anchor_rows["time_s"].min())
    ordered = work.sort_values("time_s").reset_index(drop=True)
    return ordered.loc[
        ordered["time_s"].between(start, start + window_s)
    ].copy()


def _radar_row_state(row: pd.Series, frame: pd.DataFrame) -> np.ndarray | None:
    try:
        state = np.array(
            [
                float(row["east_m"]),
                float(row["north_m"]),
                float(row["up_m"]),
                0,
                0,
                0,
            ],
            dtype=float,
        )
    except (KeyError, TypeError, ValueError):
        return None
    if not np.isfinite(state[:3]).all():
        return None
    velocity = _velocity_from_row(row)
    if velocity is None:
        velocity = _velocity_from_track(row, frame)
    if velocity is not None:
        state[3:6] = velocity
    return state


def _velocity_from_row(row: pd.Series) -> np.ndarray | None:
    required = (
        "velocity_east_mps",
        "velocity_north_mps",
        "velocity_down_mps",
    )
    if not all(column in row.index for column in required):
        return None
    try:
        velocity = np.array(
            [
                float(row["velocity_east_mps"]),
                float(row["velocity_north_mps"]),
                -float(row["velocity_down_mps"]),
            ],
            dtype=float,
        )
    except (TypeError, ValueError):
        return None
    return velocity if np.isfinite(velocity).all() else None


def _track_id_matches(values: pd.Series, track_id: int) -> pd.Series:
    """Match only track IDs accepted by the shared exact-integer parser."""

    return values.map(_optional_int).eq(track_id)


def _velocity_from_track(row: pd.Series, frame: pd.DataFrame) -> np.ndarray | None:
    track_id = _optional_int(row.get("track_id"))
    required = {*_POSITION_COLUMNS, "time_s", "track_id"}
    if track_id is None or not required.issubset(frame.columns):
        return None
    track = frame.loc[_track_id_matches(frame["track_id"], track_id)].copy()
    positions = track.loc[:, _POSITION_COLUMNS].apply(pd.to_numeric, errors="coerce")
    times = pd.to_numeric(track["time_s"], errors="coerce")
    finite = np.isfinite(times.to_numpy(dtype=float, na_value=np.nan))
    finite &= np.isfinite(
        positions.to_numpy(dtype=float, na_value=np.nan)
    ).all(axis=1)
    track = track.loc[finite].copy()
    if len(track) < 2:
        return None
    track["time_s"] = times.loc[finite].to_numpy(dtype=float, na_value=np.nan)
    positions = positions.loc[finite].to_numpy(dtype=float, na_value=np.nan)
    order = np.argsort(track["time_s"].to_numpy(dtype=float), kind="stable")
    times_array = track["time_s"].to_numpy(dtype=float)[order]
    positions = positions[order]
    dt = float(times_array[-1] - times_array[0])
    if dt <= 0.0:
        return None
    velocity = (positions[-1] - positions[0]) / dt
    return velocity if np.isfinite(velocity).all() else None


def _rf_support_score(
    time_s: float,
    position: np.ndarray,
    radar: pd.DataFrame,
) -> float:
    position = np.asarray(position, dtype=float).reshape(-1)
    position_dimension = min(position.size, len(_POSITION_COLUMNS))
    if position_dimension == 0:
        return 1.0
    position_columns = _POSITION_COLUMNS[:position_dimension]
    required = {*position_columns, "time_s"}
    if radar.empty or not required.issubset(radar.columns):
        return 1.0
    times = pd.to_numeric(radar["time_s"], errors="coerce")
    positions = radar.loc[:, position_columns].apply(pd.to_numeric, errors="coerce")
    finite = np.isfinite(times.to_numpy(dtype=float, na_value=np.nan))
    finite &= np.isfinite(
        positions.to_numpy(dtype=float, na_value=np.nan)
    ).all(axis=1)
    if not finite.any():
        return 1.0
    times_array = times.loc[finite].to_numpy(dtype=float, na_value=np.nan)
    positions_array = positions.loc[finite].to_numpy(dtype=float, na_value=np.nan)
    nearby = np.abs(times_array - float(time_s)) <= 1.0
    if not nearby.any():
        return 1.0
    distances = np.linalg.norm(
        positions_array[nearby]
        - position[:position_dimension].reshape(1, position_dimension),
        axis=1,
    )
    return float(np.min(distances) / 100.0)


def _track_support_score(row: pd.Series, radar: pd.DataFrame) -> float:
    track_id = _optional_int(row.get("track_id"))
    if track_id is None or "track_id" not in radar.columns:
        return 1.0
    count = int(_track_id_matches(radar["track_id"], track_id).sum())
    return float(1.0 / max(count, 1))


def _initial_covariance(
    position_std_m: float,
    velocity_std_mps: float,
) -> np.ndarray:
    return np.diag([position_std_m**2] * 3 + [velocity_std_mps**2] * 3)
