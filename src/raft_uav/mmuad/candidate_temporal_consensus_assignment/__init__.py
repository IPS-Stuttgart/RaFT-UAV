"""Strict configuration and reciprocal assignment boundary for temporal consensus.

The maintained implementation lives in the sibling
``candidate_temporal_consensus_assignment.py`` module. This package preserves
the public import path while routing explicit configurations through the shared
validated temporal-consensus boundary. It also solves each adjacent frame pair
once and reverses the selected edges for the opposite direction, so tied
one-to-one optima cannot produce contradictory forward and backward matches.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

from raft_uav.mmuad.candidate_temporal_consensus import _validated_config

_IMPL_PATH = Path(__file__).resolve().parent.parent / "candidate_temporal_consensus_assignment.py"
_SPEC = importlib.util.spec_from_file_location(
    "raft_uav.mmuad._candidate_temporal_consensus_assignment_legacy",
    _IMPL_PATH,
)
if _SPEC is None or _SPEC.loader is None:  # pragma: no cover
    raise ImportError(f"cannot load assignment temporal-consensus implementation from {_IMPL_PATH}")
_IMPL = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _IMPL
_SPEC.loader.exec_module(_IMPL)

_ORIGINAL_ADD_ASSIGNMENT_TEMPORAL_CANDIDATE_CONSENSUS = (
    _IMPL.add_assignment_temporal_candidate_consensus
)


def _invert_neighbor_match(
    match: dict[str, Any],
    *,
    current: pd.DataFrame,
    neighbor: pd.DataFrame,
) -> dict[str, Any]:
    """Reverse a one-to-one match without solving the tied assignment again."""

    inverse = _IMPL._empty_match(len(neighbor))
    positions = np.asarray(match["neighbor_position"], dtype=int)
    matched = np.flatnonzero(positions >= 0)
    for current_position in matched:
        neighbor_position = int(positions[current_position])
        inverse["neighbor_position"][neighbor_position] = int(current_position)
        for key in (
            "distance_m",
            "speed_mps",
            "support_count",
            "other_source_count",
            "other_branch_count",
        ):
            inverse[key][neighbor_position] = match[key][current_position]
    inverse["neighbor_rows"] = current
    return inverse


def _annotate_sequence_one_to_one(
    out: pd.DataFrame,
    sequence: pd.DataFrame,
    config: Any,
) -> None:
    """Annotate each adjacent frame pair with one reciprocal assignment."""

    times = np.sort(sequence["time_s"].dropna().unique().astype(float))
    frames = {
        float(time_s): sequence.loc[sequence["time_s"] == float(time_s)].copy()
        for time_s in times
    }
    previous_frames = {float(time_s): None for time_s in times}
    next_frames = {float(time_s): None for time_s in times}
    previous_dt = {float(time_s): np.nan for time_s in times}
    next_dt = {float(time_s): np.nan for time_s in times}
    previous_matches = {
        float(time_s): _IMPL._empty_match(len(frames[float(time_s)]))
        for time_s in times
    }
    next_matches = {
        float(time_s): _IMPL._empty_match(len(frames[float(time_s)]))
        for time_s in times
    }

    for position in range(len(times) - 1):
        current_time = float(times[position])
        neighbor_time = float(times[position + 1])
        dt_s = float(neighbor_time - current_time)
        if not (0.0 < dt_s <= config.max_time_gap_s):
            continue

        current = frames[current_time]
        neighbor = frames[neighbor_time]
        forward_match = _IMPL._one_to_one_neighbor_match(
            current,
            neighbor,
            dt_s,
            config,
        )
        backward_match = _invert_neighbor_match(
            forward_match,
            current=current,
            neighbor=neighbor,
        )
        next_frames[current_time] = neighbor
        next_dt[current_time] = dt_s
        next_matches[current_time] = forward_match
        previous_frames[neighbor_time] = current
        previous_dt[neighbor_time] = dt_s
        previous_matches[neighbor_time] = backward_match

    for time_s in times:
        current_time = float(time_s)
        current = frames[current_time]
        backward_match = previous_matches[current_time]
        forward_match = next_matches[current_time]
        _IMPL._write_assignment_match(
            out,
            current,
            backward_match,
            direction="backward",
        )
        _IMPL._write_assignment_match(
            out,
            current,
            forward_match,
            direction="forward",
        )
        _IMPL._write_bidirectional_metrics(
            out,
            current,
            previous_frames[current_time],
            next_frames[current_time],
            backward_match,
            forward_match,
            previous_dt[current_time],
            next_dt[current_time],
        )


def add_assignment_temporal_candidate_consensus(
    candidates: Any,
    *,
    config: _IMPL.TemporalConsensusConfig | None = None,
    assignment_mode: str = "one-to-one",
) -> Any:
    """Attach assignment consensus after validating the explicit configuration."""

    return _ORIGINAL_ADD_ASSIGNMENT_TEMPORAL_CANDIDATE_CONSENSUS(
        candidates,
        config=_validated_config(config),
        assignment_mode=assignment_mode,
    )


_IMPL._annotate_sequence_one_to_one = _annotate_sequence_one_to_one
_IMPL.add_assignment_temporal_candidate_consensus = (
    add_assignment_temporal_candidate_consensus
)

globals().update(
    {
        name: getattr(_IMPL, name)
        for name in dir(_IMPL)
        if not (name.startswith("__") and name.endswith("__"))
    }
)
globals()["add_assignment_temporal_candidate_consensus"] = (
    add_assignment_temporal_candidate_consensus
)

__doc__ = _IMPL.__doc__
__all__ = [name for name in dir(_IMPL) if not (name.startswith("__") and name.endswith("__"))]
