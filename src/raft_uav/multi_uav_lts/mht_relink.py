"""Beam-search relinking for first-frame-seeded Multi-UAV trajectories."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Protocol, Sequence

from ._records import Detection


class TrackletLike(Protocol):
    """Minimal tracklet interface required by the beam relinker."""

    source_id: int
    index: int
    rows: tuple[Detection, ...]

    @property
    def start_frame(self) -> int:
        ...


class LinkCost(Protocol):
    """Callable used to score one path-to-tracklet transition."""

    def __call__(
        self,
        path: list[Detection],
        candidate: tuple[Detection, ...],
        *,
        gap: int,
    ) -> float:
        ...


@dataclass(frozen=True)
class BeamRelinkResult:
    """Best beam-search assignment plus diagnostics for model selection."""

    assigned: dict[int, tuple[Detection, ...]]
    relinked_tracklets: int
    relinked_source_ids: frozenset[int]
    evaluated_hypotheses: int
    best_cost: float
    second_best_margin: float | None


@dataclass(frozen=True)
class _BeamState:
    cost: float
    path_tails: dict[int, tuple[Detection, ...]]
    links: tuple[tuple[int, int, int], ...]
    linked_rows: int


def relink_tracklets_beam(
    assigned: dict[int, list[Detection]],
    tracklets: Sequence[TrackletLike],
    *,
    max_gap: int,
    max_cost: float,
    beam_width: int,
    drop_cost: float,
    link_cost: LinkCost,
) -> BeamRelinkResult:
    """Relink fragments while retaining several competing identity hypotheses.

    A candidate can be assigned to one compatible seed path or dropped. Dropping
    incurs a quality-weighted penalty so a longer, confident fragment contributes
    more evidence than a one-frame low-confidence fragment. The beam delays hard
    choices until later tracklets provide additional motion evidence.
    """

    if beam_width <= 0:
        raise ValueError("beam_width must be a positive integer")
    if max_gap < 0:
        raise ValueError("max_gap must be non-negative")
    if not math.isfinite(max_cost) or max_cost < 0.0:
        raise ValueError("max_cost must be a finite non-negative scalar")
    if not math.isfinite(drop_cost) or drop_cost < 0.0:
        raise ValueError("drop_cost must be a finite non-negative scalar")

    initial_assigned = {
        seed_id: tuple(sorted(rows, key=lambda row: row.frame_id))
        for seed_id, rows in sorted(assigned.items())
    }
    if any(not rows for rows in initial_assigned.values()):
        raise ValueError("assigned seed paths must contain at least one row")
    initial_tails = {seed_id: rows[-2:] for seed_id, rows in initial_assigned.items()}
    beam = [_BeamState(0.0, initial_tails, (), 0)]
    evaluated = 1

    ordered_tracklets = sorted(
        tracklets,
        key=lambda item: (
            item.start_frame,
            item.rows[-1].frame_id if item.rows else -1,
            item.source_id,
            item.index,
        ),
    )
    tracklet_lookup: dict[tuple[int, int], TrackletLike] = {}
    for tracklet in ordered_tracklets:
        if not tracklet.rows:
            raise ValueError("tracklets must contain at least one row")
        key = (tracklet.source_id, tracklet.index)
        if key in tracklet_lookup:
            raise ValueError(f"duplicate tracklet key: {key}")
        tracklet_lookup[key] = tracklet
        children: list[_BeamState] = []
        for state in beam:
            children.append(
                _BeamState(
                    state.cost + _drop_penalty(tracklet, drop_cost=drop_cost),
                    state.path_tails,
                    state.links,
                    state.linked_rows,
                )
            )
            for seed_id, path_tail in sorted(state.path_tails.items()):
                gap = tracklet.start_frame - path_tail[-1].frame_id - 1
                if not 0 <= gap <= max_gap:
                    continue
                cost = link_cost(list(path_tail), tracklet.rows, gap=gap)
                if not math.isfinite(cost) or cost > max_cost:
                    continue
                relabeled_rows = tuple(
                    replace(row, object_id=seed_id) for row in tracklet.rows
                )
                updated_tails = dict(state.path_tails)
                updated_tails[seed_id] = (*path_tail, *relabeled_rows)[-2:]
                children.append(
                    _BeamState(
                        state.cost + cost,
                        updated_tails,
                        (*state.links, (tracklet.source_id, tracklet.index, seed_id)),
                        state.linked_rows + len(tracklet.rows),
                    )
                )
        evaluated += len(children)
        beam = _prune(children, beam_width=beam_width)

    ranked = sorted(beam, key=_state_rank)
    best = ranked[0]
    second_margin = None
    if len(ranked) > 1:
        second_margin = ranked[1].cost - best.cost
    return BeamRelinkResult(
        assigned=_materialize_assignment(initial_assigned, best.links, tracklet_lookup),
        relinked_tracklets=len(best.links),
        relinked_source_ids=frozenset(link[0] for link in best.links),
        evaluated_hypotheses=evaluated,
        best_cost=best.cost,
        second_best_margin=second_margin,
    )


def _materialize_assignment(
    initial_assigned: dict[int, tuple[Detection, ...]],
    links: tuple[tuple[int, int, int], ...],
    tracklet_lookup: dict[tuple[int, int], TrackletLike],
) -> dict[int, tuple[Detection, ...]]:
    result = dict(initial_assigned)
    for source_id, index, seed_id in links:
        tracklet = tracklet_lookup[(source_id, index)]
        relabeled = tuple(replace(row, object_id=seed_id) for row in tracklet.rows)
        result[seed_id] = (*result[seed_id], *relabeled)
    return result


def _drop_penalty(tracklet: TrackletLike, *, drop_cost: float) -> float:
    confidences = [min(1.0, max(0.0, row.confidence)) for row in tracklet.rows]
    mean_confidence = sum(confidences) / len(confidences)
    evidence = math.sqrt(len(tracklet.rows)) * max(0.1, mean_confidence)
    return drop_cost * evidence


def _prune(states: list[_BeamState], *, beam_width: int) -> list[_BeamState]:
    unique: dict[tuple[tuple[int, int, int], ...], _BeamState] = {}
    for state in states:
        current = unique.get(state.links)
        if current is None or _state_rank(state) < _state_rank(current):
            unique[state.links] = state
    return sorted(unique.values(), key=_state_rank)[:beam_width]


def _state_rank(state: _BeamState) -> tuple[float, int, tuple[tuple[int, int, int], ...]]:
    return (state.cost, -state.linked_rows, state.links)
