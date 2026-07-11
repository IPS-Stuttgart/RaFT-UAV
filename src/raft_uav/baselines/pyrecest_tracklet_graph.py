"""RaFT-UAV adapter for PyRecEst generic tracklet-graph utilities.

Fortem/AERPAW-specific segment construction, quarantine flags, RF contradiction
features, and CLI parameters stay in RaFT-UAV.  Generic DAG and k-best path
enumeration is delegated to :mod:`pyrecest.tracking.tracklet_graph`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

try:
    import pandas as pd
except ImportError:  # pragma: no cover - pandas is available in normal RaFT-UAV runs.
    pd = None
try:
    from pyrecest.tracking import (
        Tracklet,
        TrackletGraphConfig,
        TrackletPath,
        constant_velocity_edge_cost,
        diverse_k_best_tracklet_paths,
        k_best_tracklet_paths,
        tracklet_paths_to_dicts,
    )
except ImportError:

    @dataclass(frozen=True)
    class Tracklet:
        """Fallback tracklet container matching the PyRecEst fields used here."""

        id: Any
        start_time: float
        end_time: float
        start_state: np.ndarray
        end_state: np.ndarray
        cost: float = 0.0
        metadata: Mapping[str, Any] | None = None

    @dataclass(frozen=True)
    class TrackletGraphConfig:
        """Fallback k-best graph search configuration."""

        top_k: int = 8
        beam_width: int | None = None
        max_gap: float = 30.0
        diversity_weight: float = 0.0
        candidate_multiplier: int = 5

    @dataclass(frozen=True)
    class TrackletPath:
        """Fallback tracklet path result."""

        tracklet_ids: tuple[Any, ...]
        cost: float

        @property
        def length(self) -> int:
            return len(self.tracklet_ids)

    def constant_velocity_edge_cost(
        *,
        max_gap: float,
        max_speed: float,
        state_slice: Sequence[int],
        gap_weight: float,
        speed_weight: float,
        switch_metadata_key: str,
        switch_penalty: float,
    ):
        """Return a fallback Fortem-compatible transition cost function."""

        def edge_cost(left: Tracklet, right: Tracklet) -> float | None:
            gap = float(right.start_time) - float(left.end_time)
            if gap < 0.0 or gap > float(max_gap):
                return None
            start = np.asarray(right.start_state, dtype=float)[list(state_slice)]
            end = np.asarray(left.end_state, dtype=float)[list(state_slice)]
            distance = float(np.linalg.norm(start - end))
            speed = distance / max(gap, 1e-12)
            if speed > float(max_speed):
                return None
            cost = float(gap_weight) * gap + float(speed_weight) * speed
            left_track = (left.metadata or {}).get(switch_metadata_key)
            right_track = (right.metadata or {}).get(switch_metadata_key)
            if left_track is not None and right_track is not None and left_track != right_track:
                cost += float(switch_penalty)
            return float(cost)

        return edge_cost

    def k_best_tracklet_paths(
        tracklets: Sequence[Tracklet],
        *,
        edge_cost_fn,
        config: TrackletGraphConfig,
        node_cost_fn,
    ) -> list[TrackletPath]:
        """Enumerate simple fallback DAG paths ordered by cost."""

        ordered = sorted(tracklets, key=lambda item: (float(item.start_time), float(item.end_time), str(item.id)))
        by_id = {tracklet.id: tracklet for tracklet in ordered}
        paths: list[TrackletPath] = []

        def extend(path: tuple[Any, ...], cost: float) -> None:
            paths.append(TrackletPath(path, float(cost)))
            last = by_id[path[-1]]
            for candidate in ordered:
                if candidate.id in path:
                    continue
                transition = edge_cost_fn(last, candidate)
                if transition is None:
                    continue
                extend((*path, candidate.id), cost + float(transition) + float(node_cost_fn(candidate)))

        for tracklet in ordered:
            extend((tracklet.id,), float(node_cost_fn(tracklet)))
        return sorted(paths, key=lambda path: (path.cost, tuple(str(item) for item in path.tracklet_ids)))[: int(config.top_k)]

    def diverse_k_best_tracklet_paths(
        tracklets: Sequence[Tracklet],
        *,
        edge_cost_fn,
        config: TrackletGraphConfig,
        node_cost_fn,
    ) -> list[TrackletPath]:
        """Fallback diverse search delegates to k-best enumeration."""

        return k_best_tracklet_paths(
            tracklets,
            edge_cost_fn=edge_cost_fn,
            config=config,
            node_cost_fn=node_cost_fn,
        )

    def tracklet_paths_to_dicts(
        paths: Sequence[TrackletPath],
        *,
        tracklets: Mapping[Any, Tracklet] | None = None,
    ) -> list[dict[str, Any]]:
        """Return fallback path rows using PyRecEst-compatible column names."""

        return [
            {
                "rank": rank,
                "tracklet_ids": ";".join(str(item) for item in path.tracklet_ids),
                "cost": float(path.cost),
            }
            for rank, path in enumerate(paths)
        ]


@dataclass(frozen=True)
class FortemTrackletGraphConfig:
    """RaFT-UAV defaults for Fortem tracklet DAG enumeration."""

    top_k_paths: int = 8
    beam_width: int | None = None
    max_link_gap_s: float = 30.0
    max_transition_speed_mps: float = 65.0
    switch_penalty: float = 25.0
    gap_weight: float = 0.01
    speed_weight: float = 1.0
    coverage_reward_per_row: float = 0.001
    use_diverse_paths: bool = False
    diversity_weight: float = 500.0
    diversity_oversample: int = 5

    def pyrecest_config(self) -> TrackletGraphConfig:
        """Return the corresponding PyRecEst graph-search config."""

        return TrackletGraphConfig(
            top_k=int(self.top_k_paths),
            beam_width=self.beam_width,
            max_gap=float(self.max_link_gap_s),
            diversity_weight=float(self.diversity_weight) if self.use_diverse_paths else 0.0,
            candidate_multiplier=int(self.diversity_oversample),
        )


def fortem_tracklet_from_summary(
    row: Mapping[str, Any],
    *,
    cost_key: str = "score",
) -> Tracklet:
    """Convert a RaFT-UAV segment-diagnostic row to a PyRecEst ``Tracklet``.

    The row must contain segment id, time span, endpoint ENU coordinates, and
    usually a ``track_id``.  The conversion is structural and works with plain
    dictionaries, pandas rows, or CSV-derived mappings.
    """

    segment_id = _first_present(row, "segment_id", "micro_segment_id", "id")
    track_id = _optional(row, "track_id", default=None)
    start_state = np.array(
        [
            float(_first_present(row, "start_east_m", "first_east_m")),
            float(_first_present(row, "start_north_m", "first_north_m")),
            float(_first_present(row, "start_up_m", "first_up_m")),
        ],
        dtype=float,
    )
    end_state = np.array(
        [
            float(_first_present(row, "end_east_m", "last_east_m")),
            float(_first_present(row, "end_north_m", "last_north_m")),
            float(_first_present(row, "end_up_m", "last_up_m")),
        ],
        dtype=float,
    )
    cost_value = _optional(row, cost_key, default=0.0)
    metadata = {str(key): _jsonable(value) for key, value in row.items()}
    if track_id is not None:
        metadata["track_id"] = track_id
    return Tracklet(
        id=_jsonable(segment_id),
        start_time=float(_first_present(row, "start_time_s", "time_start_s")),
        end_time=float(_first_present(row, "end_time_s", "time_end_s")),
        start_state=start_state,
        end_state=end_state,
        cost=0.0 if cost_value is None else float(cost_value),
        metadata=metadata,
    )


def fortem_tracklet_from_rows(
    segment_id: Any,
    rows: Sequence[Mapping[str, Any]],
    *,
    track_id: Any | None = None,
    cost: float = 0.0,
) -> Tracklet:
    """Create a PyRecEst ``Tracklet`` from ordered Fortem row dictionaries."""

    if not rows:
        raise ValueError("rows must contain at least one record")
    ordered = sorted(rows, key=lambda item: float(item["time_s"]))
    first = ordered[0]
    last = ordered[-1]
    resolved_track_id = first.get("track_id") if track_id is None else track_id
    return Tracklet(
        id=_jsonable(segment_id),
        start_time=float(first["time_s"]),
        end_time=float(last["time_s"]),
        start_state=np.array(
            [float(first["east_m"]), float(first["north_m"]), float(first["up_m"])],
            dtype=float,
        ),
        end_state=np.array(
            [float(last["east_m"]), float(last["north_m"]), float(last["up_m"])],
            dtype=float,
        ),
        cost=float(cost),
        metadata={"track_id": resolved_track_id, "rows": len(ordered)},
    )


def rank_fortem_tracklet_paths(
    tracklets: Sequence[Tracklet],
    *,
    config: FortemTrackletGraphConfig = FortemTrackletGraphConfig(),
) -> list[TrackletPath]:
    """Enumerate top-k Fortem tracklet paths via PyRecEst."""

    edge_cost = _fortem_edge_cost(config)
    node_cost = _fortem_node_cost(config)
    search_config = config.pyrecest_config()
    if config.use_diverse_paths:
        return diverse_k_best_tracklet_paths(
            tracklets,
            edge_cost_fn=edge_cost,
            config=search_config,
            node_cost_fn=node_cost,
        )
    return k_best_tracklet_paths(
        tracklets,
        edge_cost_fn=edge_cost,
        config=search_config,
        node_cost_fn=node_cost,
    )


def fortem_tracklet_paths_to_rows(
    paths: Sequence[TrackletPath],
    *,
    tracklets: Mapping[Any, Tracklet] | None = None,
) -> list[dict[str, Any]]:
    """Return RaFT-UAV-compatible path diagnostics."""

    rows = tracklet_paths_to_dicts(paths, tracklets=tracklets)
    for row, path in zip(rows, paths, strict=True):
        row["path_id"] = row.pop("rank")
        row["segment_ids"] = row.pop("tracklet_ids")
        row["graph_cost"] = row["cost"]
        row["coverage_segments"] = path.length
    return rows


def _fortem_node_cost(config: FortemTrackletGraphConfig):
    def node_cost(tracklet: Tracklet) -> float:
        metadata = tracklet.metadata if tracklet.metadata is not None else {}
        rows = float(metadata.get("rows", metadata.get("row_count", 1.0)) or 1.0)
        return float(tracklet.cost) - float(config.coverage_reward_per_row) * rows

    return node_cost


def _fortem_edge_cost(config: FortemTrackletGraphConfig):
    return constant_velocity_edge_cost(
        max_gap=float(config.max_link_gap_s),
        max_speed=float(config.max_transition_speed_mps),
        state_slice=[0, 1, 2],
        gap_weight=float(config.gap_weight),
        speed_weight=float(config.speed_weight),
        switch_metadata_key="track_id",
        switch_penalty=float(config.switch_penalty),
    )


def _first_present(row: Mapping[str, Any], *names: str) -> Any:
    for name in names:
        if name in row:
            value = row[name]
            if not _is_missing_scalar(value):
                return value
    raise KeyError(f"row must contain one of {names!r}")


def _optional(row: Mapping[str, Any], name: str, *, default: Any = None) -> Any:
    value = row.get(name, default)
    return default if _is_missing_scalar(value) else value


def _jsonable(value: Any) -> Any:
    if _is_missing_scalar(value):
        return None
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _is_missing_scalar(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    if pd is not None:
        try:
            missing = pd.isna(value)
        except (TypeError, ValueError):
            return False
        if isinstance(missing, (bool, np.bool_)):
            return bool(missing)
        return False
    try:
        return bool(value != value)
    except (TypeError, ValueError):
        return False


__all__ = [
    "FortemTrackletGraphConfig",
    "fortem_tracklet_from_rows",
    "fortem_tracklet_from_summary",
    "fortem_tracklet_paths_to_rows",
    "rank_fortem_tracklet_paths",
]
