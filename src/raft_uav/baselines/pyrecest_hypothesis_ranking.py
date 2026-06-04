"""RaFT-UAV adapter for PyRecEst replay-hypothesis ranking.

The global-tracklet branch builds UAV-specific radar/RF path hypotheses, but the
truth-free innovation-consistency scoring is generic.  Keep hypothesis scoring
in PyRecEst and use this module only to translate RaFT-UAV path-replay summaries
into PyRecEst's :mod:`pyrecest.tracking.hypothesis_replay` data classes.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Hashable

from pyrecest.tracking import (
    HypothesisReplay,
    InnovationConsistencyScoreConfig,
    rank_hypothesis_replays,
    scores_to_dicts,
)


@dataclass(frozen=True)
class GlobalTrackletHypothesisRankingConfig:
    """RaFT-UAV naming wrapper around PyRecEst innovation-ranking weights."""

    graph_cost_weight: float = 1.0
    replay_nis_weight: float = 1.0
    residual_weight: float = 0.01
    switch_weight: float = 2.0
    missed_radar_weight: float = 1.0
    rejected_measurement_weight: float = 0.25
    coast_weight: float = 0.25
    unsupported_rf_weight: float = 5.0
    hard_quarantine_weight: float = 1000.0
    tail_duration_weight: float = 0.05
    coverage_reward: float = 0.001
    nis_clip: float = 50.0
    residual_clip_m: float = 500.0
    residual_normalizer_m: float = 100.0

    def to_pyrecest(self) -> InnovationConsistencyScoreConfig:
        """Return the equivalent PyRecEst score configuration."""

        return InnovationConsistencyScoreConfig(
            graph_cost_weight=float(self.graph_cost_weight),
            nis_weight=float(self.replay_nis_weight),
            residual_weight=float(self.residual_weight),
            switch_weight=float(self.switch_weight),
            missed_detection_weight=float(self.missed_radar_weight),
            rejected_weight=float(self.rejected_measurement_weight),
            coast_weight=float(self.coast_weight),
            unsupported_measurement_weight=float(self.unsupported_rf_weight),
            hard_quarantine_weight=float(self.hard_quarantine_weight),
            tail_duration_weight=float(self.tail_duration_weight),
            coverage_reward=float(self.coverage_reward),
            nis_clip=float(self.nis_clip),
            residual_clip=float(self.residual_clip_m),
            residual_normalizer=float(self.residual_normalizer_m),
        )


def rank_global_tracklet_replays(
    path_replays: Iterable[Mapping[str, Any]],
    *,
    config: GlobalTrackletHypothesisRankingConfig | None = None,
) -> list[dict[str, Any]]:
    """Rank global-tracklet path replays using PyRecEst.

    Parameters
    ----------
    path_replays:
        Iterable of RaFT-UAV path-replay dictionaries.  Each dictionary may
        contain a ``records`` list of per-event update dictionaries plus graph
        fields such as ``path_id``, ``graph_cost``, ``track_switches``,
        ``missed_radar_count``, ``unsupported_rf_count``,
        ``hard_quarantined_segments_used``, ``tail_duration_s``, and
        ``selected_radar_rows``.
    config:
        RaFT-UAV score-weight wrapper.  The actual scoring is delegated to
        PyRecEst.

    Returns
    -------
    list[dict[str, Any]]
        Ranked dictionaries preserving the PyRecEst score diagnostics and using
        ``combined_objective`` as the backward-compatible RaFT-UAV score column.
    """

    cfg = GlobalTrackletHypothesisRankingConfig() if config is None else config
    replays = [_as_pyrecest_replay(item) for item in path_replays]
    ranked = rank_hypothesis_replays(replays, config=cfg.to_pyrecest())
    rows = scores_to_dicts(ranked)
    for row in rows:
        row["combined_objective"] = row["total_score"]
        row["path_id"] = row["hypothesis_id"]
    return rows


def _as_pyrecest_replay(item: Mapping[str, Any]) -> HypothesisReplay:
    return HypothesisReplay(
        hypothesis_id=_first_present(item, ("path_id", "hypothesis_id", "id"), default="path"),
        records=_records_from_item(item),
        graph_cost=_float_from_item(item, ("graph_cost", "path_graph_score"), default=0.0),
        track_switches=_int_from_item(item, ("track_switches",), default=0),
        missed_detection_count=_int_from_item(
            item,
            ("missed_radar_count", "missed_detection_count", "missed_radar_rows"),
            default=0,
        ),
        rejected_count=_int_from_item(
            item,
            ("rejected_count", "rejected_measurement_count"),
            default=0,
        ),
        coast_count=_int_from_item(item, ("coast_count", "coasts"), default=0),
        unsupported_measurement_count=_int_from_item(
            item,
            ("unsupported_rf_count", "rf_radar_unsupported_count", "unsupported_measurement_count"),
            default=0,
        ),
        hard_quarantine_count=_int_from_item(
            item,
            ("hard_quarantined_segments_used", "hard_quarantine_count"),
            default=0,
        ),
        tail_duration_s=_float_from_item(
            item,
            ("tail_duration_s", "seconds_after_last_supported_radar"),
            default=0.0,
        ),
        coverage_count=_int_from_item(item, ("selected_radar_rows", "coverage_count"), default=0),
        metadata=_metadata_from_item(item),
    )


def _records_from_item(item: Mapping[str, Any]) -> Sequence[Any]:
    records = item.get("records")
    if records is not None:
        return records  # type: ignore[return-value]
    if "nis_values" in item:
        nis_values = item.get("nis_values") or []
        residual_values = item.get("residual_values") or []
        rows = []
        for idx, nis in enumerate(nis_values):
            row: dict[str, Any] = {"nis": nis, "action": "updated"}
            if idx < len(residual_values):
                row["residual_norm_m"] = residual_values[idx]
            rows.append(row)
        return rows
    # Fallback for summary-only rows.  This keeps the adapter useful for CSV
    # summaries where only aggregate clipped sums were retained.
    return [
        {
            "nis": _float_from_item(
                item,
                ("robust_sum_nis", "sum_nis", "radar_nis_mean"),
                default=0.0,
            ),
            "residual_norm_m": _float_from_item(
                item,
                ("robust_sum_residual", "sum_residual", "max_residual_m"),
                default=0.0,
            ),
            "action": "updated",
        }
    ]


def _metadata_from_item(item: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "segment_ids",
        "track_ids",
        "selected_radar_track_ids",
        "top_k_paths",
        "beam_width",
        "rf_radar_support_policy",
        "split_max_horizontal_speed_mps",
    )
    return {key: item[key] for key in keys if key in item}


def _first_present(item: Mapping[str, Any], keys: tuple[str, ...], *, default: Any) -> Hashable:
    for key in keys:
        if key in item and item[key] is not None:
            value = item[key]
            if isinstance(value, Hashable):
                return value
            return str(value)
    return default


def _float_from_item(item: Mapping[str, Any], keys: tuple[str, ...], *, default: float) -> float:
    for key in keys:
        if key in item and item[key] is not None:
            return float(item[key])
    return float(default)


def _int_from_item(item: Mapping[str, Any], keys: tuple[str, ...], *, default: int) -> int:
    for key in keys:
        if key in item and item[key] is not None:
            return int(item[key])
    return int(default)
