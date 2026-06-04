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

try:
    from pyrecest.tracking import (
        HypothesisReplay,
        InnovationConsistencyScoreConfig,
        rank_hypothesis_replays,
        scores_to_dicts,
    )
except ImportError:

    @dataclass(frozen=True)
    class HypothesisReplay:
        """Fallback replay container matching PyRecEst's public fields."""

        hypothesis_id: Hashable
        records: Sequence[Any]
        graph_cost: float = 0.0
        track_switches: int = 0
        missed_detection_count: int = 0
        rejected_count: int = 0
        coast_count: int = 0
        unsupported_measurement_count: int = 0
        hard_quarantine_count: int = 0
        tail_duration_s: float = 0.0
        coverage_count: int = 0
        metadata: Mapping[str, Any] | None = None

    @dataclass(frozen=True)
    class InnovationConsistencyScoreConfig:
        """Fallback score-weight container compatible with PyRecEst."""

        graph_cost_weight: float = 1.0
        nis_weight: float = 1.0
        residual_weight: float = 0.01
        switch_weight: float = 2.0
        missed_detection_weight: float = 1.0
        rejected_weight: float = 0.25
        coast_weight: float = 0.25
        unsupported_measurement_weight: float = 5.0
        hard_quarantine_weight: float = 1000.0
        tail_duration_weight: float = 0.05
        coverage_reward: float = 0.001
        nis_clip: float = 50.0
        residual_clip: float = 500.0
        residual_normalizer: float = 100.0

    def rank_hypothesis_replays(
        replays: Sequence[HypothesisReplay],
        *,
        config: InnovationConsistencyScoreConfig,
    ) -> list[dict[str, Any]]:
        """Return fallback innovation-consistency scores sorted by objective."""

        return sorted(
            (_score_replay(replay, config) for replay in replays),
            key=lambda row: (float(row["total_score"]), str(row["hypothesis_id"])),
        )

    def scores_to_dicts(scores: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        """Return score dictionaries in the shape provided by PyRecEst."""

        return [dict(score) for score in scores]

    def _score_replay(
        replay: HypothesisReplay,
        config: InnovationConsistencyScoreConfig,
    ) -> dict[str, Any]:
        nis_score = sum(
            min(_record_float(record, ("nis", "normalized_innovation_squared"), 0.0), config.nis_clip)
            for record in replay.records
        )
        residual_score = sum(
            min(
                _record_float(record, ("residual_norm_m", "residual_norm"), 0.0)
                / max(float(config.residual_normalizer), 1e-12),
                config.residual_clip / max(float(config.residual_normalizer), 1e-12),
            )
            for record in replay.records
        )
        total_score = (
            float(config.graph_cost_weight) * float(replay.graph_cost)
            + float(config.nis_weight) * float(nis_score)
            + float(config.residual_weight) * float(residual_score)
            + float(config.switch_weight) * int(replay.track_switches)
            + float(config.missed_detection_weight) * int(replay.missed_detection_count)
            + float(config.rejected_weight) * int(replay.rejected_count)
            + float(config.coast_weight) * int(replay.coast_count)
            + float(config.unsupported_measurement_weight) * int(replay.unsupported_measurement_count)
            + float(config.hard_quarantine_weight) * int(replay.hard_quarantine_count)
            + float(config.tail_duration_weight) * float(replay.tail_duration_s)
            - float(config.coverage_reward) * int(replay.coverage_count)
        )
        row: dict[str, Any] = {
            "hypothesis_id": replay.hypothesis_id,
            "total_score": float(total_score),
            "graph_cost": float(replay.graph_cost),
            "innovation_score": float(nis_score),
            "residual_score": float(residual_score),
            "track_switches": int(replay.track_switches),
            "missed_detection_count": int(replay.missed_detection_count),
            "rejected_count": int(replay.rejected_count),
            "coast_count": int(replay.coast_count),
            "unsupported_measurement_count": int(replay.unsupported_measurement_count),
            "hard_quarantine_count": int(replay.hard_quarantine_count),
            "tail_duration_s": float(replay.tail_duration_s),
            "coverage_count": int(replay.coverage_count),
        }
        for key, value in (replay.metadata or {}).items():
            row[f"metadata_{key}"] = value
        return row

    def _record_float(record: Any, keys: tuple[str, ...], default: float) -> float:
        if isinstance(record, Mapping):
            for key in keys:
                if key in record and record[key] is not None:
                    return float(record[key])
        return float(default)


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
