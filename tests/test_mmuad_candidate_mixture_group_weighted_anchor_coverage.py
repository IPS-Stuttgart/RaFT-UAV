from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from raft_uav.mmuad.candidate_mixture_group_anchor_mass_topk import (
    AnchorConditioningConfig,
)
from raft_uav.mmuad.candidate_mixture_group_mass_topk import (
    PosteriorMassGroupTopKConfig,
)
from raft_uav.mmuad.candidate_mixture_group_weighted_anchor_coverage import (
    PRIORITY_RESCUED,
    PRIORITY_RESCUE_ANCHORS,
    PRIORITY_RESCUE_WEIGHT,
    ReliabilityPriorityCoverageConfig,
    main as priority_coverage_main,
    select_reliability_priority_anchor_coverage,
)
from raft_uav.mmuad.candidate_mixture_group_weighted_anchor_quantile import (
    WeightedAnchorQuantileConfig,
)
from raft_uav.mmuad.candidate_mixture_map import CandidateMixtureMapConfig
from raft_uav.mmuad.candidate_mixture_map_grouped import HypothesisGroupConfig


def _candidates(frame_count: int = 1) -> pd.DataFrame:
    records = []
    for time_s in range(frame_count):
        records.extend(
            [
                {
                    "sequence_id": "seqA",
                    "time_s": float(time_s),
                    "source": "lidar_360",
                    "track_id": f"base-{time_s}",
                    "origin_row": f"base-{time_s}",
                    "candidate_branch": "raw",
                    "x_m": float(time_s),
                    "y_m": 0.0,
                    "z_m": 1.0,
                    "ranker_score": 10.0,
                    "predicted_sigma_m": 1.0,
                },
                {
                    "sequence_id": "seqA",
                    "time_s": float(time_s),
                    "source": "livox_avia",
                    "track_id": f"weak-{time_s}",
                    "origin_row": f"weak-{time_s}",
                    "candidate_branch": "translated",
                    "x_m": float(time_s + 10.0),
                    "y_m": 0.0,
                    "z_m": 1.0,
                    "ranker_score": 0.0,
                    "predicted_sigma_m": 1.0,
                },
                {
                    "sequence_id": "seqA",
                    "time_s": float(time_s),
                    "source": "dynamic",
                    "track_id": f"strong-{time_s}",
                    "origin_row": f"strong-{time_s}",
                    "candidate_branch": "dynamic",
                    "x_m": float(time_s + 20.0),
                    "y_m": 0.0,
                    "z_m": 1.0,
                    "ranker_score": 0.0,
                    "predicted_sigma_m": 1.0,
                },
            ]
        )
    return pd.DataFrame.from_records(records)


def _anchor(offset_m: float, frame_count: int = 1) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Sequence": ["seqA"] * frame_count,
            "Timestamp": [float(value) for value in range(frame_count)],
            "x_m": [float(value) + offset_m for value in range(frame_count)],
            "y_m": [0.0] * frame_count,
            "z_m": [1.0] * frame_count,
        }
    )


def _mixture_config() -> CandidateMixtureMapConfig:
    return CandidateMixtureMapConfig(
        top_k=0,
        score_column="ranker_score",
        fallback_score_columns=(),
        sigma_column="predicted_sigma_m",
        score_normalization="none",
        score_weight=1.0,
        temperature=1.0,
        sigma_log_weight=0.0,
        smoothness_weight=0.0,
        iterations=1,
    )


def _selection_config() -> PosteriorMassGroupTopKConfig:
    return PosteriorMassGroupTopKConfig(
        min_group_top_k=1,
        max_group_top_k=1,
        target_posterior_mass=0.5,
        posterior_temperature=1.0,
        uniform_posterior_blend=0.0,
        max_siblings_per_group=1,
        diversity_weight=0.0,
    )


def test_reliability_priority_rescues_strong_anchor_before_weak_anchor() -> None:
    _, selected, _, frames, summary = select_reliability_priority_anchor_coverage(
        _candidates(),
        anchor_estimates={"weak": _anchor(10.0), "strong": _anchor(20.0)},
        anchor_reliability={"weak": 1.0, "strong": 10.0},
        mixture_config=_mixture_config(),
        group_config=HypothesisGroupConfig(group_column="origin_row"),
        selection_config=_selection_config(),
        anchor_config=AnchorConditioningConfig(
            anchor_selection_weight=0.0,
            anchor_scale_m=1.0,
            anchor_time_tolerance_s=0.01,
        ),
        quantile_config=WeightedAnchorQuantileConfig(cost_quantile=0.5),
        coverage_config=ReliabilityPriorityCoverageConfig(
            max_anchor_distance_m=1.0,
            max_extra_groups_per_frame=1,
            max_siblings_per_rescued_group=1,
            priority_mode="reliability-distance",
            distance_scale_m=1.0,
        ),
    )

    track_ids = set(selected["track_id"].astype(str))
    assert track_ids == {"base-0", "strong-0"}
    rescued = selected.loc[selected[PRIORITY_RESCUED]].iloc[0]
    assert rescued[PRIORITY_RESCUE_ANCHORS] == "strong"
    assert rescued[PRIORITY_RESCUE_WEIGHT] == 10.0
    assert frames.loc[0, "anchors_blocked_by_budget"] == 1
    assert frames.loc[0, "weight_blocked_by_budget"] == 1.0
    assert summary["total_covered_anchors_by_rescue"] == 1
    assert summary["truth_used_for_coverage"] is False


def test_priority_coverage_records_all_anchors_supporting_same_group() -> None:
    _, selected, _, frames, _ = select_reliability_priority_anchor_coverage(
        _candidates(),
        anchor_estimates={"first": _anchor(20.0), "second": _anchor(20.0)},
        anchor_reliability={"first": 2.0, "second": 3.0},
        mixture_config=_mixture_config(),
        group_config=HypothesisGroupConfig(group_column="origin_row"),
        selection_config=_selection_config(),
        anchor_config=AnchorConditioningConfig(
            anchor_selection_weight=0.0,
            anchor_scale_m=1.0,
            anchor_time_tolerance_s=0.01,
        ),
        coverage_config=ReliabilityPriorityCoverageConfig(
            max_anchor_distance_m=1.0,
            max_extra_groups_per_frame=1,
        ),
    )

    rescued = selected.loc[selected[PRIORITY_RESCUED]].iloc[0]
    assert rescued[PRIORITY_RESCUE_ANCHORS] == "first;second"
    assert rescued[PRIORITY_RESCUE_WEIGHT] == 5.0
    assert frames.loc[0, "covered_anchors_by_rescue"] == 2
    assert frames.loc[0, "anchors_blocked_by_budget"] == 0


def test_priority_coverage_cli_writes_diagnostics(tmp_path: Path) -> None:
    candidates_csv = tmp_path / "candidates.csv"
    weak_csv = tmp_path / "weak.csv"
    strong_csv = tmp_path / "strong.csv"
    output_dir = tmp_path / "out"
    _candidates(frame_count=3).to_csv(candidates_csv, index=False)
    _anchor(10.0, frame_count=3).to_csv(weak_csv, index=False)
    _anchor(20.0, frame_count=3).to_csv(strong_csv, index=False)

    status = priority_coverage_main(
        [
            "--candidates-csv",
            str(candidates_csv),
            "--anchor-csv",
            f"weak={weak_csv}",
            "--anchor-csv",
            f"strong={strong_csv}",
            "--anchor-reliability",
            "weak=1",
            "--anchor-reliability",
            "strong=10",
            "--output-dir",
            str(output_dir),
            "--min-group-top-k",
            "1",
            "--max-group-top-k",
            "1",
            "--target-posterior-mass",
            "0.5",
            "--uniform-posterior-blend",
            "0",
            "--max-siblings-per-group",
            "1",
            "--diversity-weight",
            "0",
            "--anchor-selection-weight",
            "0",
            "--anchor-time-tolerance-s",
            "0.01",
            "--anchor-cost-quantile",
            "0.5",
            "--anchor-coverage-max-distance-m",
            "1",
            "--anchor-coverage-max-extra-groups-per-frame",
            "1",
            "--anchor-coverage-priority-mode",
            "reliability-distance",
            "--anchor-coverage-distance-scale-m",
            "1",
            "--score-column",
            "ranker_score",
            "--score-normalization",
            "none",
            "--score-weight",
            "1",
            "--sigma-log-weight",
            "0",
            "--smoothness-weight",
            "0",
            "--iterations",
            "1",
            "--hypothesis-group-column",
            "origin_row",
        ]
    )

    assert status == 0
    summary_path = output_dir / "mmuad_weighted_anchor_coverage_summary.json"
    selected_path = output_dir / "mmuad_weighted_anchor_coverage_selected_candidates.csv"
    frames_path = output_dir / "mmuad_weighted_anchor_coverage_frames.csv"
    assert summary_path.exists()
    assert selected_path.exists()
    assert frames_path.exists()
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert payload["coverage_config"]["priority_mode"] == "reliability-distance"
    assert payload["rescued_group_count"] == 3
    selected = pd.read_csv(selected_path)
    rescued = selected.loc[selected[PRIORITY_RESCUED].astype(bool)]
    assert set(rescued["track_id"].astype(str)) == {"strong-0", "strong-1", "strong-2"}
    assert (output_dir / "mmuad_candidate_mixture_estimates.csv").exists()
