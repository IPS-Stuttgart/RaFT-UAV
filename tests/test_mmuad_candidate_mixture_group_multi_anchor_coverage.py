from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from raft_uav.mmuad.candidate_mixture_group_anchor_mass_topk import (
    AnchorConditioningConfig,
)
from raft_uav.mmuad.candidate_mixture_group_mass_topk import (
    PosteriorMassGroupTopKConfig,
)
from raft_uav.mmuad.candidate_mixture_group_multi_anchor_coverage import (
    COVERAGE_RESCUED,
    AnchorGroupCoverageConfig,
    main as coverage_main,
    select_multi_anchor_coverage_hypothesis_group_topk,
)
from raft_uav.mmuad.candidate_mixture_map import CandidateMixtureMapConfig
from raft_uav.mmuad.candidate_mixture_map_grouped import HypothesisGroupConfig


def _candidates() -> pd.DataFrame:
    records = []
    for time_s in range(3):
        records.extend(
            [
                {
                    "sequence_id": "seqA",
                    "time_s": float(time_s),
                    "source": "lidar_360",
                    "track_id": f"left-{time_s}",
                    "origin_row": f"left-origin-{time_s}",
                    "candidate_branch": "raw",
                    "x_m": float(time_s),
                    "y_m": 0.0,
                    "z_m": 1.0,
                    "ranker_score": 0.0,
                    "predicted_sigma_m": 1.0,
                },
                {
                    "sequence_id": "seqA",
                    "time_s": float(time_s),
                    "source": "livox_avia",
                    "track_id": f"middle-{time_s}",
                    "origin_row": f"middle-origin-{time_s}",
                    "candidate_branch": "dynamic",
                    "x_m": float(time_s + 5),
                    "y_m": 0.0,
                    "z_m": 1.0,
                    "ranker_score": 1.0,
                    "predicted_sigma_m": 1.0,
                },
                {
                    "sequence_id": "seqA",
                    "time_s": float(time_s),
                    "source": "radar_enhance_pcl",
                    "track_id": f"right-{time_s}",
                    "origin_row": f"right-origin-{time_s}",
                    "candidate_branch": "translated",
                    "x_m": float(time_s + 10),
                    "y_m": 0.0,
                    "z_m": 1.0,
                    "ranker_score": 0.0,
                    "predicted_sigma_m": 1.0,
                },
            ]
        )
    return pd.DataFrame.from_records(records)


def _anchor(offset_m: float) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Sequence": ["seqA", "seqA", "seqA"],
            "Timestamp": [0.0, 1.0, 2.0],
            "x_m": [offset_m, 1.0 + offset_m, 2.0 + offset_m],
            "y_m": [0.0, 0.0, 0.0],
            "z_m": [1.0, 1.0, 1.0],
        }
    )


def _mixture_config() -> CandidateMixtureMapConfig:
    return CandidateMixtureMapConfig(
        top_k=0,
        score_column="ranker_score",
        fallback_score_columns=(),
        sigma_column="predicted_sigma_m",
        score_normalization="minmax",
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
        target_posterior_mass=0.95,
        posterior_temperature=1.0,
        uniform_posterior_blend=0.0,
        max_siblings_per_group=1,
        group_score_mode="max",
        diversity_weight=0.0,
        diversity_scale_m=5.0,
        diversity_cap_m=30.0,
    )


def _anchor_config() -> AnchorConditioningConfig:
    return AnchorConditioningConfig(
        anchor_selection_weight=5.0,
        anchor_scale_m=1.0,
        anchor_huber_delta=1.0,
        anchor_cost_cap=10.0,
        anchor_time_tolerance_s=0.01,
    )


def test_anchor_coverage_rescues_mode_pruned_by_fixed_group_budget() -> None:
    _, selected, _, frames, summary = (
        select_multi_anchor_coverage_hypothesis_group_topk(
            _candidates(),
            anchor_estimates={"left": _anchor(0.0), "right": _anchor(10.0)},
            mixture_config=_mixture_config(),
            group_config=HypothesisGroupConfig(group_column="origin_row"),
            selection_config=_selection_config(),
            anchor_config=_anchor_config(),
            coverage_config=AnchorGroupCoverageConfig(
                max_anchor_distance_m=1.0,
                max_extra_groups_per_frame=2,
                max_siblings_per_rescued_group=1,
            ),
        )
    )

    selected_prefix = selected["track_id"].astype(str).str.split("-").str[0]
    assert set(selected_prefix) == {"left", "right"}
    assert int(selected[COVERAGE_RESCUED].sum()) == 3
    assert (frames["rescued_groups"] == 1).all()
    coverage = summary["anchor_group_coverage"]
    assert coverage["rescued_group_count"] == 3
    assert coverage["frames_with_rescue"] == 3
    assert summary["truth_used_for_selection"] is False


def test_anchor_coverage_respects_extra_group_budget() -> None:
    _, selected, _, frames, summary = (
        select_multi_anchor_coverage_hypothesis_group_topk(
            _candidates(),
            anchor_estimates={"left": _anchor(0.0), "right": _anchor(10.0)},
            mixture_config=_mixture_config(),
            group_config=HypothesisGroupConfig(group_column="origin_row"),
            selection_config=_selection_config(),
            anchor_config=_anchor_config(),
            coverage_config=AnchorGroupCoverageConfig(
                max_anchor_distance_m=1.0,
                max_extra_groups_per_frame=0,
                max_siblings_per_rescued_group=1,
            ),
        )
    )

    assert len(selected) == 3
    assert not selected[COVERAGE_RESCUED].any()
    assert frames.empty
    assert summary["anchor_group_coverage"]["rescued_candidate_rows"] == 0


def test_anchor_coverage_rejects_nonfinite_distance() -> None:
    with pytest.raises(ValueError, match="max_anchor_distance_m"):
        select_multi_anchor_coverage_hypothesis_group_topk(
            _candidates(),
            anchor_estimates={"left": _anchor(0.0)},
            coverage_config=AnchorGroupCoverageConfig(max_anchor_distance_m=float("nan")),
        )


def test_anchor_coverage_cli_writes_diagnostics(tmp_path: Path) -> None:
    candidates_csv = tmp_path / "candidates.csv"
    left_csv = tmp_path / "left.csv"
    right_csv = tmp_path / "right.csv"
    output_dir = tmp_path / "out"
    _candidates().to_csv(candidates_csv, index=False)
    _anchor(0.0).to_csv(left_csv, index=False)
    _anchor(10.0).to_csv(right_csv, index=False)

    status = coverage_main(
        [
            "--candidates-csv",
            str(candidates_csv),
            "--anchor-csv",
            f"left={left_csv}",
            "--anchor-csv",
            f"right={right_csv}",
            "--output-dir",
            str(output_dir),
            "--min-group-top-k",
            "1",
            "--max-group-top-k",
            "1",
            "--max-siblings-per-group",
            "1",
            "--diversity-weight",
            "0",
            "--anchor-selection-weight",
            "5",
            "--anchor-scale-m",
            "1",
            "--anchor-cost-cap",
            "10",
            "--anchor-time-tolerance-s",
            "0.01",
            "--anchor-coverage-max-distance-m",
            "1",
            "--anchor-coverage-max-extra-groups-per-frame",
            "2",
            "--score-column",
            "ranker_score",
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
    assert (output_dir / "mmuad_multi_anchor_coverage_frames.csv").exists()
    selected_path = output_dir / "mmuad_multi_anchor_coverage_selected_candidates.csv"
    assert selected_path.exists()
    selected = pd.read_csv(selected_path)
    assert int(selected[COVERAGE_RESCUED].sum()) == 3
    payload = json.loads(
        (output_dir / "mmuad_multi_anchor_coverage_summary.json").read_text(
            encoding="utf-8"
        )
    )
    assert payload["anchor_group_coverage"]["rescued_group_count"] == 3
    assert (output_dir / "mmuad_candidate_mixture_estimates.csv").exists()
