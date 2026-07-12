from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from raft_uav.mmuad.candidate_mixture_group_anchor_mass_topk import (
    AnchorConditioningConfig,
    main as anchor_mass_main,
    select_anchor_posterior_mass_hypothesis_group_topk,
)
from raft_uav.mmuad.candidate_mixture_group_mass_topk import (
    PosteriorMassGroupTopKConfig,
    select_posterior_mass_hypothesis_group_topk,
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
                    "track_id": f"good-{time_s}",
                    "origin_row": f"good-origin-{time_s}",
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
                    "track_id": f"bad-{time_s}",
                    "origin_row": f"bad-origin-{time_s}",
                    "candidate_branch": "translated",
                    "x_m": float(time_s + 10),
                    "y_m": 0.0,
                    "z_m": 1.0,
                    "ranker_score": 1.0,
                    "predicted_sigma_m": 1.0,
                },
            ]
        )
    return pd.DataFrame.from_records(records)


def _anchor_alias_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Sequence": ["seqA", "seqA", "seqA"],
            "Timestamp": [0.0, 1.0, 2.0],
            "x_m": [0.0, 1.0, 2.0],
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


def test_anchor_conditioning_preserves_low_score_anchor_coherent_groups() -> None:
    selected, anchors, summary = select_anchor_posterior_mass_hypothesis_group_topk(
        _candidates(),
        initial_estimates=_anchor_alias_rows(),
        mixture_config=_mixture_config(),
        group_config=HypothesisGroupConfig(group_column="origin_row"),
        selection_config=_selection_config(),
        anchor_config=AnchorConditioningConfig(
            anchor_selection_weight=5.0,
            anchor_scale_m=1.0,
            anchor_huber_delta=1.0,
            anchor_cost_cap=4.0,
            anchor_time_tolerance_s=0.01,
        ),
    )

    assert anchors["sequence_id"].tolist() == ["seqA", "seqA", "seqA"]
    assert selected["track_id"].astype(str).str.startswith("good-").all()
    assert selected["mixture_anchor_matched"].all()
    assert summary["anchor_conditioning"]["matched_frame_fraction"] == 1.0
    assert summary["truth_used_for_selection"] is False


def test_zero_anchor_weight_matches_state_independent_mass_selector() -> None:
    baseline, _ = select_posterior_mass_hypothesis_group_topk(
        _candidates(),
        mixture_config=_mixture_config(),
        group_config=HypothesisGroupConfig(group_column="origin_row"),
        selection_config=_selection_config(),
    )
    selected, _, _ = select_anchor_posterior_mass_hypothesis_group_topk(
        _candidates(),
        initial_estimates=_anchor_alias_rows(),
        mixture_config=_mixture_config(),
        group_config=HypothesisGroupConfig(group_column="origin_row"),
        selection_config=_selection_config(),
        anchor_config=AnchorConditioningConfig(anchor_selection_weight=0.0),
    )

    assert selected["track_id"].astype(str).tolist() == baseline["track_id"].astype(str).tolist()


def test_anchor_mass_cli_writes_selection_and_grouped_outputs(tmp_path: Path) -> None:
    candidates_csv = tmp_path / "candidates.csv"
    anchors_csv = tmp_path / "anchors.csv"
    output_dir = tmp_path / "out"
    _candidates().to_csv(candidates_csv, index=False)
    _anchor_alias_rows().to_csv(anchors_csv, index=False)

    status = anchor_mass_main(
        [
            "--candidates-csv",
            str(candidates_csv),
            "--initial-estimates-csv",
            str(anchors_csv),
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
            "--anchor-time-tolerance-s",
            "0.01",
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
    selected_path = output_dir / "mmuad_anchor_posterior_mass_group_topk_candidates.csv"
    summary_path = output_dir / "mmuad_anchor_posterior_mass_group_topk_summary.json"
    assert selected_path.exists()
    assert summary_path.exists()
    assert (output_dir / "mmuad_candidate_mixture_estimates.csv").exists()
    selected = pd.read_csv(selected_path)
    assert selected["track_id"].astype(str).str.startswith("good-").all()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["anchor_conditioning"]["matched_frame_count"] == 3
