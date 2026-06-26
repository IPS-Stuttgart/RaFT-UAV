from __future__ import annotations

import json

import pandas as pd

from raft_uav.mmuad.candidate_mixture_map import CandidateMixtureMapConfig
from raft_uav.mmuad.candidate_mixture_map_adaptive import (
    AdaptiveMixtureTopKConfig,
    main as adaptive_mixture_main,
    run_adaptive_candidate_mixture_map,
    select_adaptive_mixture_candidates,
)


def _ambiguity_rows() -> pd.DataFrame:
    records = []
    for candidate_index in range(5):
        records.append(
            {
                "sequence_id": "seqA",
                "time_s": 0.0,
                "source": "lidar_360" if candidate_index % 2 == 0 else "livox_avia",
                "candidate_branch": "raw" if candidate_index < 3 else "translated",
                "track_id": f"clear-{candidate_index}",
                "x_m": float(candidate_index),
                "y_m": 0.0,
                "z_m": 1.0,
                "ranker_score": 1.0 if candidate_index == 0 else 0.0,
                "predicted_sigma_m": 1.0,
                "branch_consensus_score": 1.0,
            }
        )
        records.append(
            {
                "sequence_id": "seqA",
                "time_s": 1.0,
                "source": "lidar_360" if candidate_index % 2 == 0 else "livox_avia",
                "candidate_branch": "raw" if candidate_index < 3 else "translated",
                "track_id": f"ambiguous-{candidate_index}",
                "x_m": float(candidate_index),
                "y_m": 1.0,
                "z_m": 1.0,
                "ranker_score": 0.5,
                "predicted_sigma_m": 30.0,
                "branch_consensus_score": 0.0,
            }
        )
    return pd.DataFrame.from_records(records)


def _trajectory_rows() -> pd.DataFrame:
    records = []
    for time_s in range(4):
        for candidate_index, offset in enumerate((0.0, 2.0, -3.0)):
            records.append(
                {
                    "sequence_id": "seqA",
                    "time_s": float(time_s),
                    "source": ("lidar_360", "livox_avia", "radar_enhance_pcl")[
                        candidate_index
                    ],
                    "candidate_branch": ("raw", "translated", "dynamic")[
                        candidate_index
                    ],
                    "track_id": f"{time_s}-{candidate_index}",
                    "x_m": float(time_s) + offset,
                    "y_m": 0.0,
                    "z_m": 1.0,
                    "ranker_score": (0.9, 0.4, 0.2)[candidate_index],
                    "predicted_sigma_m": (1.0, 5.0, 8.0)[candidate_index],
                    "branch_consensus_score": (1.0, 0.4, 0.1)[candidate_index],
                }
            )
    return pd.DataFrame.from_records(records)


def test_adaptive_selector_expands_k_for_ambiguous_frames() -> None:
    selected = select_adaptive_mixture_candidates(
        _ambiguity_rows(),
        config=AdaptiveMixtureTopKConfig(
            min_top_k=2,
            max_top_k=5,
            min_per_branch=0,
            min_per_source=0,
            score_column="ranker_score",
            score_temperature=0.05,
        ),
    )

    counts = selected.groupby("time_s").size().to_dict()
    chosen_k = selected.groupby("time_s")["mixture_adaptive_top_k"].first().to_dict()
    assert counts == {0.0: 2, 1.0: 5}
    assert chosen_k == {0.0: 2, 1.0: 5}
    clear_ambiguity = selected.loc[
        selected["time_s"] == 0.0,
        "mixture_adaptive_ambiguity",
    ].iloc[0]
    ambiguous_ambiguity = selected.loc[
        selected["time_s"] == 1.0,
        "mixture_adaptive_ambiguity",
    ].iloc[0]
    assert clear_ambiguity < ambiguous_ambiguity


def test_adaptive_selector_respects_branch_quota_floor() -> None:
    rows = _trajectory_rows().loc[lambda frame: frame["time_s"] == 0.0]
    selected = select_adaptive_mixture_candidates(
        rows,
        config=AdaptiveMixtureTopKConfig(
            min_top_k=1,
            max_top_k=5,
            min_per_branch=1,
            min_per_source=0,
            score_column="ranker_score",
            score_temperature=0.05,
            entropy_weight=0.0,
            margin_weight=1.0,
            sigma_weight=0.0,
            consensus_weight=0.0,
        ),
    )

    assert len(selected) == 3
    assert set(selected["candidate_branch"]) == {"raw", "translated", "dynamic"}
    assert selected["mixture_adaptive_top_k"].iloc[0] == 3


def test_adaptive_candidate_mixture_map_smoke() -> None:
    result = run_adaptive_candidate_mixture_map(
        _trajectory_rows(),
        adaptive_config=AdaptiveMixtureTopKConfig(
            min_top_k=2,
            max_top_k=3,
            min_per_branch=0,
            min_per_source=0,
            score_column="ranker_score",
            score_temperature=0.1,
        ),
        mixture_config=CandidateMixtureMapConfig(
            top_k=3,
            score_column="ranker_score",
            sigma_column="predicted_sigma_m",
            smoothness_weight=10.0,
            iterations=2,
        ),
    )

    assert len(result.mixture_result.estimates) == 4
    assert not result.mixture_result.assignments.empty
    assert result.selection_summary["frame_count"] == 4
    assert result.selection_summary["adaptive_top_k_min"] >= 2
    assert result.selection_summary["adaptive_top_k_max"] <= 3


def test_adaptive_candidate_mixture_cli_writes_outputs(tmp_path) -> None:
    candidates_csv = tmp_path / "candidates.csv"
    output_dir = tmp_path / "out"
    _trajectory_rows().to_csv(candidates_csv, index=False)

    status = adaptive_mixture_main(
        [
            "--candidates-csv",
            str(candidates_csv),
            "--output-dir",
            str(output_dir),
            "--min-top-k",
            "2",
            "--max-top-k",
            "3",
            "--min-per-branch",
            "0",
            "--min-per-source",
            "0",
            "--score-column",
            "ranker_score",
            "--adaptive-score-temperature",
            "0.1",
            "--iterations",
            "2",
            "--smoothness-weight",
            "10",
        ]
    )

    assert status == 0
    assert (output_dir / "mmuad_adaptive_mixture_candidates.csv").exists()
    assert (output_dir / "mmuad_candidate_mixture_estimates.csv").exists()
    summary = json.loads(
        (output_dir / "mmuad_adaptive_mixture_selection_summary.json").read_text(
            encoding="utf-8"
        )
    )
    assert summary["frame_count"] == 4
    assert summary["adaptive_top_k_min"] >= 2
