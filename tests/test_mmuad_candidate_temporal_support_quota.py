from __future__ import annotations

import json

import pandas as pd
import pytest

from raft_uav.mmuad.candidate_reservoir import ReservoirConfig
from raft_uav.mmuad.candidate_temporal_support_quota import (
    TemporalSupportConfig,
    attach_temporal_support_features,
    build_temporal_support_reservoir,
    main as temporal_reservoir_main,
)


def _candidate_rows() -> pd.DataFrame:
    records = []
    for time_s in (0.0, 1.0, 2.0):
        records.extend(
            [
                {
                    "sequence_id": "seqA",
                    "time_s": time_s,
                    "source": "lidar_360",
                    "candidate_branch": "raw",
                    "track_id": f"coherent-{time_s:g}",
                    "x_m": time_s,
                    "y_m": 0.0,
                    "z_m": 1.0,
                    "confidence": 0.10,
                },
                {
                    "sequence_id": "seqA",
                    "time_s": time_s,
                    "source": "lidar_360",
                    "candidate_branch": "raw",
                    "track_id": f"distractor-{time_s:g}",
                    "x_m": 20.0 + 100.0 * time_s,
                    "y_m": 0.0,
                    "z_m": 1.0,
                    "confidence": 0.99,
                },
            ]
        )
    return pd.DataFrame.from_records(records)


def _truth_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["seqA", "seqA", "seqA"],
            "time_s": [0.0, 1.0, 2.0],
            "x_m": [0.0, 1.0, 2.0],
            "y_m": [0.0, 0.0, 0.0],
            "z_m": [1.0, 1.0, 1.0],
        }
    )


def test_temporal_support_detects_coherent_low_score_track() -> None:
    annotated = attach_temporal_support_features(
        _candidate_rows(),
        config=TemporalSupportConfig(
            max_frame_gap_s=1.1,
            max_speed_mps=5.0,
            distance_scale_m=5.0,
        ),
    ).set_index("track_id")

    assert annotated.loc["coherent-1", "candidate_temporal_support_sides"] == 2
    assert annotated.loc["coherent-1", "candidate_temporal_two_sided"]
    assert annotated.loc["coherent-1", "candidate_temporal_support_score"] > 1.0
    assert annotated.loc["distractor-1", "candidate_temporal_support_sides"] == 0


def test_temporal_quota_restores_candidate_missed_by_score_only_reservoir() -> None:
    base_config = ReservoirConfig(
        global_top_n=1,
        per_source_top_n=0,
        per_branch_top_n=0,
        max_candidates_per_frame=2,
        score_column="ranker_score",
        fallback_score_column="confidence",
    )
    baseline = build_temporal_support_reservoir(
        _candidate_rows(),
        reservoir_config=base_config,
        temporal_config=TemporalSupportConfig(temporal_top_n=0),
    )
    augmented = build_temporal_support_reservoir(
        _candidate_rows(),
        reservoir_config=base_config,
        temporal_config=TemporalSupportConfig(
            temporal_top_n=1,
            max_frame_gap_s=1.1,
            max_speed_mps=5.0,
            distance_scale_m=5.0,
            min_support_sides=1,
        ),
    )

    assert set(baseline["track_id"]) == {
        "distractor-0",
        "distractor-1",
        "distractor-2",
    }
    assert set(augmented["track_id"]) == {
        "coherent-0",
        "coherent-1",
        "coherent-2",
        "distractor-0",
        "distractor-1",
        "distractor-2",
    }
    coherent_middle = augmented.loc[augmented["track_id"] == "coherent-1"].iloc[0]
    assert "temporal_support:2side" in coherent_middle["candidate_reservoir_reason"]
    assert coherent_middle["candidate_reservoir_protected"]


def test_temporal_support_can_require_same_source_and_branch() -> None:
    rows = pd.DataFrame(
        {
            "sequence_id": ["seqA", "seqA"],
            "time_s": [0.0, 1.0],
            "source": ["lidar_360", "livox_avia"],
            "candidate_branch": ["raw", "translated"],
            "track_id": ["a", "b"],
            "x_m": [0.0, 0.1],
            "y_m": [0.0, 0.0],
            "z_m": [1.0, 1.0],
            "confidence": [0.5, 0.5],
        }
    )

    unrestricted = attach_temporal_support_features(
        rows,
        config=TemporalSupportConfig(max_frame_gap_s=1.1, max_speed_mps=5.0),
    )
    restricted = attach_temporal_support_features(
        rows,
        config=TemporalSupportConfig(
            max_frame_gap_s=1.1,
            max_speed_mps=5.0,
            require_same_source=True,
            require_same_branch=True,
        ),
    )

    assert unrestricted["candidate_temporal_support_sides"].tolist() == [1, 1]
    assert restricted["candidate_temporal_support_sides"].tolist() == [0, 0]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("max_frame_gap_s", float("nan")),
        ("max_frame_gap_s", 0.0),
        ("max_speed_mps", float("inf")),
        ("distance_scale_m", -1.0),
        ("temporal_top_n", -1),
        ("min_support_sides", 3),
    ],
)
def test_temporal_support_rejects_invalid_controls(field: str, value: float) -> None:
    kwargs = {field: value}
    with pytest.raises(ValueError):
        attach_temporal_support_features(
            _candidate_rows(),
            config=TemporalSupportConfig(**kwargs),
        )


def test_temporal_support_cli_writes_oracle_and_summary_outputs(tmp_path) -> None:
    candidate_csv = tmp_path / "candidates.csv"
    truth_csv = tmp_path / "truth.csv"
    output_csv = tmp_path / "reservoir.csv"
    summary_json = tmp_path / "summary.json"
    oracle_frame_csv = tmp_path / "oracle_frames.csv"
    oracle_summary_csv = tmp_path / "oracle_summary.csv"
    oracle_by_sequence_csv = tmp_path / "oracle_by_sequence.csv"
    _candidate_rows().to_csv(candidate_csv, index=False)
    _truth_rows().to_csv(truth_csv, index=False)

    status = temporal_reservoir_main(
        [
            "--candidate-csv",
            f"raw={candidate_csv}",
            "--output-csv",
            str(output_csv),
            "--summary-json",
            str(summary_json),
            "--truth-csv",
            str(truth_csv),
            "--oracle-frame-csv",
            str(oracle_frame_csv),
            "--oracle-summary-csv",
            str(oracle_summary_csv),
            "--oracle-by-sequence-csv",
            str(oracle_by_sequence_csv),
            "--global-top-n",
            "1",
            "--per-source-top-n",
            "0",
            "--per-branch-top-n",
            "0",
            "--max-candidates-per-frame",
            "2",
            "--temporal-top-n",
            "1",
            "--max-frame-gap-s",
            "1.1",
            "--max-speed-mps",
            "5",
            "--max-truth-time-delta-s",
            "0.1",
        ]
    )

    assert status == 0
    reservoir = pd.read_csv(output_csv)
    summary = json.loads(summary_json.read_text(encoding="utf-8"))
    oracle = pd.read_csv(oracle_summary_csv)
    assert len(reservoir) == 6
    assert summary["temporal_quota_candidate_rows"] == 3
    assert summary["temporal_two_sided_rows"] >= 1
    assert oracle.loc[0, "oracle_all_3d_m_mse"] == 0.0
    assert oracle_frame_csv.exists()
    assert oracle_by_sequence_csv.exists()
