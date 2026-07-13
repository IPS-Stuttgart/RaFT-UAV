from __future__ import annotations

import json

import pandas as pd
import pytest

from raft_uav.mmuad.candidate_constant_velocity_bridge_quota import (
    ConstantVelocityBridgeConfig,
    attach_constant_velocity_bridge_features,
    build_constant_velocity_bridge_reservoir,
    main as bridge_reservoir_main,
)
from raft_uav.mmuad.candidate_reservoir import ReservoirConfig


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
                    "track_id": f"zigzag-{time_s:g}",
                    "x_m": 20.0 if time_s != 1.0 else 100.0,
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


def _bridge_config(**kwargs: object) -> ConstantVelocityBridgeConfig:
    defaults = {
        "max_frame_gap_s": 1.1,
        "max_speed_mps": 200.0,
        "max_interpolation_error_m": 3.0,
        "interpolation_scale_m": 2.0,
    }
    defaults.update(kwargs)
    return ConstantVelocityBridgeConfig(**defaults)


def test_bridge_supports_coherent_middle_candidate_and_rejects_zigzag() -> None:
    annotated = attach_constant_velocity_bridge_features(
        _candidate_rows(),
        config=_bridge_config(),
    ).set_index("track_id")

    assert annotated.loc["coherent-1", "candidate_cv_bridge_supported"]
    assert annotated.loc["coherent-1", "candidate_cv_bridge_error_m"] == pytest.approx(0.0)
    assert annotated.loc["coherent-1", "candidate_cv_bridge_score"] == pytest.approx(1.0)
    assert not annotated.loc["zigzag-1", "candidate_cv_bridge_supported"]
    assert not annotated.loc["coherent-0", "candidate_cv_bridge_supported"]
    assert not annotated.loc["coherent-2", "candidate_cv_bridge_supported"]


def test_bridge_uses_time_weighted_interpolation() -> None:
    rows = pd.DataFrame(
        {
            "sequence_id": ["seqA", "seqA", "seqA"],
            "time_s": [0.0, 1.0, 3.0],
            "source": ["lidar_360"] * 3,
            "track_id": ["previous", "middle", "next"],
            "x_m": [0.0, 1.0, 3.0],
            "y_m": [0.0, 0.0, 0.0],
            "z_m": [1.0, 1.0, 1.0],
            "confidence": [0.5, 0.5, 0.5],
        }
    )

    annotated = attach_constant_velocity_bridge_features(
        rows,
        config=_bridge_config(max_frame_gap_s=2.1),
    ).set_index("track_id")

    assert annotated.loc["middle", "candidate_cv_bridge_supported"]
    assert annotated.loc["middle", "candidate_cv_bridge_error_m"] == pytest.approx(0.0)
    assert annotated.loc["middle", "candidate_cv_bridge_prev_dt_s"] == pytest.approx(1.0)
    assert annotated.loc["middle", "candidate_cv_bridge_next_dt_s"] == pytest.approx(2.0)


def test_bridge_quota_recovers_candidate_missed_by_score_only_reservoir() -> None:
    reservoir_config = ReservoirConfig(
        global_top_n=1,
        per_source_top_n=0,
        per_branch_top_n=0,
        max_candidates_per_frame=2,
        score_column="ranker_score",
        fallback_score_column="confidence",
    )
    baseline = build_constant_velocity_bridge_reservoir(
        _candidate_rows(),
        reservoir_config=reservoir_config,
        bridge_config=_bridge_config(bridge_top_n=0),
    )
    augmented = build_constant_velocity_bridge_reservoir(
        _candidate_rows(),
        reservoir_config=reservoir_config,
        bridge_config=_bridge_config(bridge_top_n=1),
    )

    assert set(baseline["track_id"]) == {"zigzag-0", "zigzag-1", "zigzag-2"}
    assert set(augmented["track_id"]) == {
        "zigzag-0",
        "zigzag-1",
        "zigzag-2",
        "coherent-1",
    }
    middle = augmented.loc[augmented["track_id"] == "coherent-1"].iloc[0]
    assert "cv_bridge" in middle["candidate_reservoir_reason"]
    assert middle["candidate_reservoir_protected"]


def test_bridge_can_require_same_source_and_branch() -> None:
    rows = _candidate_rows()
    endpoint = rows["track_id"].isin(["coherent-0", "coherent-2"])
    rows.loc[endpoint, "source"] = "livox_avia"
    rows.loc[endpoint, "candidate_branch"] = "translated"

    unrestricted = attach_constant_velocity_bridge_features(
        rows,
        config=_bridge_config(),
    ).set_index("track_id")
    restricted = attach_constant_velocity_bridge_features(
        rows,
        config=_bridge_config(require_same_source=True, require_same_branch=True),
    ).set_index("track_id")

    assert unrestricted.loc["coherent-1", "candidate_cv_bridge_supported"]
    assert not restricted.loc["coherent-1", "candidate_cv_bridge_supported"]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("bridge_top_n", -1),
        ("max_frame_gap_s", 0.0),
        ("max_speed_mps", float("nan")),
        ("max_interpolation_error_m", float("inf")),
        ("interpolation_scale_m", -1.0),
        ("max_neighbors_per_side", -1),
    ],
)
def test_bridge_rejects_invalid_controls(field: str, value: float) -> None:
    with pytest.raises(ValueError):
        attach_constant_velocity_bridge_features(
            _candidate_rows(),
            config=ConstantVelocityBridgeConfig(**{field: value}),
        )


def test_bridge_cli_writes_summary_and_oracle_outputs(tmp_path) -> None:
    candidate_csv = tmp_path / "candidates.csv"
    truth_csv = tmp_path / "truth.csv"
    output_csv = tmp_path / "reservoir.csv"
    summary_json = tmp_path / "summary.json"
    oracle_frame_csv = tmp_path / "oracle_frames.csv"
    oracle_summary_csv = tmp_path / "oracle_summary.csv"
    oracle_by_sequence_csv = tmp_path / "oracle_by_sequence.csv"
    _candidate_rows().to_csv(candidate_csv, index=False)
    _truth_rows().to_csv(truth_csv, index=False)

    status = bridge_reservoir_main(
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
            "--bridge-top-n",
            "1",
            "--max-frame-gap-s",
            "1.1",
            "--max-speed-mps",
            "200",
            "--max-interpolation-error-m",
            "3",
            "--max-truth-time-delta-s",
            "0.1",
        ]
    )

    assert status == 0
    reservoir = pd.read_csv(output_csv)
    summary = json.loads(summary_json.read_text(encoding="utf-8"))
    oracle = pd.read_csv(oracle_summary_csv)
    assert len(reservoir) == 4
    assert summary["cv_bridge_quota_rows"] == 1
    assert summary["cv_bridge_supported_rows"] == 1
    expected_oracle_mse = (20.0**2 + 18.0**2) / 3.0
    assert oracle.loc[0, "oracle_all_3d_m_mse"] == pytest.approx(expected_oracle_mse)
    assert oracle_frame_csv.exists()
    assert oracle_by_sequence_csv.exists()
