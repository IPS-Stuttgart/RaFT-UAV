from __future__ import annotations

from pathlib import Path

import pandas as pd

from raft_uav.mmuad.candidate_oracle_gap import build_candidate_oracle_gap
from raft_uav.mmuad.candidate_oracle_gap import main as candidate_oracle_gap_main


def _truth_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["seq001"],
            "time_s": [0.0],
            "x_m": [0.0],
            "y_m": [0.0],
            "z_m": [0.0],
        }
    )


def _candidate_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["seq001", "seq001", "seq001"],
            "time_s": [0.0, 0.0, 0.0],
            "source": ["lidar_360", "lidar_360", "radar_enhance_pcl"],
            "track_id": ["near", "far", "selected"],
            "x_m": [0.2, 10.0, 5.0],
            "y_m": [0.0, 0.0, 0.0],
            "z_m": [0.0, 0.0, 0.0],
            "confidence": [0.6, 0.9, 0.5],
        }
    )


def _selected_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["seq001"],
            "time_s": [0.0],
            "source": ["radar-enhance-pcl"],
            "track_id": ["selected"],
            "x_m": [5.0],
            "y_m": [0.0],
            "z_m": [0.0],
            "confidence": [0.5],
        }
    )


def test_candidate_oracle_gap_reports_sensor_regret() -> None:
    rows = build_candidate_oracle_gap(
        _candidate_frame(),
        _selected_frame(),
        _truth_frame(),
        max_time_delta_s=0.1,
    )

    lidar = rows.loc[rows["sensor"] == "lidar_360"].iloc[0]
    radar = rows.loc[rows["sensor"] == "radar_enhance_pcl"].iloc[0]
    assert float(lidar["nearest_minus_truth_error_m"]) == 0.2
    assert float(lidar["selected_minus_truth_error_m"]) == 5.0
    assert float(lidar["candidate_regret_m"]) == 4.8
    assert not bool(lidar["selected_source_matches_sensor"])
    assert bool(radar["selected_source_matches_sensor"])
    assert float(radar["candidate_regret_m"]) == 0.0


def test_candidate_oracle_gap_cli_writes_requested_csv(tmp_path: Path) -> None:
    truth = tmp_path / "truth.csv"
    candidates = tmp_path / "candidates.csv"
    selected = tmp_path / "selected.csv"
    output = tmp_path / "out"
    _truth_frame().to_csv(truth, index=False)
    _candidate_frame().to_csv(candidates, index=False)
    _selected_frame().to_csv(selected, index=False)

    status = candidate_oracle_gap_main(
        [
            "--truth-file",
            str(truth),
            "--selected-tracklets",
            str(selected),
            "--candidate-csv",
            str(candidates),
            "--output-dir",
            str(output),
            "--max-time-delta-s",
            "0.1",
        ]
    )

    assert status == 0
    path = output / "mmuad_candidate_oracle_gap.csv"
    assert path.exists()
    rows = pd.read_csv(path)
    assert {
        "sensor",
        "nearest_candidate_track_id",
        "selected_candidate_track_id",
        "selected_minus_truth_error_m",
        "nearest_minus_truth_error_m",
        "candidate_regret_m",
    }.issubset(rows.columns)
    assert len(rows) == 2
