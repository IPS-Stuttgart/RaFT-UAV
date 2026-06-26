from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from raft_uav.mmuad.candidate_consensus_uncertainty import (
    apply_consensus_conditioned_uncertainty,
    main as consensus_uncertainty_main,
    train_consensus_conditioned_uncertainty,
)
from raft_uav.mmuad.schema import CandidateFrame


def _candidate_rows() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for time_s in range(4):
        true_x = float(time_s * 2)
        rows.extend(
            [
                {
                    "sequence_id": "seqA",
                    "time_s": float(time_s),
                    "source": "lidar_360",
                    "track_id": f"supported-{time_s}",
                    "candidate_branch": "source_translation",
                    "candidate_origin_row": f"lidar-{time_s}",
                    "x_m": true_x,
                    "y_m": 0.0,
                    "z_m": 2.0,
                    "std_xy_m": 10.0,
                    "std_z_m": 10.0,
                    "ranker_score": 0.5,
                    "confidence": 0.5,
                    "cluster_point_count": 20,
                },
                {
                    "sequence_id": "seqA",
                    "time_s": float(time_s) + 0.01,
                    "source": "livox_avia",
                    "track_id": f"support-{time_s}",
                    "candidate_branch": "dynamic",
                    "candidate_origin_row": f"livox-{time_s}",
                    "x_m": true_x + 0.2,
                    "y_m": 0.1,
                    "z_m": 2.0,
                    "std_xy_m": 10.0,
                    "std_z_m": 10.0,
                    "ranker_score": 0.45,
                    "confidence": 0.45,
                    "cluster_point_count": 15,
                },
                {
                    "sequence_id": "seqA",
                    "time_s": float(time_s),
                    "source": "radar_enhance_pcl",
                    "track_id": f"isolated-{time_s}",
                    "candidate_branch": "raw",
                    "candidate_origin_row": f"radar-{time_s}",
                    "x_m": true_x + 20.0,
                    "y_m": 10.0,
                    "z_m": 8.0,
                    "std_xy_m": 10.0,
                    "std_z_m": 10.0,
                    "ranker_score": 0.8,
                    "confidence": 0.8,
                    "cluster_point_count": 3,
                },
            ]
        )
    return pd.DataFrame.from_records(rows)


def _truth_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["seqA"] * 4,
            "time_s": [float(value) for value in range(4)],
            "x_m": [float(value * 2) for value in range(4)],
            "y_m": [0.0] * 4,
            "z_m": [2.0] * 4,
        }
    )


def _model():
    model, _, _, _ = train_consensus_conditioned_uncertainty(
        CandidateFrame(_candidate_rows()),
        _truth_rows(),
        model_type="ridge",
        sigma_min_m=1.0,
        sigma_max_m=30.0,
        ridge_alpha=0.1,
        max_truth_time_delta_s=0.05,
        time_window_s=0.05,
        distance_gate_m=3.0,
    )
    return model


def test_consensus_sigma_shrink_preserves_isolated_uncertainty() -> None:
    applied = apply_consensus_conditioned_uncertainty(
        CandidateFrame(_candidate_rows()),
        _model(),
        consensus_sigma_weight=1.0,
        consensus_sigma_min_factor=0.5,
        time_window_s=0.05,
        distance_gate_m=3.0,
        replace_covariance=True,
        z_scale=2.0,
    ).rows.set_index("track_id")

    supported = applied.loc["supported-0"]
    isolated = applied.loc["isolated-0"]
    assert supported["candidate_uncertainty_consensus_factor"] < isolated[
        "candidate_uncertainty_consensus_factor"
    ]
    assert supported["predicted_sigma_m"] < supported["raw_predicted_sigma_m"]
    assert isolated["predicted_sigma_m"] == pytest.approx(
        isolated["raw_predicted_sigma_m"]
    )
    assert np.allclose(applied["std_xy_m"], applied["predicted_sigma_m"])
    assert np.allclose(applied["std_z_m"], 2.0 * applied["predicted_sigma_m"])


def test_zero_weight_preserves_backward_compatible_output() -> None:
    applied = apply_consensus_conditioned_uncertainty(
        CandidateFrame(_candidate_rows()),
        _model(),
        consensus_sigma_weight=0.0,
    ).rows

    assert "raw_predicted_sigma_m" not in applied.columns
    assert "candidate_uncertainty_consensus_factor" not in applied.columns
    assert applied["predicted_sigma_m"].notna().all()


def test_consensus_sigma_shrink_cli_writes_adjusted_columns(tmp_path: Path) -> None:
    candidates_csv = tmp_path / "candidates.csv"
    truth_csv = tmp_path / "truth.csv"
    model_json = tmp_path / "model.json"
    output_csv = tmp_path / "adjusted.csv"
    _candidate_rows().to_csv(candidates_csv, index=False)
    _truth_rows().to_csv(truth_csv, index=False)

    assert (
        consensus_uncertainty_main(
            [
                "train",
                "--candidates-csv",
                str(candidates_csv),
                "--truth-csv",
                str(truth_csv),
                "--model-json",
                str(model_json),
                "--model-type",
                "ridge",
                "--max-truth-time-delta-s",
                "0.05",
                "--consensus-distance-gate-m",
                "3",
            ]
        )
        == 0
    )
    assert (
        consensus_uncertainty_main(
            [
                "apply",
                "--candidates-csv",
                str(candidates_csv),
                "--model-json",
                str(model_json),
                "--output-csv",
                str(output_csv),
                "--consensus-distance-gate-m",
                "3",
                "--consensus-sigma-weight",
                "1",
                "--consensus-sigma-min-factor",
                "0.5",
            ]
        )
        == 0
    )

    output = pd.read_csv(output_csv)
    assert "raw_predicted_sigma_m" in output.columns
    assert "candidate_uncertainty_consensus_factor" in output.columns
    assert (output["predicted_sigma_m"] <= output["raw_predicted_sigma_m"]).all()
