from __future__ import annotations

import json
from pathlib import Path
from zipfile import ZipFile

import pandas as pd
import pytest

from raft_uav.mmuad.track5_estimate_ensemble import EstimateInput
from raft_uav.mmuad.track5_trajectory_medoid_ensemble import (
    build_track5_trajectory_medoid_ensemble,
    write_track5_trajectory_medoid_outputs,
)


def _template() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Sequence": ["seq0001", "seq0001", "seq0001"],
            "Timestamp": [0.0, 1.0, 2.0],
            "Position": ["(0,0,0)"] * 3,
            "Classification": [2, 2, 2],
        }
    )


def _estimate(x_values: list[float], *, times: list[float] | None = None) -> pd.DataFrame:
    sample_times = times if times is not None else [0.0, 1.0, 2.0]
    return pd.DataFrame(
        {
            "sequence_id": ["seq0001"] * len(sample_times),
            "time_s": sample_times,
            "state_x_m": x_values,
            "state_y_m": [0.0] * len(sample_times),
            "state_z_m": [0.0] * len(sample_times),
        }
    )


def test_track5_trajectory_medoid_selects_central_complete_candidate() -> None:
    estimates, diagnostics = build_track5_trajectory_medoid_ensemble(
        [
            ("left", _estimate([0.0, 0.0, 0.0]), 1.0),
            ("central", _estimate([1.0, 1.0, 1.0]), 1.0),
            ("outlier", _estimate([100.0, 100.0, 100.0]), 1.0),
        ],
        _template(),
    )

    assert estimates["state_x_m"].tolist() == pytest.approx([1.0, 1.0, 1.0])
    assert estimates["trajectory_medoid_sequence_label"].unique().tolist() == [
        "central"
    ]
    assert not estimates["trajectory_medoid_fallback"].any()
    selected = diagnostics.loc[diagnostics["selected"]].iloc[0]
    assert selected["candidate_label"] == "central"
    assert selected["trajectory_medoid_score_m"] < diagnostics.loc[
        diagnostics["candidate_label"] == "left",
        "trajectory_medoid_score_m",
    ].iloc[0]


def test_track5_trajectory_medoid_respects_candidate_weights() -> None:
    estimates, diagnostics = build_track5_trajectory_medoid_ensemble(
        [
            ("trusted", _estimate([0.0, 0.0, 0.0]), 5.0),
            ("middle", _estimate([10.0, 10.0, 10.0]), 1.0),
            ("outlier", _estimate([100.0, 100.0, 100.0]), 1.0),
        ],
        _template(),
    )

    assert estimates["trajectory_medoid_sequence_label"].unique().tolist() == [
        "trusted"
    ]
    assert estimates["state_x_m"].tolist() == pytest.approx([0.0, 0.0, 0.0])
    assert diagnostics.loc[diagnostics["selected"], "candidate_weight"].iloc[0] == 5.0


def test_track5_trajectory_medoid_uses_candidate_fallback_for_missing_row() -> None:
    estimates, diagnostics = build_track5_trajectory_medoid_ensemble(
        [
            ("partial", _estimate([0.0, 0.0], times=[0.0, 2.0]), 10.0),
            ("complete", _estimate([10.0, 10.0, 10.0]), 1.0),
        ],
        _template(),
        min_coverage_fraction=0.5,
        max_nearest_time_delta_s=0.0,
    )

    assert estimates["trajectory_medoid_sequence_label"].unique().tolist() == [
        "partial"
    ]
    assert estimates["trajectory_medoid_row_label"].tolist() == [
        "partial",
        "complete",
        "partial",
    ]
    assert estimates["trajectory_medoid_fallback"].tolist() == [False, True, False]
    selected = diagnostics.loc[diagnostics["selected"]].iloc[0]
    assert selected["coverage_fraction"] == pytest.approx(2.0 / 3.0)


def test_track5_trajectory_medoid_rejects_normalized_label_collisions() -> None:
    with pytest.raises(ValueError, match="collide after normalization"):
        build_track5_trajectory_medoid_ensemble(
            [
                ("candidate", _estimate([0.0, 0.0, 0.0]), 1.0),
                (" candidate ", _estimate([1.0, 1.0, 1.0]), 1.0),
            ],
            _template(),
        )


def test_track5_trajectory_medoid_writes_leaderboard_ready_outputs(
    tmp_path: Path,
) -> None:
    left_path = tmp_path / "left.csv"
    central_path = tmp_path / "central.csv"
    outlier_path = tmp_path / "outlier.csv"
    _estimate([0.0, 0.0, 0.0]).to_csv(left_path, index=False)
    _estimate([1.0, 1.0, 1.0]).to_csv(central_path, index=False)
    _estimate([100.0, 100.0, 100.0]).to_csv(outlier_path, index=False)

    paths = write_track5_trajectory_medoid_outputs(
        estimate_inputs=[
            EstimateInput("left", left_path, 1.0),
            EstimateInput("central", central_path, 1.0),
            EstimateInput("outlier", outlier_path, 1.0),
        ],
        template=_template(),
        output_dir=tmp_path / "out",
        class_map={"seq0001": "2"},
    )

    validation = json.loads(paths["validation_json"].read_text(encoding="utf-8"))
    manifest = json.loads(paths["manifest_json"].read_text(encoding="utf-8"))
    assert validation["leaderboard_ready"] is True
    assert validation["codabench_upload_ready"] is True
    assert manifest["sequence_count"] == 1
    assert manifest["fallback_row_count"] == 0
    assert manifest["selected_sequences"][0]["candidate_label"] == "central"
    with ZipFile(paths["official_zip"]) as archive:
        assert archive.namelist() == ["mmaud_results.csv"]
