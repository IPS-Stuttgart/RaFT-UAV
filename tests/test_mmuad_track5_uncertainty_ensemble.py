from __future__ import annotations

import json
from pathlib import Path
from zipfile import ZipFile

import pandas as pd
import pytest

from raft_uav.mmuad.track5_estimate_ensemble import EstimateInput
from raft_uav.mmuad.track5_uncertainty_ensemble import build_track5_uncertainty_ensemble
from raft_uav.mmuad.track5_uncertainty_ensemble import main as uncertainty_ensemble_main
from raft_uav.mmuad.track5_uncertainty_ensemble import write_track5_uncertainty_ensemble_outputs


def _template() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Sequence": ["seq0001", "seq0001", "seq0002"],
            "Timestamp": [0.0, 5.0, 0.0],
            "Position": ["(0,0,0)"] * 3,
            "Classification": [2, 2, 1],
        }
    )


def _low_sigma_estimate() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["seq0001", "seq0001", "seq0002"],
            "time_s": [0.0, 10.0, 0.0],
            "state_x_m": [0.0, 10.0, 4.0],
            "state_y_m": [0.0, 0.0, 4.0],
            "state_z_m": [0.0, 0.0, 4.0],
            "predicted_sigma_m": [1.0, 1.0, 1.0],
        }
    )


def _high_sigma_estimate() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["seq0001", "seq0001", "seq0002"],
            "time_s": [0.0, 10.0, 0.0],
            "state_x_m": [20.0, 30.0, 40.0],
            "state_y_m": [20.0, 20.0, 40.0],
            "state_z_m": [20.0, 20.0, 40.0],
            "predicted_sigma_m": [20.0, 20.0, 20.0],
        }
    )


def test_uncertainty_ensemble_template_sequence_cells_are_officially_normalized(
    tmp_path: Path,
) -> None:
    estimate_csv = tmp_path / "estimate.csv"
    pd.DataFrame(
        {
            "sequence_id": ["seq0001", "seq0001"],
            "time_s": [0.0, 2.0],
            "state_x_m": [0.0, 2.0],
            "state_y_m": [0.0, 0.0],
            "state_z_m": [5.0, 5.0],
            "predicted_sigma_m": [2.0, 2.0],
        }
    ).to_csv(estimate_csv, index=False)
    template = pd.DataFrame(
        {
            "Sequence": [" seq0001 "],
            "Timestamp": [1.0],
            "Position": ["(0,0,0)"],
            "Classification": [2],
        }
    )

    estimates, diagnostics = build_track5_uncertainty_ensemble(
        [EstimateInput("base", estimate_csv, 1.0)],
        template=template,
        uncertainty_column="predicted_sigma_m",
    )

    row = estimates.iloc[0]
    assert row["sequence_id"] == "seq0001"
    assert row["ensemble_source_count"] == 1
    assert row["state_x_m"] == pytest.approx(1.0)
    assert row["state_y_m"] == pytest.approx(0.0)
    assert row["state_z_m"] == pytest.approx(5.0)
    assert row["ensemble_effective_sigma_m"] == pytest.approx(2.0)
    assert diagnostics.iloc[0]["valid_input_count"] == 1


def test_uncertainty_ensemble_downweights_high_sigma_estimate(tmp_path: Path) -> None:
    low = tmp_path / "low.csv"
    high = tmp_path / "high.csv"
    _low_sigma_estimate().to_csv(low, index=False)
    _high_sigma_estimate().to_csv(high, index=False)

    estimates, diagnostics = build_track5_uncertainty_ensemble(
        [EstimateInput("low", low, 1.0), EstimateInput("high", high, 1.0)],
        template=_template(),
        uncertainty_column="predicted_sigma_m",
    )

    midpoint = estimates.loc[
        (estimates["sequence_id"] == "seq0001") & (estimates["time_s"] == 5.0)
    ].iloc[0]
    # Low-sigma midpoint is (5,0,0); high-sigma midpoint is (25,20,20), but
    # inverse-variance weighting keeps the ensemble close to the low-sigma row.
    assert midpoint["state_x_m"] == pytest.approx(5.049875311720699)
    assert midpoint["state_y_m"] == pytest.approx(0.04987531172069825)
    assert midpoint["state_z_m"] == pytest.approx(0.04987531172069825)
    assert midpoint["ensemble_source_count"] == 2
    assert diagnostics["valid_input_count"].tolist() == [2, 2, 2]


def test_uncertainty_ensemble_writes_upload_ready_artifacts(tmp_path: Path) -> None:
    low = tmp_path / "low.csv"
    high = tmp_path / "high.csv"
    _low_sigma_estimate().to_csv(low, index=False)
    _high_sigma_estimate().to_csv(high, index=False)

    paths = write_track5_uncertainty_ensemble_outputs(
        estimate_inputs=[EstimateInput("low", low, 1.0), EstimateInput("high", high, 1.0)],
        template=_template(),
        output_dir=tmp_path / "out",
        class_map={"seq0001": "2", "seq0002": "1"},
        uncertainty_column="predicted_sigma_m",
    )

    assert paths["official_zip"].exists()
    validation = json.loads(paths["validation_json"].read_text(encoding="utf-8"))
    manifest = json.loads(paths["manifest_json"].read_text(encoding="utf-8"))
    assert validation["leaderboard_ready"] is True
    assert validation["codabench_upload_ready"] is True
    assert manifest["valid_ensemble_rows"] == 3
    with ZipFile(paths["official_zip"]) as archive:
        assert archive.namelist() == ["mmaud_results.csv"]
    official = pd.read_csv(paths["official_results_csv"])
    assert official["Classification"].tolist() == [2, 2, 1]


def test_uncertainty_ensemble_cli_writes_outputs(tmp_path: Path) -> None:
    low = tmp_path / "low.csv"
    high = tmp_path / "high.csv"
    template = tmp_path / "template.csv"
    class_map = tmp_path / "class_map.csv"
    output = tmp_path / "out"
    _low_sigma_estimate().to_csv(low, index=False)
    _high_sigma_estimate().to_csv(high, index=False)
    _template().to_csv(template, index=False)
    pd.DataFrame({"sequence_id": ["seq0001", "seq0002"], "uav_type": [2, 1]}).to_csv(
        class_map,
        index=False,
    )

    status = uncertainty_ensemble_main(
        [
            "--estimate-csv",
            f"low={low}",
            "--estimate-csv",
            f"high={high}",
            "--template",
            str(template),
            "--class-map",
            str(class_map),
            "--output-dir",
            str(output),
            "--uncertainty-column",
            "predicted_sigma_m",
            "--require-leaderboard-ready",
        ]
    )

    assert status == 0
    assert (output / "ug2_submission.zip").exists()
    assert (output / "mmuad_track5_uncertainty_ensemble_manifest.json").exists()
