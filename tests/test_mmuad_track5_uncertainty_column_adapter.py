from __future__ import annotations

import json
from pathlib import Path
import tomllib

import pandas as pd
import pytest

from raft_uav.mmuad.track5_estimate_ensemble import EstimateInput
from raft_uav.mmuad.track5_uncertainty_column_adapter import main as adapter_main
from raft_uav.mmuad.track5_uncertainty_column_adapter import normalize_uncertainty_estimate_inputs
from raft_uav.mmuad.track5_uncertainty_column_adapter import write_uncertainty_column_adapter_outputs


def _template() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Sequence": ["seq0001", "seq0001"],
            "Timestamp": [0.0, 10.0],
            "Position": ["(0,0,0)", "(10,0,0)"],
            "Classification": [2, 2],
        }
    )


def _low_sigma_estimate() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["seq0001", "seq0001"],
            "time_s": [0.0, 10.0],
            "state_x_m": [0.0, 10.0],
            "state_y_m": [0.0, 0.0],
            "state_z_m": [0.0, 0.0],
            "low_sigma": [1.0, 1.0],
        }
    )


def _high_sigma_estimate() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["seq0001", "seq0001"],
            "time_s": [0.0, 10.0],
            "state_x_m": [100.0, 110.0],
            "state_y_m": [100.0, 100.0],
            "state_z_m": [100.0, 100.0],
            "model_uncertainty": [50.0, 50.0],
        }
    )


def _zero_padded_estimate() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["001"],
            "time_s": [0.0],
            "state_x_m": [1.0],
            "state_y_m": [2.0],
            "state_z_m": [3.0],
            "model_sigma": [2.0],
        }
    )


def _zero_padded_template() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Sequence": ["001"],
            "Timestamp": [0.0],
            "Position": ["(0,0,0)"],
            "Classification": [2],
        }
    )


def test_uncertainty_adapter_normalizes_per_input_columns(tmp_path: Path) -> None:
    low_csv = tmp_path / "low.csv"
    high_csv = tmp_path / "high.csv"
    _low_sigma_estimate().to_csv(low_csv, index=False)
    _high_sigma_estimate().to_csv(high_csv, index=False)

    normalized, summary = normalize_uncertainty_estimate_inputs(
        [
            EstimateInput("low", low_csv, 1.0),
            EstimateInput("high", high_csv, 1.0),
        ],
        output_dir=tmp_path / "out",
        uncertainty_columns={"low": "low_sigma", "high": "model_uncertainty"},
        output_uncertainty_column="predicted_sigma_m",
    )

    assert [item.label for item in normalized] == ["low", "high"]
    assert summary["source_uncertainty_column"].tolist() == ["low_sigma", "model_uncertainty"]
    low_rows = pd.read_csv(normalized[0].path)
    high_rows = pd.read_csv(normalized[1].path)
    assert low_rows["predicted_sigma_m"].tolist() == [1.0, 1.0]
    assert high_rows["predicted_sigma_m"].tolist() == [50.0, 50.0]


def test_uncertainty_adapter_matches_sanitized_cli_labels(tmp_path: Path) -> None:
    low_csv = tmp_path / "low.csv"
    _low_sigma_estimate().to_csv(low_csv, index=False)

    normalized, summary = normalize_uncertainty_estimate_inputs(
        [EstimateInput("sensor/low estimate", low_csv, 1.0)],
        output_dir=tmp_path / "out",
        uncertainty_columns={"sensor_low_estimate": "low_sigma"},
    )

    rows = pd.read_csv(normalized[0].path)
    assert normalized[0].path.name == "sensor_low_estimate.csv"
    assert summary["source_uncertainty_column"].tolist() == ["low_sigma"]
    assert rows["predicted_sigma_m"].tolist() == [1.0, 1.0]


def test_uncertainty_adapter_matches_padded_requested_columns(tmp_path: Path) -> None:
    estimate_csv = tmp_path / "estimate.csv"
    pd.DataFrame(
        {
            "sequence_id": ["001"],
            "time_s": [0.0],
            "state_x_m": [1.0],
            "state_y_m": [2.0],
            "state_z_m": [3.0],
            " model_sigma ": [2.5],
        }
    ).to_csv(estimate_csv, index=False)

    normalized, summary = normalize_uncertainty_estimate_inputs(
        [EstimateInput("model", estimate_csv, 1.0)],
        output_dir=tmp_path / "out",
        uncertainty_columns={"model": "model_sigma"},
    )

    rows = pd.read_csv(normalized[0].path)
    assert summary["source_uncertainty_column"].tolist() == [" model_sigma "]
    assert rows["predicted_sigma_m"].tolist() == [2.5]


def test_uncertainty_adapter_auto_detects_padded_default_uncertainty_column(
    tmp_path: Path,
) -> None:
    estimate_csv = tmp_path / "estimate.csv"
    pd.DataFrame(
        {
            "sequence_id": ["001"],
            "time_s": [0.0],
            "state_x_m": [1.0],
            "state_y_m": [2.0],
            "state_z_m": [3.0],
            " predicted_sigma_m ": [4.0],
        }
    ).to_csv(estimate_csv, index=False)

    normalized, summary = normalize_uncertainty_estimate_inputs(
        [EstimateInput("model", estimate_csv, 1.0)],
        output_dir=tmp_path / "out",
    )

    rows = pd.read_csv(normalized[0].path)
    assert summary["source_uncertainty_column"].tolist() == [" predicted_sigma_m "]
    assert rows["predicted_sigma_m"].tolist() == [4.0]


def test_uncertainty_adapter_can_run_upload_ready_ensemble(tmp_path: Path) -> None:
    low_csv = tmp_path / "low.csv"
    high_csv = tmp_path / "high.csv"
    _low_sigma_estimate().to_csv(low_csv, index=False)
    _high_sigma_estimate().to_csv(high_csv, index=False)

    paths = write_uncertainty_column_adapter_outputs(
        estimate_inputs=[
            EstimateInput("low", low_csv, 1.0),
            EstimateInput("high", high_csv, 1.0),
        ],
        output_dir=tmp_path / "out",
        uncertainty_columns={"low": "low_sigma", "high": "model_uncertainty"},
        template=_template(),
        class_map={"seq0001": "2"},
        run_ensemble=True,
    )

    assert paths["summary_csv"].exists()
    assert paths["manifest_json"].exists()
    assert paths["ensemble_official_zip"].exists()
    ensemble = pd.read_csv(paths["ensemble_ensemble_estimates_csv"])
    assert ensemble.loc[0, "state_x_m"] == pytest.approx(0.03998400639744103)
    assert ensemble.loc[0, "state_y_m"] == pytest.approx(0.03998400639744103)
    manifest = json.loads(paths["manifest_json"].read_text(encoding="utf-8"))
    assert manifest["run_ensemble"] is True


def test_uncertainty_adapter_preserves_zero_padded_sequence_ids(tmp_path: Path) -> None:
    estimate_csv = tmp_path / "estimate.csv"
    _zero_padded_estimate().to_csv(estimate_csv, index=False)

    normalized, _ = normalize_uncertainty_estimate_inputs(
        [EstimateInput("model", estimate_csv, 1.0)],
        output_dir=tmp_path / "out",
        uncertainty_columns={"model": "model_sigma"},
        output_uncertainty_column="predicted_sigma_m",
    )

    rows = pd.read_csv(normalized[0].path, dtype=str, keep_default_na=False)
    assert rows.loc[0, "sequence_id"] == "001"


def test_uncertainty_adapter_run_ensemble_matches_zero_padded_template(
    tmp_path: Path,
) -> None:
    estimate_csv = tmp_path / "estimate.csv"
    _zero_padded_estimate().to_csv(estimate_csv, index=False)

    paths = write_uncertainty_column_adapter_outputs(
        estimate_inputs=[EstimateInput("model", estimate_csv, 1.0)],
        output_dir=tmp_path / "out",
        uncertainty_columns={"model": "model_sigma"},
        template=_zero_padded_template(),
        class_map={"001": "2"},
        run_ensemble=True,
    )

    ensemble = pd.read_csv(
        paths["ensemble_ensemble_estimates_csv"],
        dtype=str,
        keep_default_na=False,
    )
    assert ensemble.loc[0, "sequence_id"] == "001"
    assert float(ensemble.loc[0, "state_x_m"]) == pytest.approx(1.0)
    assert float(ensemble.loc[0, "state_y_m"]) == pytest.approx(2.0)
    assert float(ensemble.loc[0, "state_z_m"]) == pytest.approx(3.0)
    assert int(float(ensemble.loc[0, "ensemble_source_count"])) == 1

    validation = json.loads(paths["ensemble_validation_json"].read_text(encoding="utf-8"))
    assert validation["leaderboard_ready"] is True


def test_uncertainty_adapter_cli_writes_outputs(tmp_path: Path) -> None:
    low_csv = tmp_path / "low.csv"
    high_csv = tmp_path / "high.csv"
    template_csv = tmp_path / "template.csv"
    class_map_csv = tmp_path / "class_map.csv"
    output_dir = tmp_path / "out"
    _low_sigma_estimate().to_csv(low_csv, index=False)
    _high_sigma_estimate().to_csv(high_csv, index=False)
    _template().to_csv(template_csv, index=False)
    pd.DataFrame({"sequence_id": ["seq0001"], "uav_type": [2]}).to_csv(
        class_map_csv,
        index=False,
    )

    status = adapter_main(
        [
            "--estimate-csv",
            f"low={low_csv}",
            "--estimate-csv",
            f"high={high_csv}",
            "--uncertainty-column",
            "low=low_sigma",
            "--uncertainty-column",
            "high=model_uncertainty",
            "--output-dir",
            str(output_dir),
            "--run-ensemble",
            "--template",
            str(template_csv),
            "--class-map",
            str(class_map_csv),
        ]
    )

    assert status == 0
    assert (output_dir / "mmuad_track5_uncertainty_column_adapter_manifest.json").exists()
    assert (output_dir / "uncertainty_ensemble" / "ug2_submission.zip").exists()


def test_uncertainty_adapter_entrypoint_is_exposed() -> None:
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    assert (
        pyproject["project"]["scripts"]["raft-uav-mmuad-track5-uncertainty-column-adapter"]
        == "raft_uav.mmuad.track5_uncertainty_column_adapter:main"
    )
