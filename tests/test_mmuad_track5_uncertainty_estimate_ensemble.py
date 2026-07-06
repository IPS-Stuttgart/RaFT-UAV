from __future__ import annotations

import json
from pathlib import Path
import tomllib
from zipfile import ZipFile

import pandas as pd
import pytest

from raft_uav.mmuad.track5_uncertainty_estimate_ensemble import (
    build_track5_uncertainty_estimate_ensemble,
)
from raft_uav.mmuad.track5_uncertainty_estimate_ensemble import main as uncertainty_main
from raft_uav.mmuad.track5_uncertainty_estimate_ensemble import parse_uncertainty_estimate_spec
from raft_uav.mmuad.track5_uncertainty_estimate_ensemble import (
    write_track5_uncertainty_estimate_ensemble_outputs,
)


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
            "state_x_m": [100.0, 110.0, 104.0],
            "state_y_m": [100.0, 100.0, 104.0],
            "state_z_m": [100.0, 100.0, 104.0],
            "predicted_sigma_m": [50.0, 50.0, 50.0],
        }
    )


def test_uncertainty_estimate_spec_accepts_sigma_column() -> None:
    spec = parse_uncertainty_estimate_spec("mix=/tmp/estimates.csv@0.75:predicted_sigma_m")
    assert spec.label == "mix"
    assert str(spec.path).endswith("estimates.csv")
    assert spec.weight == pytest.approx(0.75)
    assert spec.sigma_column == "predicted_sigma_m"


def test_uncertainty_ensemble_prefers_low_sigma_estimate() -> None:
    ensemble, diagnostics = build_track5_uncertainty_estimate_ensemble(
        [
            ("low", _low_sigma_estimate(), 1.0, "predicted_sigma_m"),
            ("high", _high_sigma_estimate(), 1.0, "predicted_sigma_m"),
        ],
        _template(),
        sigma_min_m=1.0,
        sigma_max_m=100.0,
    )

    midpoint = ensemble.loc[
        (ensemble["sequence_id"] == "seq0001") & (ensemble["time_s"] == 5.0)
    ].iloc[0]
    assert midpoint["state_x_m"] == pytest.approx(5.039984006397441)
    assert midpoint["state_y_m"] == pytest.approx(0.03998400639744103)
    assert midpoint["state_z_m"] == pytest.approx(0.03998400639744103)
    assert midpoint["ensemble_source_count"] == 2
    assert diagnostics["valid_input_count"].tolist() == [2, 2, 2]


def test_uncertainty_ensemble_writes_upload_ready_outputs(tmp_path: Path) -> None:
    low_csv = tmp_path / "low.csv"
    high_csv = tmp_path / "high.csv"
    _low_sigma_estimate().to_csv(low_csv, index=False)
    _high_sigma_estimate().to_csv(high_csv, index=False)

    paths = write_track5_uncertainty_estimate_ensemble_outputs(
        estimate_inputs=[
            parse_uncertainty_estimate_spec(f"low={low_csv}:predicted_sigma_m"),
            parse_uncertainty_estimate_spec(f"high={high_csv}:predicted_sigma_m"),
        ],
        template=_template(),
        output_dir=tmp_path / "out",
        class_map={"seq0001": "2", "seq0002": "1"},
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
    low_csv = tmp_path / "low.csv"
    high_csv = tmp_path / "high.csv"
    template_csv = tmp_path / "template.csv"
    class_map_csv = tmp_path / "class_map.csv"
    output_dir = tmp_path / "out"
    _low_sigma_estimate().to_csv(low_csv, index=False)
    _high_sigma_estimate().to_csv(high_csv, index=False)
    _template().to_csv(template_csv, index=False)
    pd.DataFrame({"sequence_id": ["seq0001", "seq0002"], "uav_type": [2, 1]}).to_csv(
        class_map_csv,
        index=False,
    )

    status = uncertainty_main(
        [
            "--estimate-csv",
            f"low={low_csv}:predicted_sigma_m",
            "--estimate-csv",
            f"high={high_csv}:predicted_sigma_m",
            "--template",
            str(template_csv),
            "--class-map",
            str(class_map_csv),
            "--output-dir",
            str(output_dir),
            "--require-leaderboard-ready",
        ]
    )

    assert status == 0
    assert (output_dir / "ug2_submission.zip").exists()
    assert (output_dir / "mmuad_track5_uncertainty_ensemble_manifest.json").exists()


def test_uncertainty_ensemble_entrypoint_is_exposed() -> None:
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    assert (
        pyproject["project"]["scripts"]["raft-uav-mmuad-track5-uncertainty-ensemble"]
        == "raft_uav.mmuad.track5_uncertainty_estimate_ensemble:main"
    )
