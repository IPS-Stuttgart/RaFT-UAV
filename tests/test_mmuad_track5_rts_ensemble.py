from __future__ import annotations

import json
from pathlib import Path
import tomllib
from zipfile import ZipFile

import numpy as np
import pandas as pd
import pytest

from raft_uav.mmuad.track5_estimate_ensemble import parse_estimate_spec
from raft_uav.mmuad.track5_rts_ensemble import build_track5_rts_ensemble
from raft_uav.mmuad.track5_rts_ensemble import main as rts_main
from raft_uav.mmuad.track5_rts_ensemble import write_track5_rts_ensemble_outputs


def _template() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Sequence": ["seq0001"] * 5,
            "Timestamp": np.arange(5, dtype=float),
            "Position": ["(0,0,0)"] * 5,
            "Classification": [2] * 5,
        }
    )


def _good_estimate() -> pd.DataFrame:
    times = np.arange(5, dtype=float)
    return pd.DataFrame(
        {
            "sequence_id": ["seq0001"] * 5,
            "time_s": times,
            "state_x_m": times,
            "state_y_m": np.zeros(5),
            "state_z_m": np.ones(5),
        }
    )


def _spiky_estimate() -> pd.DataFrame:
    rows = _good_estimate().copy()
    rows.loc[2, "state_x_m"] = 22.0
    return rows


def test_rts_ensemble_downweights_disagreement_spike_temporally() -> None:
    estimates, diagnostics = build_track5_rts_ensemble(
        [
            ("good", _good_estimate(), 1.0),
            ("spiky", _spiky_estimate(), 1.0),
        ],
        _template(),
        measurement_sigma_m=1.0,
        process_accel_std_mps2=0.1,
        spread_variance_scale=10.0,
    )

    middle = estimates.loc[estimates["time_s"] == 2.0, "state_x_m"].iloc[0]
    weighted_middle = diagnostics.loc[diagnostics["time_s"] == 2.0, "weighted_x_m"].iloc[0]
    assert weighted_middle == pytest.approx(12.0)
    assert middle == pytest.approx(2.0, abs=2.0)
    assert diagnostics.loc[diagnostics["time_s"] == 2.0, "input_spread_m"].iloc[0] > 5.0


def test_rts_ensemble_writes_leaderboard_ready_artifacts(tmp_path: Path) -> None:
    good_csv = tmp_path / "good.csv"
    spiky_csv = tmp_path / "spiky.csv"
    _good_estimate().to_csv(good_csv, index=False)
    _spiky_estimate().to_csv(spiky_csv, index=False)
    paths = write_track5_rts_ensemble_outputs(
        estimate_inputs=[
            parse_estimate_spec(f"good={good_csv}@1.0"),
            parse_estimate_spec(f"spiky={spiky_csv}@1.0"),
        ],
        template=_template(),
        output_dir=tmp_path / "out",
        class_map={"seq0001": "2"},
        measurement_sigma_m=1.0,
        process_accel_std_mps2=0.1,
        spread_variance_scale=10.0,
    )

    validation = json.loads(paths["validation_json"].read_text(encoding="utf-8"))
    manifest = json.loads(paths["manifest_json"].read_text(encoding="utf-8"))
    assert validation["leaderboard_ready"] is True
    assert validation["codabench_upload_ready"] is True
    assert manifest["valid_rows"] == 5
    assert paths["official_zip"].exists()
    with ZipFile(paths["official_zip"]) as archive:
        assert archive.namelist() == ["mmaud_results.csv"]
    official = pd.read_csv(paths["official_results_csv"])
    assert official["Classification"].tolist() == [2, 2, 2, 2, 2]


def test_rts_ensemble_cli_writes_outputs(tmp_path: Path) -> None:
    good_csv = tmp_path / "good.csv"
    spiky_csv = tmp_path / "spiky.csv"
    template_csv = tmp_path / "template.csv"
    class_map_csv = tmp_path / "class_map.csv"
    output_dir = tmp_path / "out"
    _good_estimate().to_csv(good_csv, index=False)
    _spiky_estimate().to_csv(spiky_csv, index=False)
    _template().to_csv(template_csv, index=False)
    pd.DataFrame({"sequence_id": ["seq0001"], "uav_type": [2]}).to_csv(
        class_map_csv,
        index=False,
    )

    status = rts_main(
        [
            "--estimate-csv",
            f"good={good_csv}@1.0",
            "--estimate-csv",
            f"spiky={spiky_csv}@1.0",
            "--template",
            str(template_csv),
            "--class-map",
            str(class_map_csv),
            "--output-dir",
            str(output_dir),
            "--measurement-sigma-m",
            "1.0",
            "--process-accel-std-mps2",
            "0.1",
            "--spread-variance-scale",
            "10.0",
            "--require-leaderboard-ready",
        ]
    )

    assert status == 0
    assert (output_dir / "ug2_submission.zip").exists()
    assert (output_dir / "mmuad_track5_rts_ensemble_manifest.json").exists()


def test_rts_ensemble_entrypoint_is_exposed() -> None:
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    assert (
        pyproject["project"]["scripts"]["raft-uav-mmuad-track5-rts-ensemble"]
        == "raft_uav.mmuad.track5_rts_ensemble:main"
    )
