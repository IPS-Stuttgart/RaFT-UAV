from __future__ import annotations

import json
from pathlib import Path
import tomllib
from zipfile import ZipFile

import pandas as pd
import pytest

from raft_uav.mmuad.track5_agreement_gated_ensemble import (
    build_agreement_gated_track5_ensemble,
)
from raft_uav.mmuad.track5_agreement_gated_ensemble import main as gated_main
from raft_uav.mmuad.track5_agreement_gated_ensemble import (
    write_agreement_gated_track5_ensemble_outputs,
)
from raft_uav.mmuad.track5_estimate_ensemble import parse_estimate_spec


def _template() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Sequence": ["seq0001", "seq0001", "seq0002"],
            "Timestamp": [0.0, 5.0, 0.0],
            "Position": ["(0,0,0)"] * 3,
            "Classification": [2, 2, 1],
        }
    )


def _primary_estimate() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["seq0001", "seq0001", "seq0002"],
            "time_s": [0.0, 10.0, 0.0],
            "state_x_m": [0.0, 10.0, 4.0],
            "state_y_m": [0.0, 0.0, 4.0],
            "state_z_m": [0.0, 0.0, 4.0],
        }
    )


def _nearby_estimate() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["seq0001", "seq0001", "seq0002"],
            "time_s": [0.0, 10.0, 0.0],
            "state_x_m": [1.0, 11.0, 5.0],
            "state_y_m": [0.0, 0.0, 4.0],
            "state_z_m": [0.0, 0.0, 4.0],
        }
    )


def _outlier_estimate() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["seq0001", "seq0001", "seq0002"],
            "time_s": [0.0, 10.0, 0.0],
            "state_x_m": [1000.0, 1010.0, 1000.0],
            "state_y_m": [1000.0, 1000.0, 1000.0],
            "state_z_m": [1000.0, 1000.0, 1000.0],
        }
    )


def test_agreement_gated_ensemble_averages_when_inputs_agree() -> None:
    estimates, diagnostics = build_agreement_gated_track5_ensemble(
        [
            ("primary", _primary_estimate(), 1.0),
            ("nearby", _nearby_estimate(), 1.0),
        ],
        _template(),
        spread_gate_m=10.0,
        primary_label="primary",
    )

    midpoint = estimates.loc[
        (estimates["sequence_id"] == "seq0001") & (estimates["time_s"] == 5.0)
    ].iloc[0]
    assert midpoint["state_x_m"] == pytest.approx(5.5)
    assert midpoint["agreement_action"] == "weighted_mean"
    assert diagnostics["agreement_action"].tolist() == ["weighted_mean", "weighted_mean", "weighted_mean"]


def test_agreement_gated_ensemble_falls_back_when_inputs_disagree() -> None:
    estimates, diagnostics = build_agreement_gated_track5_ensemble(
        [
            ("primary", _primary_estimate(), 1.0),
            ("outlier", _outlier_estimate(), 1.0),
        ],
        _template(),
        spread_gate_m=10.0,
        primary_label="primary",
    )

    midpoint = estimates.loc[
        (estimates["sequence_id"] == "seq0001") & (estimates["time_s"] == 5.0)
    ].iloc[0]
    assert midpoint["state_x_m"] == pytest.approx(5.0)
    assert midpoint["state_y_m"] == pytest.approx(0.0)
    assert midpoint["agreement_action"] == "primary_fallback"
    assert midpoint["agreement_selected_label"] == "primary"
    assert diagnostics.loc[1, "position_spread_m"] > 100.0


def test_agreement_gated_ensemble_writes_leaderboard_ready_artifacts(tmp_path: Path) -> None:
    primary_csv = tmp_path / "primary.csv"
    nearby_csv = tmp_path / "nearby.csv"
    _primary_estimate().to_csv(primary_csv, index=False)
    _nearby_estimate().to_csv(nearby_csv, index=False)
    paths = write_agreement_gated_track5_ensemble_outputs(
        estimate_inputs=[
            parse_estimate_spec(f"primary={primary_csv}@1.0"),
            parse_estimate_spec(f"nearby={nearby_csv}@1.0"),
        ],
        template=_template(),
        output_dir=tmp_path / "out",
        class_map={"seq0001": "2", "seq0002": "1"},
        spread_gate_m=10.0,
        primary_label="primary",
    )

    validation = json.loads(paths["validation_json"].read_text(encoding="utf-8"))
    manifest = json.loads(paths["manifest_json"].read_text(encoding="utf-8"))
    assert validation["leaderboard_ready"] is True
    assert validation["codabench_upload_ready"] is True
    assert manifest["agreement_action_counts"] == {"weighted_mean": 3}
    with ZipFile(paths["official_zip"]) as archive:
        assert archive.namelist() == ["mmaud_results.csv"]
    official = pd.read_csv(paths["official_results_csv"])
    assert official["Classification"].tolist() == [2, 2, 1]


def test_agreement_gated_ensemble_cli_writes_outputs(tmp_path: Path) -> None:
    primary_csv = tmp_path / "primary.csv"
    outlier_csv = tmp_path / "outlier.csv"
    template_csv = tmp_path / "template.csv"
    class_map_csv = tmp_path / "class_map.csv"
    output_dir = tmp_path / "out"
    _primary_estimate().to_csv(primary_csv, index=False)
    _outlier_estimate().to_csv(outlier_csv, index=False)
    _template().to_csv(template_csv, index=False)
    pd.DataFrame({"sequence_id": ["seq0001", "seq0002"], "uav_type": [2, 1]}).to_csv(
        class_map_csv,
        index=False,
    )

    status = gated_main(
        [
            "--estimate-csv",
            f"primary={primary_csv}@1.0",
            "--estimate-csv",
            f"outlier={outlier_csv}@1.0",
            "--template",
            str(template_csv),
            "--class-map",
            str(class_map_csv),
            "--output-dir",
            str(output_dir),
            "--spread-gate-m",
            "10",
            "--primary-label",
            "primary",
            "--require-leaderboard-ready",
        ]
    )

    assert status == 0
    assert (output_dir / "ug2_submission.zip").exists()
    assert (output_dir / "mmuad_track5_agreement_gated_manifest.json").exists()


def test_agreement_gated_ensemble_entrypoint_is_exposed() -> None:
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    assert (
        pyproject["project"]["scripts"]["raft-uav-mmuad-track5-agreement-gated-ensemble"]
        == "raft_uav.mmuad.track5_agreement_gated_ensemble:main"
    )
