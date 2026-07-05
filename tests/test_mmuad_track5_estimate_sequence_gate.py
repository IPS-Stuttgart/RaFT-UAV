from __future__ import annotations

import json
from pathlib import Path
import tomllib

import pandas as pd
import pytest

from raft_uav.mmuad.track5_estimate_sequence_gate import blend_track5_estimate_sequence_gate
from raft_uav.mmuad.track5_estimate_sequence_gate import main as sequence_gate_main
from raft_uav.mmuad.track5_estimate_sequence_gate import write_track5_estimate_sequence_gate_outputs


def _template() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Sequence": ["seq0001", "seq0001", "seq0002"],
            "Timestamp": [0.0, 5.0, 0.0],
            "Position": ["(0,0,0)"] * 3,
            "Classification": [2, 2, 1],
        }
    )


def _base_estimates() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["seq0001", "seq0001", "seq0002"],
            "time_s": [0.0, 10.0, 0.0],
            "state_x_m": [0.0, 10.0, 4.0],
            "state_y_m": [0.0, 0.0, 4.0],
            "state_z_m": [0.0, 0.0, 4.0],
        }
    )


def _alternate_estimates() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["seq0001", "seq0001", "seq0002"],
            "time_s": [0.0, 10.0, 0.0],
            "state_x_m": [2.0, 12.0, 8.0],
            "state_y_m": [2.0, 2.0, 8.0],
            "state_z_m": [2.0, 2.0, 8.0],
        }
    )


def _weights() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["seq0001", "seq0002"],
            "weight": [0.25, 1.0],
        }
    )


def test_estimate_sequence_gate_blends_after_template_resample() -> None:
    estimates, diagnostics, weights = blend_track5_estimate_sequence_gate(
        base_estimates=_base_estimates(),
        alternate_estimates=_alternate_estimates(),
        template=_template(),
        sequence_weights=_weights(),
    )

    midpoint = estimates.loc[
        (estimates["sequence_id"] == "seq0001") & (estimates["time_s"] == 5.0)
    ].iloc[0]
    assert midpoint["state_x_m"] == pytest.approx(5.5)
    assert midpoint["state_y_m"] == pytest.approx(0.5)
    assert midpoint["state_z_m"] == pytest.approx(0.5)
    seq2 = estimates.loc[estimates["sequence_id"] == "seq0002"].iloc[0]
    assert seq2["state_x_m"] == pytest.approx(8.0)
    assert diagnostics["sequence_gate_weight"].tolist() == [0.25, 0.25, 1.0]
    assert weights.set_index("sequence_id").loc["seq0002", "sequence_gate_weight"] == 1.0


def test_estimate_sequence_gate_writes_leaderboard_ready_outputs(tmp_path: Path) -> None:
    estimates, diagnostics, weights = blend_track5_estimate_sequence_gate(
        base_estimates=_base_estimates(),
        alternate_estimates=_alternate_estimates(),
        template=_template(),
        sequence_weights=_weights(),
    )
    paths = write_track5_estimate_sequence_gate_outputs(
        estimates=estimates,
        diagnostics=diagnostics,
        sequence_weights=weights,
        output_dir=tmp_path,
        base_estimates_path=tmp_path / "base.csv",
        alternate_estimates_path=tmp_path / "alternate.csv",
        sequence_weights_path=tmp_path / "weights.csv",
        template=_template(),
        class_map={"seq0001": "2", "seq0002": "1"},
        require_leaderboard_ready=True,
    )

    assert paths["zip"].exists()
    validation = json.loads(paths["validation_json"].read_text(encoding="utf-8"))
    manifest = json.loads(paths["manifest_json"].read_text(encoding="utf-8"))
    assert validation["leaderboard_ready"] is True
    assert validation["codabench_upload_ready"] is True
    assert manifest["sequence_count"] == 2
    official = pd.read_csv(paths["results_csv"])
    assert official["Classification"].tolist() == [2, 2, 1]


def test_estimate_sequence_gate_cli_writes_artifacts(tmp_path: Path) -> None:
    base_csv = tmp_path / "base.csv"
    alternate_csv = tmp_path / "alternate.csv"
    weights_csv = tmp_path / "weights.csv"
    template_csv = tmp_path / "template.csv"
    class_map_csv = tmp_path / "class_map.csv"
    output_dir = tmp_path / "out"
    _base_estimates().to_csv(base_csv, index=False)
    _alternate_estimates().to_csv(alternate_csv, index=False)
    _weights().to_csv(weights_csv, index=False)
    _template().to_csv(template_csv, index=False)
    pd.DataFrame({"sequence_id": ["seq0001", "seq0002"], "uav_type": [2, 1]}).to_csv(
        class_map_csv,
        index=False,
    )

    status = sequence_gate_main(
        [
            "--base-estimates",
            str(base_csv),
            "--alternate-estimates",
            str(alternate_csv),
            "--sequence-weights",
            str(weights_csv),
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
    assert (output_dir / "mmuad_track5_estimate_sequence_gate_manifest.json").exists()


def test_estimate_sequence_gate_entrypoint_is_exposed() -> None:
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    assert (
        pyproject["project"]["scripts"]["raft-uav-mmuad-track5-estimate-sequence-gate"]
        == "raft_uav.mmuad.track5_estimate_sequence_gate:main"
    )
