from __future__ import annotations

import json
from pathlib import Path
import tomllib
from zipfile import ZipFile

import pandas as pd
import pytest

from raft_uav.mmuad.track5_segment_source_gate import SegmentSourceGateConfig
from raft_uav.mmuad.track5_segment_source_gate import build_track5_segment_source_gate
from raft_uav.mmuad.track5_segment_source_gate import main as segment_gate_main
from raft_uav.mmuad.track5_segment_source_gate import write_track5_segment_source_gate_outputs
from raft_uav.mmuad.track5_estimate_ensemble import parse_estimate_spec


def _template() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Sequence": ["seq0001"] * 5,
            "Timestamp": [0.0, 1.0, 2.0, 3.0, 4.0],
            "Position": ["(0,0,0)"] * 5,
            "Classification": [2] * 5,
        }
    )


def _smooth_estimate() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["seq0001"] * 5,
            "time_s": [0.0, 1.0, 2.0, 3.0, 4.0],
            "state_x_m": [0.0, 1.0, 2.0, 3.0, 4.0],
            "state_y_m": [0.0] * 5,
            "state_z_m": [1.0] * 5,
        }
    )


def _spiky_estimate() -> pd.DataFrame:
    rows = _smooth_estimate()
    rows.loc[2, "state_x_m"] = 200.0
    return rows


def test_segment_source_gate_rejects_local_dynamics_spike() -> None:
    estimates, diagnostics = build_track5_segment_source_gate(
        [
            ("spiky", _spiky_estimate(), 1.0),
            ("smooth", _smooth_estimate(), 0.8),
        ],
        _template(),
        config=SegmentSourceGateConfig(
            speed_limit_mps=20.0,
            acceleration_limit_mps2=20.0,
            switch_penalty=0.5,
            switch_jump_penalty_per_m=0.0,
        ),
    )

    assert estimates["selected_source_label"].tolist() == ["smooth"] * 5
    assert estimates["state_x_m"].tolist() == pytest.approx([0.0, 1.0, 2.0, 3.0, 4.0])
    assert diagnostics["valid_source_count"].tolist() == [2, 2, 2, 2, 2]


def test_segment_source_gate_weight_prior_prefers_higher_weight_smooth_duplicate() -> None:
    estimates, _ = build_track5_segment_source_gate(
        [
            ("primary", _smooth_estimate(), 1.0),
            ("secondary", _smooth_estimate(), 0.2),
        ],
        _template(),
        config=SegmentSourceGateConfig(
            speed_limit_mps=20.0,
            acceleration_limit_mps2=20.0,
            switch_penalty=0.5,
        ),
    )

    assert estimates["selected_source_label"].tolist() == ["primary"] * 5


def test_segment_source_gate_writes_upload_ready_artifacts(tmp_path: Path) -> None:
    spiky_csv = tmp_path / "spiky.csv"
    smooth_csv = tmp_path / "smooth.csv"
    _spiky_estimate().to_csv(spiky_csv, index=False)
    _smooth_estimate().to_csv(smooth_csv, index=False)
    paths = write_track5_segment_source_gate_outputs(
        estimate_inputs=[
            parse_estimate_spec(f"spiky={spiky_csv}@1.0"),
            parse_estimate_spec(f"smooth={smooth_csv}@0.8"),
        ],
        template=_template(),
        output_dir=tmp_path / "out",
        class_map={"seq0001": "2"},
        config=SegmentSourceGateConfig(
            speed_limit_mps=20.0,
            acceleration_limit_mps2=20.0,
            switch_penalty=0.5,
            switch_jump_penalty_per_m=0.0,
        ),
    )

    validation = json.loads(paths["validation_json"].read_text(encoding="utf-8"))
    manifest = json.loads(paths["manifest_json"].read_text(encoding="utf-8"))
    assert validation["leaderboard_ready"] is True
    assert validation["codabench_upload_ready"] is True
    assert manifest["row_count"] == 5
    with ZipFile(paths["official_zip"]) as archive:
        assert archive.namelist() == ["mmaud_results.csv"]
    official = pd.read_csv(paths["official_results_csv"])
    assert official["Classification"].tolist() == [2, 2, 2, 2, 2]


def test_segment_source_gate_cli_writes_outputs(tmp_path: Path) -> None:
    spiky_csv = tmp_path / "spiky.csv"
    smooth_csv = tmp_path / "smooth.csv"
    template_csv = tmp_path / "template.csv"
    class_map_csv = tmp_path / "class_map.csv"
    output_dir = tmp_path / "out"
    _spiky_estimate().to_csv(spiky_csv, index=False)
    _smooth_estimate().to_csv(smooth_csv, index=False)
    _template().to_csv(template_csv, index=False)
    pd.DataFrame({"sequence_id": ["seq0001"], "uav_type": [2]}).to_csv(
        class_map_csv,
        index=False,
    )

    status = segment_gate_main(
        [
            "--estimate-csv",
            f"spiky={spiky_csv}@1.0",
            "--estimate-csv",
            f"smooth={smooth_csv}@0.8",
            "--template",
            str(template_csv),
            "--class-map",
            str(class_map_csv),
            "--output-dir",
            str(output_dir),
            "--speed-limit-mps",
            "20",
            "--acceleration-limit-mps2",
            "20",
            "--switch-penalty",
            "0.5",
            "--switch-jump-penalty-per-m",
            "0",
            "--require-leaderboard-ready",
        ]
    )

    assert status == 0
    assert (output_dir / "ug2_submission.zip").exists()
    assert (output_dir / "mmuad_track5_segment_source_gate_manifest.json").exists()


def test_segment_source_gate_entrypoint_is_exposed() -> None:
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    assert (
        pyproject["project"]["scripts"]["raft-uav-mmuad-track5-segment-source-gate"]
        == "raft_uav.mmuad.track5_segment_source_gate:main"
    )
