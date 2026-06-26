from __future__ import annotations

import json
from pathlib import Path
import tomllib
from zipfile import ZipFile

import pandas as pd
import pytest

from raft_uav.mmuad.track5_estimate_ensemble import build_track5_estimate_ensemble
from raft_uav.mmuad.track5_estimate_ensemble import main as ensemble_main
from raft_uav.mmuad.track5_estimate_ensemble import parse_estimate_spec
from raft_uav.mmuad.track5_estimate_ensemble import write_track5_estimate_ensemble_outputs


def _template() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Sequence": ["seq0001", "seq0001", "seq0002"],
            "Timestamp": [0.0, 5.0, 0.0],
            "Position": ["(0,0,0)"] * 3,
            "Classification": [2, 2, 1],
        }
    )


def _estimate_a() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["seq0001", "seq0001", "seq0002"],
            "time_s": [0.0, 10.0, 0.0],
            "state_x_m": [0.0, 10.0, 4.0],
            "state_y_m": [0.0, 0.0, 4.0],
            "state_z_m": [0.0, 0.0, 4.0],
        }
    )


def _estimate_b() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["seq0001", "seq0001", "seq0002"],
            "time_s": [0.0, 10.0, 0.0],
            "state_x_m": [2.0, 12.0, 8.0],
            "state_y_m": [2.0, 2.0, 8.0],
            "state_z_m": [2.0, 2.0, 8.0],
        }
    )


def test_parse_estimate_spec_accepts_label_path_and_weight() -> None:
    item = parse_estimate_spec("robust=/tmp/estimates.csv@0.25")
    assert item.label == "robust"
    assert str(item.path).endswith("estimates.csv")
    assert item.weight == pytest.approx(0.25)


def test_track5_estimate_ensemble_weighted_average_after_template_resample() -> None:
    ensemble, diagnostics = build_track5_estimate_ensemble(
        [
            ("a", _estimate_a(), 0.75),
            ("b", _estimate_b(), 0.25),
        ],
        _template(),
    )

    midpoint = ensemble.loc[
        (ensemble["sequence_id"] == "seq0001") & (ensemble["time_s"] == 5.0)
    ].iloc[0]
    assert midpoint["state_x_m"] == pytest.approx(5.5)
    assert midpoint["state_y_m"] == pytest.approx(0.5)
    assert midpoint["state_z_m"] == pytest.approx(0.5)
    seq2 = ensemble.loc[ensemble["sequence_id"] == "seq0002"].iloc[0]
    assert seq2["state_x_m"] == pytest.approx(5.0)
    assert seq2["ensemble_source_count"] == 2
    assert diagnostics["valid_input_count"].tolist() == [2, 2, 2]


def test_track5_estimate_ensemble_writes_leaderboard_ready_artifacts(tmp_path: Path) -> None:
    a_csv = tmp_path / "a.csv"
    b_csv = tmp_path / "b.csv"
    _estimate_a().to_csv(a_csv, index=False)
    _estimate_b().to_csv(b_csv, index=False)
    paths = write_track5_estimate_ensemble_outputs(
        estimate_inputs=[
            parse_estimate_spec(f"a={a_csv}@0.75"),
            parse_estimate_spec(f"b={b_csv}@0.25"),
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


def test_track5_estimate_ensemble_cli_writes_outputs(tmp_path: Path) -> None:
    a_csv = tmp_path / "a.csv"
    b_csv = tmp_path / "b.csv"
    template_csv = tmp_path / "template.csv"
    class_map_csv = tmp_path / "class_map.csv"
    output_dir = tmp_path / "out"
    _estimate_a().to_csv(a_csv, index=False)
    _estimate_b().to_csv(b_csv, index=False)
    _template().to_csv(template_csv, index=False)
    pd.DataFrame({"sequence_id": ["seq0001", "seq0002"], "uav_type": [2, 1]}).to_csv(
        class_map_csv,
        index=False,
    )

    status = ensemble_main(
        [
            "--estimate-csv",
            f"a={a_csv}@0.75",
            "--estimate-csv",
            f"b={b_csv}@0.25",
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
    assert (output_dir / "mmuad_track5_ensemble_manifest.json").exists()


def test_track5_estimate_ensemble_entrypoint_is_exposed() -> None:
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    assert (
        pyproject["project"]["scripts"]["raft-uav-mmuad-track5-estimate-ensemble"]
        == "raft_uav.mmuad.track5_estimate_ensemble:main"
    )
