from __future__ import annotations

import json
from pathlib import Path
import tomllib
from zipfile import ZipFile

import pandas as pd
import pytest

from raft_uav.mmuad.track5_consensus_estimate_ensemble import (
    build_consensus_estimate_ensemble,
    main as consensus_main,
    write_consensus_estimate_ensemble_outputs,
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


def _estimate_good() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["seq0001", "seq0001", "seq0002"],
            "time_s": [0.0, 10.0, 0.0],
            "state_x_m": [0.0, 10.0, 5.0],
            "state_y_m": [0.0, 0.0, 5.0],
            "state_z_m": [0.0, 0.0, 5.0],
        }
    )


def _estimate_near() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["seq0001", "seq0001", "seq0002"],
            "time_s": [0.0, 10.0, 0.0],
            "state_x_m": [1.0, 11.0, 6.0],
            "state_y_m": [0.0, 0.0, 5.0],
            "state_z_m": [0.0, 0.0, 5.0],
        }
    )


def _estimate_outlier() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["seq0001", "seq0001", "seq0002"],
            "time_s": [0.0, 10.0, 0.0],
            "state_x_m": [100.0, 110.0, 100.0],
            "state_y_m": [0.0, 0.0, 100.0],
            "state_z_m": [0.0, 0.0, 100.0],
        }
    )


def test_consensus_ensemble_downweights_far_outlier() -> None:
    ensemble, diagnostics = build_consensus_estimate_ensemble(
        [
            ("good", _estimate_good(), 1.0),
            ("near", _estimate_near(), 1.0),
            ("outlier", _estimate_outlier(), 1.0),
        ],
        _template(),
        distance_floor_m=1.0,
        distance_power=2.0,
    )

    midpoint = ensemble.loc[
        (ensemble["sequence_id"] == "seq0001") & (ensemble["time_s"] == 5.0)
    ].iloc[0]
    assert 5.0 <= midpoint["state_x_m"] < 8.0
    assert midpoint["ensemble_source_count"] == 3
    assert midpoint["mean_consensus_distance_m"] > 20.0
    assert diagnostics["valid_input_count"].tolist() == [3, 3, 3]


def test_consensus_ensemble_writes_upload_ready_artifacts(tmp_path: Path) -> None:
    good_csv = tmp_path / "good.csv"
    near_csv = tmp_path / "near.csv"
    outlier_csv = tmp_path / "outlier.csv"
    _estimate_good().to_csv(good_csv, index=False)
    _estimate_near().to_csv(near_csv, index=False)
    _estimate_outlier().to_csv(outlier_csv, index=False)

    paths = write_consensus_estimate_ensemble_outputs(
        estimate_inputs=[
            parse_estimate_spec(f"good={good_csv}"),
            parse_estimate_spec(f"near={near_csv}"),
            parse_estimate_spec(f"outlier={outlier_csv}"),
        ],
        template=_template(),
        output_dir=tmp_path / "out",
        class_map={"seq0001": "2", "seq0002": "1"},
        distance_power=2.0,
    )

    assert paths["official_zip"].exists()
    validation = json.loads(paths["validation_json"].read_text(encoding="utf-8"))
    manifest = json.loads(paths["manifest_json"].read_text(encoding="utf-8"))
    assert validation["leaderboard_ready"] is True
    assert validation["codabench_upload_ready"] is True
    assert manifest["valid_ensemble_rows"] == 3
    official = pd.read_csv(paths["official_results_csv"])
    assert official["Classification"].tolist() == [2, 2, 1]
    with ZipFile(paths["official_zip"]) as archive:
        assert archive.namelist() == ["mmaud_results.csv"]


def test_consensus_ensemble_cli_writes_outputs(tmp_path: Path) -> None:
    good_csv = tmp_path / "good.csv"
    near_csv = tmp_path / "near.csv"
    template_csv = tmp_path / "template.csv"
    class_map_csv = tmp_path / "class_map.csv"
    output_dir = tmp_path / "out"
    _estimate_good().to_csv(good_csv, index=False)
    _estimate_near().to_csv(near_csv, index=False)
    _template().to_csv(template_csv, index=False)
    pd.DataFrame({"sequence_id": ["seq0001", "seq0002"], "uav_type": [2, 1]}).to_csv(
        class_map_csv,
        index=False,
    )

    status = consensus_main(
        [
            "--estimate-csv",
            f"good={good_csv}",
            "--estimate-csv",
            f"near={near_csv}",
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
    assert (output_dir / "mmuad_track5_consensus_ensemble_manifest.json").exists()


def test_consensus_ensemble_entrypoint_is_exposed() -> None:
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    assert (
        pyproject["project"]["scripts"]["raft-uav-mmuad-track5-consensus-ensemble"]
        == "raft_uav.mmuad.track5_consensus_estimate_ensemble:main"
    )
