from __future__ import annotations

import json
from pathlib import Path
import tomllib
from zipfile import ZipFile

import pandas as pd
import pytest

from raft_uav.mmuad.track5_class_conditioned_ensemble import (
    build_class_conditioned_estimate_ensemble,
    main as class_ensemble_main,
    search_class_conditioned_ensemble_weights,
    write_class_conditioned_ensemble_outputs,
)
from raft_uav.mmuad.track5_estimate_ensemble import EstimateInput


def _template() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Sequence": ["seqA", "seqA", "seqB", "seqB"],
            "Timestamp": [0.0, 1.0, 0.0, 1.0],
            "Position": ["(0,0,0)"] * 4,
            "Classification": [0, 0, 1, 1],
        }
    )


def _truth() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["seqA", "seqA", "seqB", "seqB"],
            "time_s": [0.0, 1.0, 0.0, 1.0],
            "x_m": [0.0, 1.0, 10.0, 11.0],
            "y_m": [0.0, 0.0, 0.0, 0.0],
            "z_m": [0.0, 0.0, 0.0, 0.0],
        }
    )


def _estimate_a() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["seqA", "seqA", "seqB", "seqB"],
            "time_s": [0.0, 1.0, 0.0, 1.0],
            "state_x_m": [0.0, 1.0, 20.0, 21.0],
            "state_y_m": [0.0, 0.0, 0.0, 0.0],
            "state_z_m": [0.0, 0.0, 0.0, 0.0],
        }
    )


def _estimate_b() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["seqA", "seqA", "seqB", "seqB"],
            "time_s": [0.0, 1.0, 0.0, 1.0],
            "state_x_m": [8.0, 9.0, 10.0, 11.0],
            "state_y_m": [0.0, 0.0, 0.0, 0.0],
            "state_z_m": [0.0, 0.0, 0.0, 0.0],
        }
    )


def _write_inputs(tmp_path: Path) -> tuple[Path, Path, Path, Path, Path]:
    a_csv = tmp_path / "a.csv"
    b_csv = tmp_path / "b.csv"
    template_csv = tmp_path / "template.csv"
    truth_csv = tmp_path / "truth.csv"
    class_map_csv = tmp_path / "class_map.csv"
    _estimate_a().to_csv(a_csv, index=False)
    _estimate_b().to_csv(b_csv, index=False)
    _template().to_csv(template_csv, index=False)
    _truth().to_csv(truth_csv, index=False)
    pd.DataFrame({"sequence_id": ["seqA", "seqB"], "uav_type": [0, 1]}).to_csv(
        class_map_csv,
        index=False,
    )
    return a_csv, b_csv, template_csv, truth_csv, class_map_csv


def test_class_conditioned_weight_search_selects_different_weights_by_class(tmp_path: Path) -> None:
    a_csv, b_csv, _, _, _ = _write_inputs(tmp_path)
    grid, config = search_class_conditioned_ensemble_weights(
        [EstimateInput("a", a_csv), EstimateInput("b", b_csv)],
        template=_template(),
        truth=_truth(),
        class_map={"seqA": "0", "seqB": "1"},
        weight_step=0.5,
    )

    assert not grid.empty
    assert config["class_weights"]["0"] == {"a": 1.0, "b": 0.0}
    assert config["class_weights"]["1"] == {"a": 0.0, "b": 1.0}
    assert config["metrics"]["0"]["pose_mse_m2"] == pytest.approx(0.0)
    assert config["metrics"]["1"]["pose_mse_m2"] == pytest.approx(0.0)


def test_class_conditioned_ensemble_applies_per_class_weights(tmp_path: Path) -> None:
    a_csv, b_csv, _, _, _ = _write_inputs(tmp_path)
    config = {
        "global_weights": {"a": 0.5, "b": 0.5},
        "class_weights": {"0": {"a": 1.0, "b": 0.0}, "1": {"a": 0.0, "b": 1.0}},
    }
    estimates, diagnostics = build_class_conditioned_estimate_ensemble(
        [EstimateInput("a", a_csv), EstimateInput("b", b_csv)],
        template=_template(),
        class_map={"seqA": "0", "seqB": "1"},
        weight_config=config,
    )

    assert estimates["state_x_m"].tolist() == [0.0, 1.0, 10.0, 11.0]
    assert set(diagnostics["class_conditioned_ensemble_class"]) == {"0", "1"}


def test_class_conditioned_ensemble_writes_upload_ready_outputs(tmp_path: Path) -> None:
    a_csv, b_csv, _, _, _ = _write_inputs(tmp_path)
    paths = write_class_conditioned_ensemble_outputs(
        estimate_inputs=[EstimateInput("a", a_csv), EstimateInput("b", b_csv)],
        template=_template(),
        class_map={"seqA": "0", "seqB": "1"},
        weight_config={
            "global_weights": {"a": 0.5, "b": 0.5},
            "class_weights": {"0": {"a": 1.0, "b": 0.0}, "1": {"a": 0.0, "b": 1.0}},
        },
        output_dir=tmp_path / "out",
        default_classification=0,
    )

    assert paths["official_zip"].exists()
    validation = json.loads(paths["validation_json"].read_text(encoding="utf-8"))
    assert validation["leaderboard_ready"] is True
    with ZipFile(paths["official_zip"]) as archive:
        assert archive.namelist() == ["mmaud_results.csv"]


def test_class_conditioned_ensemble_cli_search_and_write(tmp_path: Path) -> None:
    a_csv, b_csv, template_csv, truth_csv, class_map_csv = _write_inputs(tmp_path)
    status = class_ensemble_main(
        [
            "--estimate-csv",
            f"a={a_csv}",
            "--estimate-csv",
            f"b={b_csv}",
            "--template",
            str(template_csv),
            "--truth-csv",
            str(truth_csv),
            "--class-map",
            str(class_map_csv),
            "--output-dir",
            str(tmp_path / "out"),
            "--weight-step",
            "0.5",
            "--write-submission",
            "--require-leaderboard-ready",
        ]
    )

    assert status == 0
    assert (tmp_path / "out" / "mmuad_track5_class_ensemble_weights.json").exists()
    assert (tmp_path / "out" / "class_conditioned_submission" / "ug2_submission.zip").exists()


def test_class_conditioned_ensemble_entrypoint_is_exposed() -> None:
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    assert (
        pyproject["project"]["scripts"]["raft-uav-mmuad-track5-class-ensemble"]
        == "raft_uav.mmuad.track5_class_conditioned_ensemble:main"
    )
