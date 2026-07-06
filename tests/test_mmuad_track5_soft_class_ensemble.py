from __future__ import annotations

import json
from pathlib import Path
import tomllib
from zipfile import ZipFile

import pandas as pd
import pytest

from raft_uav.mmuad.track5_estimate_ensemble import EstimateInput
from raft_uav.mmuad.track5_soft_class_ensemble import (
    build_soft_class_conditioned_estimate_ensemble,
    main as soft_class_main,
    write_soft_class_conditioned_ensemble_outputs,
)


def _template() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Sequence": ["seq0001", "seq0002"],
            "Timestamp": [0.0, 0.0],
            "Position": ["(0,0,0)", "(0,0,0)"],
            "Classification": [0, 1],
        }
    )


def _estimate_a() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["seq0001", "seq0002"],
            "time_s": [0.0, 0.0],
            "state_x_m": [0.0, 10.0],
            "state_y_m": [0.0, 10.0],
            "state_z_m": [0.0, 10.0],
        }
    )


def _estimate_b() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["seq0001", "seq0002"],
            "time_s": [0.0, 0.0],
            "state_x_m": [100.0, 20.0],
            "state_y_m": [100.0, 20.0],
            "state_z_m": [100.0, 20.0],
        }
    )


def _probabilities() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["seq0001", "seq0002"],
            "predicted_probability_0": [0.75, 0.10],
            "predicted_probability_1": [0.25, 0.90],
            "predicted_probability_2": [0.0, 0.0],
            "predicted_probability_3": [0.0, 0.0],
        }
    )


def _weight_config() -> dict[str, object]:
    return {
        "schema": "test-class-conditioned",
        "aggregation_policy": "weighted-mean",
        "trim_fraction": 0.2,
        "global_weights": {"a": 1.0, "b": 0.0},
        "class_weights": {
            "0": {"a": 1.0, "b": 0.0},
            "1": {"a": 0.0, "b": 1.0},
        },
    }


def test_soft_class_ensemble_blends_class_specific_pose_hypotheses(tmp_path: Path) -> None:
    a_csv = tmp_path / "a.csv"
    b_csv = tmp_path / "b.csv"
    _estimate_a().to_csv(a_csv, index=False)
    _estimate_b().to_csv(b_csv, index=False)

    estimates, diagnostics = build_soft_class_conditioned_estimate_ensemble(
        [EstimateInput("a", a_csv), EstimateInput("b", b_csv)],
        template=_template(),
        class_probabilities=_probabilities(),
        weight_config=_weight_config(),
    )

    seq1 = estimates.loc[estimates["sequence_id"] == "seq0001"].iloc[0]
    seq2 = estimates.loc[estimates["sequence_id"] == "seq0002"].iloc[0]
    assert seq1["state_x_m"] == pytest.approx(25.0)
    assert seq1["state_y_m"] == pytest.approx(25.0)
    assert seq2["state_x_m"] == pytest.approx(19.0)
    assert seq2["state_z_m"] == pytest.approx(19.0)
    assert estimates["soft_class_probability_available"].all()
    assert diagnostics["effective_probability_sum"].tolist() == pytest.approx([1.0, 1.0])


def test_soft_class_ensemble_writes_upload_ready_artifacts(tmp_path: Path) -> None:
    a_csv = tmp_path / "a.csv"
    b_csv = tmp_path / "b.csv"
    _estimate_a().to_csv(a_csv, index=False)
    _estimate_b().to_csv(b_csv, index=False)

    paths = write_soft_class_conditioned_ensemble_outputs(
        estimate_inputs=[EstimateInput("a", a_csv), EstimateInput("b", b_csv)],
        template=_template(),
        class_probabilities=_probabilities(),
        weight_config=_weight_config(),
        output_dir=tmp_path / "out",
        class_map={"seq0001": "0", "seq0002": "1"},
    )

    validation = json.loads(paths["validation_json"].read_text(encoding="utf-8"))
    manifest = json.loads(paths["manifest_json"].read_text(encoding="utf-8"))
    assert validation["leaderboard_ready"] is True
    assert validation["codabench_upload_ready"] is True
    assert manifest["class_probability_sequence_count"] == 2
    with ZipFile(paths["official_zip"]) as archive:
        assert archive.namelist() == ["mmaud_results.csv"]


def test_soft_class_ensemble_cli_writes_outputs(tmp_path: Path) -> None:
    a_csv = tmp_path / "a.csv"
    b_csv = tmp_path / "b.csv"
    template_csv = tmp_path / "template.csv"
    probs_csv = tmp_path / "probabilities.csv"
    weights_json = tmp_path / "weights.json"
    class_map_csv = tmp_path / "class_map.csv"
    output_dir = tmp_path / "out"
    _estimate_a().to_csv(a_csv, index=False)
    _estimate_b().to_csv(b_csv, index=False)
    _template().to_csv(template_csv, index=False)
    _probabilities().to_csv(probs_csv, index=False)
    weights_json.write_text(json.dumps(_weight_config()), encoding="utf-8")
    pd.DataFrame({"sequence_id": ["seq0001", "seq0002"], "uav_type": [0, 1]}).to_csv(
        class_map_csv,
        index=False,
    )

    status = soft_class_main(
        [
            "--estimate-csv",
            f"a={a_csv}",
            "--estimate-csv",
            f"b={b_csv}",
            "--template",
            str(template_csv),
            "--class-probabilities-csv",
            str(probs_csv),
            "--weight-config-json",
            str(weights_json),
            "--class-map",
            str(class_map_csv),
            "--output-dir",
            str(output_dir),
            "--require-leaderboard-ready",
        ]
    )

    assert status == 0
    assert (output_dir / "mmuad_track5_soft_class_ensemble_estimates.csv").exists()
    assert (output_dir / "ug2_submission.zip").exists()


def test_soft_class_ensemble_entrypoint_is_exposed() -> None:
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    assert (
        pyproject["project"]["scripts"]["raft-uav-mmuad-track5-soft-class-ensemble"]
        == "raft_uav.mmuad.track5_soft_class_ensemble:main"
    )
