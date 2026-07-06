from __future__ import annotations

import json
from pathlib import Path
import tomllib

import pandas as pd
import pytest

from raft_uav.mmuad.track5_estimate_sequence_gate_fit import fit_estimate_sequence_gate_weights
from raft_uav.mmuad.track5_estimate_sequence_gate_fit import main as sequence_gate_fit_main
from raft_uav.mmuad.track5_estimate_sequence_gate_fit import write_estimate_sequence_gate_fit_outputs


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


def _base_estimates() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["seqA", "seqA", "seqB", "seqB"],
            "time_s": [0.0, 1.0, 0.0, 1.0],
            "state_x_m": [0.0, 1.0, 20.0, 21.0],
            "state_y_m": [0.0, 0.0, 0.0, 0.0],
            "state_z_m": [0.0, 0.0, 0.0, 0.0],
        }
    )


def _alternate_estimates() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["seqA", "seqA", "seqB", "seqB"],
            "time_s": [0.0, 1.0, 0.0, 1.0],
            "state_x_m": [5.0, 6.0, 10.0, 11.0],
            "state_y_m": [0.0, 0.0, 0.0, 0.0],
            "state_z_m": [0.0, 0.0, 0.0, 0.0],
        }
    )


def test_estimate_sequence_gate_fit_finds_sequence_oracle_weights() -> None:
    result = fit_estimate_sequence_gate_weights(
        base_estimates=_base_estimates(),
        alternate_estimates=_alternate_estimates(),
        template=_template(),
        truth=_truth(),
        weight_grid=pd.Series([0.0, 0.5, 1.0]).to_numpy(),
    )

    weights = dict(
        zip(
            result["oracle_weights"]["sequence_id"],
            result["oracle_weights"]["sequence_gate_weight"],
            strict=True,
        )
    )
    assert weights["seqA"] == pytest.approx(0.0)
    assert weights["seqB"] == pytest.approx(1.0)
    oracle = result["summary"].loc[result["summary"]["model"] == "oracle_same_split"].iloc[0]
    assert oracle["metric_pose_mse_m2"] == pytest.approx(0.0)
    assert set(result["loso_weights"].columns).issuperset(
        {"sequence_id", "sequence_gate_weight", "nearest_train_sequence_id"}
    )


def test_estimate_sequence_gate_fit_predicts_apply_weights() -> None:
    apply_template = pd.DataFrame(
        {
            "Sequence": ["seqC", "seqC"],
            "Timestamp": [0.0, 1.0],
            "Position": ["(0,0,0)"] * 2,
            "Classification": [0, 0],
        }
    )
    apply_base = pd.DataFrame(
        {
            "sequence_id": ["seqC", "seqC"],
            "time_s": [0.0, 1.0],
            "state_x_m": [0.0, 1.0],
            "state_y_m": [0.0, 0.0],
            "state_z_m": [0.0, 0.0],
        }
    )
    apply_alt = pd.DataFrame(
        {
            "sequence_id": ["seqC", "seqC"],
            "time_s": [0.0, 1.0],
            "state_x_m": [5.0, 6.0],
            "state_y_m": [0.0, 0.0],
            "state_z_m": [0.0, 0.0],
        }
    )
    result = fit_estimate_sequence_gate_weights(
        base_estimates=_base_estimates(),
        alternate_estimates=_alternate_estimates(),
        template=pd.concat([_template(), apply_template], ignore_index=True),
        truth=_truth(),
        apply_base_estimates=apply_base,
        apply_alternate_estimates=apply_alt,
    )

    assert "apply_weights" in result
    assert result["apply_weights"]["sequence_id"].tolist() == ["seqC"]
    assert result["apply_weights"]["sequence_gate_weight"].between(0.0, 1.0).all()


def test_estimate_sequence_gate_fit_writes_artifacts(tmp_path: Path) -> None:
    base = tmp_path / "base.csv"
    alternate = tmp_path / "alternate.csv"
    template = tmp_path / "template.csv"
    truth = tmp_path / "truth.csv"
    _base_estimates().to_csv(base, index=False)
    _alternate_estimates().to_csv(alternate, index=False)
    _template().to_csv(template, index=False)
    _truth().to_csv(truth, index=False)
    result = fit_estimate_sequence_gate_weights(
        base_estimates=_base_estimates(),
        alternate_estimates=_alternate_estimates(),
        template=_template(),
        truth=_truth(),
    )
    paths = write_estimate_sequence_gate_fit_outputs(
        result=result,
        output_dir=tmp_path / "out",
        base_estimates_path=base,
        alternate_estimates_path=alternate,
        template_path=template,
        truth_path=truth,
        weight_grid=pd.Series([0.0, 0.25, 0.5, 0.75, 1.0]).to_numpy(),
    )

    assert paths["summary_csv"].exists()
    assert paths["oracle_weights_csv"].exists()
    payload = json.loads(paths["summary_json"].read_text(encoding="utf-8"))
    assert payload["train_sequence_count"] == 2


def test_estimate_sequence_gate_fit_cli_writes_outputs(tmp_path: Path) -> None:
    base = tmp_path / "base.csv"
    alternate = tmp_path / "alternate.csv"
    template = tmp_path / "template.csv"
    truth = tmp_path / "truth.csv"
    _base_estimates().to_csv(base, index=False)
    _alternate_estimates().to_csv(alternate, index=False)
    _template().to_csv(template, index=False)
    _truth().to_csv(truth, index=False)

    status = sequence_gate_fit_main(
        [
            "--base-estimates",
            str(base),
            "--alternate-estimates",
            str(alternate),
            "--template",
            str(template),
            "--truth-csv",
            str(truth),
            "--output-dir",
            str(tmp_path / "cli"),
            "--weight-grid",
            "0,0.5,1",
        ]
    )

    assert status == 0
    assert (tmp_path / "cli" / "mmuad_track5_estimate_sequence_gate_fit_summary.csv").exists()
    assert (tmp_path / "cli" / "mmuad_track5_estimate_sequence_gate_oracle_weights.csv").exists()


def test_estimate_sequence_gate_fit_entrypoint_is_exposed() -> None:
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    assert (
        pyproject["project"]["scripts"]["raft-uav-mmuad-track5-estimate-sequence-gate-fit"]
        == "raft_uav.mmuad.track5_estimate_sequence_gate_fit:main"
    )
