from __future__ import annotations

import json
from pathlib import Path
import tomllib

import pandas as pd
import pytest
import numpy as np

from raft_uav.mmuad.track5_sequence_gate_fit import _make_numpy_model
from raft_uav.mmuad.track5_sequence_gate_fit import fit_track5_sequence_gate
from raft_uav.mmuad.track5_sequence_gate_fit import main as sequence_gate_fit_main
from raft_uav.mmuad.track5_sequence_gate_fit import write_track5_sequence_gate_fit_outputs
from raft_uav.mmuad.track5_submission_ensemble import load_track5_submission


def _rows(sequence_offsets: dict[str, float]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for sequence, offset in sequence_offsets.items():
        for timestamp, x in ((0.0, 0.0), (1.0, 2.0)):
            rows.append(
                {
                    "Sequence": sequence,
                    "Timestamp": timestamp,
                    "Position": f"({x + offset}, 0.0, 1.0)",
                    "Classification": 1,
                }
            )
    return pd.DataFrame(rows)


def _write_fit_inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    base_path = tmp_path / "base.csv"
    alternate_path = tmp_path / "alternate.csv"
    truth_path = tmp_path / "truth.csv"
    _rows({"seq0001": 0.0, "seq0002": 0.0, "seq0003": 0.0}).to_csv(
        base_path,
        index=False,
    )
    _rows({"seq0001": 10.0, "seq0002": 10.0, "seq0003": 10.0}).to_csv(
        alternate_path,
        index=False,
    )
    _rows({"seq0001": 5.0, "seq0002": 2.5, "seq0003": 0.0}).to_csv(
        truth_path,
        index=False,
    )
    return base_path, alternate_path, truth_path


def test_sequence_gate_fit_finds_oracle_sequence_weights(tmp_path: Path) -> None:
    base_path, alternate_path, truth_path = _write_fit_inputs(tmp_path)

    result = fit_track5_sequence_gate(
        base_submission=load_track5_submission(base_path),
        alternate_submission=load_track5_submission(alternate_path),
        truth=load_track5_submission(truth_path),
        weight_grid=pd.Series([0.0, 0.25, 0.5]).to_numpy(float),
        models=("tree_d1_leaf1",),
    )

    weights = dict(
        zip(
            result.oracle_weights["sequence_id"],
            result.oracle_weights["oracle_weight"],
            strict=True,
        )
    )
    assert weights["seq0001"] == pytest.approx(0.5)
    assert weights["seq0002"] == pytest.approx(0.25)
    assert weights["seq0003"] == pytest.approx(0.0)
    assert set(result.same_split_weights.columns) == {"sequence_id", "blend_weight"}
    assert set(result.loso_weights.columns) == {"sequence_id", "blend_weight"}
    assert len(result.summary) == 1


def test_sequence_gate_fit_writes_summary_and_weight_tables(tmp_path: Path) -> None:
    base_path, alternate_path, truth_path = _write_fit_inputs(tmp_path)
    result = fit_track5_sequence_gate(
        base_submission=load_track5_submission(base_path),
        alternate_submission=load_track5_submission(alternate_path),
        truth=load_track5_submission(truth_path),
        weight_grid=pd.Series([0.0, 0.25, 0.5]).to_numpy(float),
        models=("ridge",),
    )

    paths = write_track5_sequence_gate_fit_outputs(
        result=result,
        output_dir=tmp_path / "out",
        base_submission_path=base_path,
        alternate_submission_path=alternate_path,
        truth_path=truth_path,
        weight_grid=pd.Series([0.0, 0.25, 0.5]).to_numpy(float),
        protocol="unit-test",
    )

    payload = json.loads(paths["summary_json"].read_text(encoding="utf-8"))
    assert payload["schema"] == "raft-uav-mmuad-track5-sequence-gate-fit-v1"
    assert payload["protocol"] == "unit-test"
    assert paths["summary_csv"].exists()
    assert pd.read_csv(paths["oracle_weights_csv"])["oracle_weight"].tolist() == [
        0.5,
        0.25,
        0.0,
    ]


def test_sequence_gate_fit_cli_writes_outputs(tmp_path: Path) -> None:
    base_path, alternate_path, truth_path = _write_fit_inputs(tmp_path)
    output_dir = tmp_path / "out"

    status = sequence_gate_fit_main(
        [
            "--base-submission",
            str(base_path),
            "--alternate-submission",
            str(alternate_path),
            "--truth",
            str(truth_path),
            "--output-dir",
            str(output_dir),
            "--weight-min",
            "0.0",
            "--weight-max",
            "0.5",
            "--weight-step",
            "0.25",
            "--model",
            "tree_d1_leaf1",
        ]
    )

    assert status == 0
    assert (output_dir / "mmuad_track5_sequence_gate_fit_summary.csv").exists()
    assert (output_dir / "mmuad_track5_sequence_gate_loso_weights.csv").exists()


def test_sequence_gate_fit_numpy_fallback_models_fit_and_predict() -> None:
    x = np.asarray(
        [
            [0.0, 0.0],
            [1.0, 0.2],
            [2.0, 0.8],
            [3.0, 1.0],
        ]
    )
    y = np.asarray([0.0, 0.1, 0.4, 0.5])

    for name in ("ridge", "tree_d1_leaf1", "rf_depth2", "extra_depth2"):
        model = _make_numpy_model(name, random_state=7)
        model.fit(x, y)
        predicted = model.predict(x)

        assert predicted.shape == (4,)
        assert np.isfinite(predicted).all()


def test_sequence_gate_fit_entrypoint_is_exposed() -> None:
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    assert (
        pyproject["project"]["scripts"]["raft-uav-mmuad-track5-sequence-gate-fit"]
        == "raft_uav.mmuad.track5_sequence_gate_fit:main"
    )
