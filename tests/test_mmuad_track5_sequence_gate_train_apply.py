from __future__ import annotations

import json
from pathlib import Path
import tomllib
from zipfile import ZipFile

import pandas as pd

from raft_uav.mmuad.track5_sequence_gate_train_apply import main as train_apply_main
from raft_uav.mmuad.track5_sequence_gate_train_apply import run_track5_sequence_gate_train_apply


def _official_rows(sequence_offsets: dict[str, float], classification: int = 1) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for sequence, offset in sequence_offsets.items():
        for timestamp, x in ((0.0, 0.0), (1.0, 2.0)):
            rows.append(
                {
                    "Sequence": sequence,
                    "Timestamp": timestamp,
                    "Position": f"({x + offset}, 0.0, 1.0)",
                    "Classification": classification,
                }
            )
    return pd.DataFrame(rows)


def _normalized_truth(sequence_offsets: dict[str, float]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for sequence, offset in sequence_offsets.items():
        for timestamp, x in ((0.0, 0.0), (1.0, 2.0)):
            rows.append(
                {
                    "sequence_id": sequence,
                    "time_s": timestamp,
                    "x_m": x + offset,
                    "y_m": 0.0,
                    "z_m": 1.0,
                    "uav_type": 1,
                }
            )
    return pd.DataFrame(rows)


def _write_inputs(tmp_path: Path) -> dict[str, Path]:
    paths = {
        "train_base": tmp_path / "train_base.csv",
        "train_alternate": tmp_path / "train_alternate.csv",
        "train_truth": tmp_path / "train_truth.csv",
        "apply_base": tmp_path / "apply_base.csv",
        "apply_alternate": tmp_path / "apply_alternate.csv",
        "apply_truth": tmp_path / "apply_truth.csv",
    }
    _official_rows({"seq0001": 0.0, "seq0002": 0.0, "seq0003": 0.0}).to_csv(
        paths["train_base"],
        index=False,
    )
    _official_rows({"seq0001": 10.0, "seq0002": 10.0, "seq0003": 10.0}).to_csv(
        paths["train_alternate"],
        index=False,
    )
    _normalized_truth({"seq0001": 5.0, "seq0002": 2.5, "seq0003": 0.0}).to_csv(
        paths["train_truth"],
        index=False,
    )
    _official_rows({"seq0101": 0.0, "seq0102": 0.0}).to_csv(
        paths["apply_base"],
        index=False,
    )
    _official_rows({"seq0101": 10.0, "seq0102": 10.0}).to_csv(
        paths["apply_alternate"],
        index=False,
    )
    _official_rows({"seq0101": 5.0, "seq0102": 5.0}).to_csv(
        paths["apply_truth"],
        index=False,
    )
    return paths


def test_sequence_gate_train_apply_writes_zip_manifest_and_scorecard(tmp_path: Path) -> None:
    paths = _write_inputs(tmp_path)
    result = run_track5_sequence_gate_train_apply(
        train_base_submission_path=paths["train_base"],
        train_alternate_submission_path=paths["train_alternate"],
        train_truth_path=paths["train_truth"],
        apply_base_submission_path=paths["apply_base"],
        apply_alternate_submission_path=paths["apply_alternate"],
        apply_truth_path=paths["apply_truth"],
        output_dir=tmp_path / "out",
        weight_grid=pd.Series([0.0, 0.25, 0.5]).to_numpy(float),
        models=("ridge",),
        require_leaderboard_ready=True,
    )

    assert result.manifest_json.exists()
    assert result.fit_paths["apply_weights_csv"].exists()
    assert result.gate_paths["zip"].exists()
    assert result.scorecard_paths["scorecard_json"].exists()
    manifest = json.loads(result.manifest_json.read_text(encoding="utf-8"))
    assert manifest["schema"] == "raft-uav-mmuad-track5-sequence-gate-train-apply-v1"
    assert manifest["apply_sequence_count"] == 2
    with ZipFile(result.gate_paths["zip"]) as archive:
        assert archive.namelist() == ["mmaud_results.csv"]
    scorecard = json.loads(result.scorecard_paths["scorecard_json"].read_text(encoding="utf-8"))
    assert scorecard["scorecard_leaderboard_ready"] is True


def test_sequence_gate_train_apply_cli_writes_outputs(tmp_path: Path) -> None:
    paths = _write_inputs(tmp_path)
    output_dir = tmp_path / "cli_out"

    status = train_apply_main(
        [
            "--train-base-submission",
            str(paths["train_base"]),
            "--train-alternate-submission",
            str(paths["train_alternate"]),
            "--train-truth",
            str(paths["train_truth"]),
            "--apply-base-submission",
            str(paths["apply_base"]),
            "--apply-alternate-submission",
            str(paths["apply_alternate"]),
            "--apply-truth",
            str(paths["apply_truth"]),
            "--output-dir",
            str(output_dir),
            "--weight-step",
            "0.25",
            "--model",
            "ridge",
            "--require-leaderboard-ready",
        ]
    )

    assert status == 0
    assert (output_dir / "mmuad_track5_sequence_gate_train_apply_manifest.json").exists()
    assert (output_dir / "sequence_gate_apply" / "ug2_submission_sequence_gate.zip").exists()


def test_sequence_gate_train_apply_entrypoint_is_exposed() -> None:
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    assert (
        pyproject["project"]["scripts"]["raft-uav-mmuad-track5-sequence-gate-train-apply"]
        == "raft_uav.mmuad.track5_sequence_gate_train_apply:main"
    )
