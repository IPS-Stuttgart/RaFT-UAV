from __future__ import annotations

import json
from pathlib import Path
import tomllib
from zipfile import ZipFile

import pandas as pd
import pytest

from raft_uav.mmuad.track5_sequence_gate import blend_track5_sequence_gate
from raft_uav.mmuad.track5_sequence_gate import main as sequence_gate_main
from raft_uav.mmuad.track5_sequence_gate import write_track5_sequence_gate_outputs
from raft_uav.mmuad.track5_submission_ensemble import load_track5_submission


def _submission_rows(offset: float = 0.0, classification: int = 1) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Sequence": ["seq0001", "seq0001", "seq0002"],
            "Timestamp": [0.0, 1.0, 0.0],
            "Position": [
                f"({0.0 + offset}, 0.0, 1.0)",
                f"({2.0 + offset}, 0.0, 1.0)",
                f"({10.0 + offset}, 1.0, 2.0)",
            ],
            "Classification": [classification, classification, 2],
        }
    )


def _template_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Sequence": ["seq0001", "seq0001", "seq0002"],
            "Timestamp": [0.0, 1.0, 0.0],
            "Position": ["(0, 0, 0)"] * 3,
            "Classification": [1, 1, 2],
        }
    )


def test_sequence_gate_blends_positions_by_sequence_and_preserves_base_classes(
    tmp_path: Path,
) -> None:
    base_path = tmp_path / "base.csv"
    alternate_path = tmp_path / "alternate.csv"
    _submission_rows(offset=0.0, classification=1).to_csv(base_path, index=False)
    _submission_rows(offset=4.0, classification=3).to_csv(alternate_path, index=False)
    weights = pd.DataFrame({"sequence_id": ["seq0001"], "blend_weight": [0.25]})

    result = blend_track5_sequence_gate(
        base_submission=load_track5_submission(base_path),
        alternate_submission=load_track5_submission(alternate_path),
        sequence_weights=weights,
        default_weight=0.0,
    )

    seq1 = result.estimates.loc[
        (result.estimates["sequence_id"] == "seq0001")
        & (result.estimates["time_s"] == 0.0)
    ].iloc[0]
    seq2 = result.estimates.loc[result.estimates["sequence_id"] == "seq0002"].iloc[0]
    assert seq1["state_x_m"] == pytest.approx(1.0)
    assert seq2["state_x_m"] == pytest.approx(10.0)
    assert result.estimates["Classification"].tolist() == [1, 1, 2]
    assert result.diagnostics["weight_source"].tolist() == [
        "sequence_weights",
        "sequence_weights",
        "default",
    ]


def test_sequence_gate_normalizes_sequence_weight_ids(tmp_path: Path) -> None:
    base_path = tmp_path / "base.csv"
    alternate_path = tmp_path / "alternate.csv"
    _submission_rows(offset=0.0, classification=1).to_csv(base_path, index=False)
    _submission_rows(offset=4.0, classification=3).to_csv(alternate_path, index=False)
    weights = pd.DataFrame(
        {
            "Sequence": [" seq0002 ", None, "nan"],
            "weight": [1.0, 0.25, 0.75],
        }
    )

    result = blend_track5_sequence_gate(
        base_submission=load_track5_submission(base_path),
        alternate_submission=load_track5_submission(alternate_path),
        sequence_weights=weights,
        default_weight=0.0,
    )

    seq2 = result.estimates.loc[result.estimates["sequence_id"] == "seq0002"].iloc[0]
    assert seq2["state_x_m"] == pytest.approx(14.0)
    assert seq2["sequence_gate_weight"] == pytest.approx(1.0)
    assert result.diagnostics["weight_source"].tolist() == [
        "default",
        "default",
        "sequence_weights",
    ]
    assert result.sequence_weights["sequence_id"].tolist() == ["seq0001", "seq0002"]
    assert result.sequence_weights["weight_source"].tolist() == ["default", "sequence_weights"]


def test_sequence_gate_rejects_out_of_range_weights(tmp_path: Path) -> None:
    base_path = tmp_path / "base.csv"
    alternate_path = tmp_path / "alternate.csv"
    _submission_rows().to_csv(base_path, index=False)
    _submission_rows(offset=1.0).to_csv(alternate_path, index=False)

    with pytest.raises(ValueError, match=r"in \[0, 1\]"):
        blend_track5_sequence_gate(
            base_submission=load_track5_submission(base_path),
            alternate_submission=load_track5_submission(alternate_path),
            sequence_weights=pd.DataFrame({"Sequence": ["seq0001"], "weight": [1.5]}),
        )


def test_sequence_gate_writes_leaderboard_ready_zip(tmp_path: Path) -> None:
    base_path = tmp_path / "base.csv"
    alternate_path = tmp_path / "alternate.csv"
    base_rows = _submission_rows(offset=0.0, classification=1)
    base_rows.loc[1, "Classification"] = 2
    base_rows.to_csv(base_path, index=False)
    _submission_rows(offset=2.0, classification=3).to_csv(alternate_path, index=False)
    result = blend_track5_sequence_gate(
        base_submission=load_track5_submission(base_path),
        alternate_submission=load_track5_submission(alternate_path),
        sequence_weights=pd.DataFrame({"Sequence": ["seq0001", "seq0002"], "weight": [0.5, 1.0]}),
    )

    paths = write_track5_sequence_gate_outputs(
        result=result,
        output_dir=tmp_path / "out",
        base_submission_path=base_path,
        alternate_submission_path=alternate_path,
        sequence_weights_path=tmp_path / "weights.csv",
        template=_template_rows(),
        require_leaderboard_ready=True,
    )

    assert paths["zip"].exists()
    assert paths["results_csv"].exists()
    validation = json.loads(paths["validation_json"].read_text(encoding="utf-8"))
    assert validation["leaderboard_ready"] is True
    with ZipFile(paths["zip"]) as archive:
        assert archive.namelist() == ["mmaud_results.csv"]
    results = pd.read_csv(paths["results_csv"])
    assert results["Classification"].tolist() == [1, 2, 2]


def test_sequence_gate_cli_writes_manifest(tmp_path: Path) -> None:
    base_path = tmp_path / "base.csv"
    alternate_path = tmp_path / "alternate.csv"
    weights_path = tmp_path / "weights.csv"
    template_path = tmp_path / "template.csv"
    _submission_rows(offset=0.0, classification=1).to_csv(base_path, index=False)
    _submission_rows(offset=2.0, classification=3).to_csv(alternate_path, index=False)
    pd.DataFrame({"Sequence": ["seq0001"], "weight": [0.5]}).to_csv(weights_path, index=False)
    _template_rows().to_csv(template_path, index=False)
    output_dir = tmp_path / "out"

    status = sequence_gate_main(
        [
            "--base-submission",
            str(base_path),
            "--alternate-submission",
            str(alternate_path),
            "--sequence-weights",
            str(weights_path),
            "--template",
            str(template_path),
            "--output-dir",
            str(output_dir),
            "--require-leaderboard-ready",
        ]
    )

    assert status == 0
    manifest = json.loads(
        (output_dir / "mmuad_track5_sequence_gate_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["row_count"] == 3
    assert manifest["defaulted_sequence_count"] == 1
    assert manifest["validation"]["leaderboard_ready"] is True


def test_sequence_gate_cli_preserves_zero_padded_sequence_ids(tmp_path: Path) -> None:
    base_path = tmp_path / "base.csv"
    alternate_path = tmp_path / "alternate.csv"
    weights_path = tmp_path / "weights.csv"
    output_dir = tmp_path / "out"
    base_rows = pd.DataFrame(
        {
            "Sequence": ["001"],
            "Timestamp": [0.0],
            "Position": ["(0.0, 0.0, 0.0)"],
            "Classification": [1],
        }
    )
    alternate_rows = base_rows.copy()
    alternate_rows["Position"] = ["(4.0, 0.0, 0.0)"]
    base_rows.to_csv(base_path, index=False)
    alternate_rows.to_csv(alternate_path, index=False)
    pd.DataFrame({"Sequence": ["001"], "weight": [1.0]}).to_csv(weights_path, index=False)

    status = sequence_gate_main(
        [
            "--base-submission",
            str(base_path),
            "--alternate-submission",
            str(alternate_path),
            "--sequence-weights",
            str(weights_path),
            "--output-dir",
            str(output_dir),
        ]
    )

    assert status == 0
    estimates = pd.read_csv(output_dir / "mmuad_track5_sequence_gate_estimates.csv", dtype={"sequence_id": "string"})
    diagnostics = pd.read_csv(
        output_dir / "mmuad_track5_sequence_gate_diagnostics.csv",
        dtype={"sequence_id": "string"},
    )
    manifest = json.loads(
        (output_dir / "mmuad_track5_sequence_gate_manifest.json").read_text(encoding="utf-8")
    )
    assert estimates.loc[0, "sequence_id"] == "001"
    assert estimates.loc[0, "state_x_m"] == pytest.approx(4.0)
    assert diagnostics.loc[0, "weight_source"] == "sequence_weights"
    assert manifest["defaulted_sequence_count"] == 0


def test_sequence_gate_cli_accepts_padded_weight_headers(tmp_path: Path) -> None:
    base_path = tmp_path / "base.csv"
    alternate_path = tmp_path / "alternate.csv"
    weights_path = tmp_path / "weights.csv"
    output_dir = tmp_path / "out"
    base_rows = pd.DataFrame(
        {
            "Sequence": ["001"],
            "Timestamp": [0.0],
            "Position": ["(0.0, 0.0, 0.0)"],
            "Classification": [1],
        }
    )
    alternate_rows = base_rows.copy()
    alternate_rows["Position"] = ["(4.0, 0.0, 0.0)"]
    base_rows.to_csv(base_path, index=False)
    alternate_rows.to_csv(alternate_path, index=False)
    weights_path.write_text(" Sequence , weight \n001,1.0\n", encoding="utf-8")

    status = sequence_gate_main(
        [
            "--base-submission",
            str(base_path),
            "--alternate-submission",
            str(alternate_path),
            "--sequence-weights",
            str(weights_path),
            "--output-dir",
            str(output_dir),
        ]
    )

    assert status == 0
    diagnostics = pd.read_csv(
        output_dir / "mmuad_track5_sequence_gate_diagnostics.csv",
        dtype={"sequence_id": "string"},
    )
    assert diagnostics.loc[0, "sequence_id"] == "001"
    assert diagnostics.loc[0, "weight_source"] == "sequence_weights"
    assert diagnostics.loc[0, "sequence_gate_weight"] == pytest.approx(1.0)


def test_sequence_gate_cli_preserves_padded_template_sequence_ids(tmp_path: Path) -> None:
    base_path = tmp_path / "base.csv"
    alternate_path = tmp_path / "alternate.csv"
    weights_path = tmp_path / "weights.csv"
    template_path = tmp_path / "template.csv"
    output_dir = tmp_path / "out"
    base_rows = pd.DataFrame(
        {
            "Sequence": ["001"],
            "Timestamp": [0.0],
            "Position": ["(0.0, 0.0, 0.0)"],
            "Classification": [1],
        }
    )
    alternate_rows = base_rows.copy()
    alternate_rows["Position"] = ["(4.0, 0.0, 0.0)"]
    base_rows.to_csv(base_path, index=False)
    alternate_rows.to_csv(alternate_path, index=False)
    pd.DataFrame({"Sequence": ["001"], "weight": [1.0]}).to_csv(weights_path, index=False)
    template_path.write_text(
        " Sequence , Timestamp , Position , Classification \n"
        "001,0.0,\"(0,0,0)\",1\n",
        encoding="utf-8",
    )

    status = sequence_gate_main(
        [
            "--base-submission",
            str(base_path),
            "--alternate-submission",
            str(alternate_path),
            "--sequence-weights",
            str(weights_path),
            "--template",
            str(template_path),
            "--output-dir",
            str(output_dir),
            "--require-leaderboard-ready",
        ]
    )

    assert status == 0
    validation = json.loads(
        (output_dir / "mmuad_track5_sequence_gate_validation.json").read_text(encoding="utf-8")
    )
    assert validation["leaderboard_ready"] is True
    assert set(validation["sequences"]) == {"001"}
    assert validation["sequences"]["001"]["missing_template_timestamp_count"] == 0
    assert validation["sequences"]["001"]["extra_prediction_count"] == 0


def test_sequence_gate_entrypoint_is_exposed() -> None:
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    assert (
        pyproject["project"]["scripts"]["raft-uav-mmuad-track5-sequence-gate"]
        == "raft_uav.mmuad.track5_sequence_gate:main"
    )
