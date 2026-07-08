from __future__ import annotations

from pathlib import Path
import tomllib

import pandas as pd

from raft_uav.mmuad import track5_sequence_gate_fit as _impl
from raft_uav.mmuad.track5_sequence_gate_fit_text_cli import _read_csv_preserving_sequence_id


def test_sequence_gate_fit_wrapper_preserves_normalized_sequence_ids(tmp_path: Path) -> None:
    csv_path = tmp_path / "normalized.csv"
    pd.DataFrame(
        {
            "sequence_id": ["001"],
            "time_s": [0.0],
            "state_x_m": [1.0],
            "state_y_m": [2.0],
            "state_z_m": [3.0],
        }
    ).to_csv(csv_path, index=False)

    rows = _read_csv_preserving_sequence_id(csv_path)

    assert rows.loc[0, "sequence_id"] == "001"


def test_sequence_gate_fit_wrapper_accepts_scalar_dtype(tmp_path: Path) -> None:
    csv_path = tmp_path / "normalized.csv"
    pd.DataFrame(
        {
            "sequence_id": ["001"],
            "time_s": [0.0],
            "state_x_m": [1.0],
            "state_y_m": [2.0],
            "state_z_m": [3.0],
        }
    ).to_csv(csv_path, index=False)

    rows = _read_csv_preserving_sequence_id(csv_path, dtype=str)

    assert rows.loc[0, "sequence_id"] == "001"
    assert rows.loc[0, "time_s"] == "0.0"


def test_sequence_gate_fit_wrapper_overrides_sequence_dtype_mapping(tmp_path: Path) -> None:
    csv_path = tmp_path / "normalized.csv"
    pd.DataFrame({"Sequence": ["001"], "value": [4]}).to_csv(csv_path, index=False)

    rows = _read_csv_preserving_sequence_id(csv_path, dtype={"Sequence": int, "value": int})

    assert str(rows.loc[0, "Sequence"]) == "001"
    assert rows.loc[0, "value"] == 4


def test_sequence_gate_fit_wrapper_strips_padded_normalized_headers(
    tmp_path: Path,
    monkeypatch,
) -> None:
    csv_path = tmp_path / "truth_normalized_padded.csv"
    csv_path.write_text(
        " sequence_id , time_s , x_m , y_m , z_m , uav_type \n"
        "001,0.0,1.0,2.0,3.0,4\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(_impl.pd, "read_csv", _read_csv_preserving_sequence_id)

    rows = _impl._load_track5_gate_rows(csv_path)

    assert rows["sequence_id"].tolist() == ["001"]
    assert rows["time_s"].tolist() == [0.0]
    assert rows[["state_x_m", "state_y_m", "state_z_m"]].iloc[0].tolist() == [
        1.0,
        2.0,
        3.0,
    ]
    assert rows["Classification"].tolist() == [4]


def test_sequence_gate_fit_console_script_uses_text_id_wrapper() -> None:
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))

    assert (
        pyproject["project"]["scripts"]["raft-uav-mmuad-track5-sequence-gate-fit"]
        == "raft_uav.mmuad.track5_sequence_gate_fit_text_cli:main"
    )
