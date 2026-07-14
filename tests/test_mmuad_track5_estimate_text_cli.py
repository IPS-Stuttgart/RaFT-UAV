from __future__ import annotations

from io import StringIO
from pathlib import Path
import tomllib

import pandas as pd
import pytest

import raft_uav.mmuad.track5_estimate_text_cli as estimate_text_cli
from raft_uav.mmuad.track5_estimate_text_cli import _read_csv_preserving_sequence_id


def test_estimate_fit_wrapper_preserves_normalized_sequence_ids(tmp_path: Path) -> None:
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


def test_estimate_fit_wrapper_preserves_schema_sequence_aliases(tmp_path: Path) -> None:
    csv_path = tmp_path / "scene_alias.csv"
    pd.DataFrame(
        {
            "scene_id": ["001"],
            "time_s": [0.0],
            "state_x_m": [1.0],
            "state_y_m": [2.0],
            "state_z_m": [3.0],
        }
    ).to_csv(csv_path, index=False)

    rows = _read_csv_preserving_sequence_id(csv_path, converters={"scene_id": int})

    assert rows.loc[0, "scene_id"] == "001"


def test_estimate_fit_wrapper_rewinds_file_like_csv_after_header_probe() -> None:
    csv_stream = StringIO(
        "sequence_id,time_s,state_x_m,state_y_m,state_z_m\n"
        "001,0.0,1.0,2.0,3.0\n"
    )

    rows = _read_csv_preserving_sequence_id(csv_stream)

    assert rows.loc[0, "sequence_id"] == "001"
    assert rows.loc[0, "time_s"] == 0.0


def test_estimate_fit_wrapper_accepts_scalar_dtype_without_coercing_sequence_ids(
    tmp_path: Path,
) -> None:
    csv_path = tmp_path / "normalized.csv"
    pd.DataFrame(
        {
            "sequence_id": ["001"],
            "time_s": [1.25],
            "state_x_m": [1.0],
            "state_y_m": [2.0],
            "state_z_m": [3.0],
        }
    ).to_csv(csv_path, index=False)

    rows = _read_csv_preserving_sequence_id(csv_path, dtype=float)

    assert rows.loc[0, "sequence_id"] == "001"
    assert rows.loc[0, "time_s"] == 1.25


def test_estimate_fit_wrapper_ignores_sequence_converters_that_coerce_ids(
    tmp_path: Path,
) -> None:
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

    rows = _read_csv_preserving_sequence_id(csv_path, converters={"sequence_id": int})

    assert rows.loc[0, "sequence_id"] == "001"


def test_estimate_fit_main_scopes_text_reader_to_implementation_module(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    csv_path = tmp_path / "typed.csv"
    csv_path.write_text("sequence_id,value\n001,1\n", encoding="utf-8")
    original_global_reader = pd.read_csv
    original_impl_pd = estimate_text_cli._impl.pd
    observations: dict[str, object] = {}

    def fake_main(_argv: list[str] | None = None) -> int:
        observations["global_reader_is_original"] = pd.read_csv is original_global_reader
        observations["global_value"] = pd.read_csv(csv_path).loc[0, "value"]
        observations["legacy_sequence_id"] = estimate_text_cli._impl.pd.read_csv(
            csv_path
        ).loc[0, "sequence_id"]
        return 0

    monkeypatch.setattr(estimate_text_cli._impl, "main", fake_main)

    assert estimate_text_cli.main([]) == 0
    assert observations["global_reader_is_original"] is True
    assert observations["global_value"] == 1
    assert observations["legacy_sequence_id"] == "001"
    assert pd.read_csv is original_global_reader
    assert estimate_text_cli._impl.pd is original_impl_pd


def test_estimate_fit_console_script_uses_text_id_wrapper() -> None:
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))

    assert (
        pyproject["project"]["scripts"]["raft-uav-mmuad-track5-estimate-sequence-gate-fit"]
        == "raft_uav.mmuad.track5_estimate_text_cli:main"
    )
