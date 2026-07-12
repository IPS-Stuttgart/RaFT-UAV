from __future__ import annotations

import importlib
from pathlib import Path

import pandas as pd

import raft_uav.mmuad.estimate_csv as estimate_csv


def test_estimate_csv_guard_survives_module_reload(tmp_path: Path) -> None:
    csv_path = tmp_path / "estimates.csv"
    csv_path.write_text(" sequence_id ,value\n001,2\n", encoding="utf-8")

    original_reader = getattr(
        pd.read_csv,
        estimate_csv._ORIGINAL_READER_ATTRIBUTE,
        pd.read_csv,
    )

    importlib.reload(estimate_csv)
    importlib.reload(estimate_csv)

    ordinary = pd.read_csv(csv_path)
    assert ordinary.columns.tolist() == [" sequence_id ", "value"]

    guarded = estimate_csv.read_estimate_csv(csv_path)
    assert guarded.columns.tolist() == ["sequence_id", "value"]
    assert guarded["sequence_id"].tolist() == ["001"]
    assert getattr(
        pd.read_csv,
        estimate_csv._ORIGINAL_READER_ATTRIBUTE,
    ) is original_reader
