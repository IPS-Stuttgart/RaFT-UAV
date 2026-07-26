from __future__ import annotations

import gzip
from pathlib import Path

import pytest

from raft_uav.mmuad import estimate_csv


def test_guarded_estimate_csv_reader_forwards_explicit_compression(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "estimate.data"
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        handle.write("Sequence,state_x_m\n001,1.0\n")

    monkeypatch.setattr(estimate_csv, "_called_from_track5_estimate_grid", lambda: True)
    monkeypatch.setattr(
        estimate_csv,
        "_called_from_candidate_reservoir_cli",
        lambda: False,
    )

    rows = estimate_csv._read_csv_with_track5_estimate_grid_guard(
        path,
        compression="gzip",
    )

    assert rows.to_dict("records") == [
        {"Sequence": "001", "state_x_m": "1.0"}
    ]
