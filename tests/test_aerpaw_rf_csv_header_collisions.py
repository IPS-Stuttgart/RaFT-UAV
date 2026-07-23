"""Regression tests for ambiguous raw Keysight RF CSV headers."""

from __future__ import annotations

from pathlib import Path

import pytest

from raft_uav.io.aerpaw import read_rf_csv


@pytest.mark.parametrize(
    "header",
    [
        "Time,Time,Latitude,Longitude",
        "Time, Time ,Latitude,Longitude",
    ],
)
def test_read_rf_csv_rejects_duplicate_trimmed_physical_headers(
    tmp_path: Path,
    header: str,
) -> None:
    rf_path = tmp_path / "rf.csv"
    rf_path.write_text(
        f"{header}\n2025-10-07 19:42:20.000,conflict,35.72749,-78.69621\n",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match=r"RF CSV has duplicate columns after trimming whitespace: 'Time'",
    ):
        read_rf_csv(rf_path)


def test_read_rf_csv_preserves_unique_header_whitespace_normalization(
    tmp_path: Path,
) -> None:
    rf_path = tmp_path / "rf.csv"
    rf_path.write_text(
        " Time , Latitude , Longitude \n"
        "2025-10-07 19:42:20.000,35.72749,-78.69621\n",
        encoding="utf-8",
    )

    frame = read_rf_csv(rf_path)

    assert frame.columns.tolist() == ["Time", "Latitude", "Longitude"]
