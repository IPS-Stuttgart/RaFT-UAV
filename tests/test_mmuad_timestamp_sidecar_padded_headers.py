from __future__ import annotations

from pathlib import Path

import pytest

from raft_uav.mmuad.sequence import _timestamp_sidecar_time_for_file


def test_padded_filename_header_keeps_explicit_timestamp_mapping(tmp_path: Path) -> None:
    first = tmp_path / "a.pcd"
    second = tmp_path / "b.pcd"
    first.write_bytes(b"")
    second.write_bytes(b"")
    (tmp_path / "timestamps.csv").write_text(
        " filename , timestamp_s \n"
        "b.pcd,10.0\n"
        "a.pcd,20.0\n",
        encoding="utf-8",
    )

    assert _timestamp_sidecar_time_for_file(first) == pytest.approx(20.0)
    assert _timestamp_sidecar_time_for_file(second) == pytest.approx(10.0)
