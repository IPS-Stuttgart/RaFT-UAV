from __future__ import annotations

from pathlib import Path

from raft_uav.mmuad import image_evidence


def test_image_file_rows_handles_files_without_numeric_names(tmp_path: Path) -> None:
    image_path = tmp_path / "frame.png"
    image_path.write_bytes(b"not an actual image")

    rows = image_evidence._image_file_rows([image_path])

    assert rows.empty
    assert list(rows.columns) == ["image_path", "image_time_s"]
