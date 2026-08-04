from __future__ import annotations

from pathlib import Path

import pytest

from raft_uav.multi_uav_lts.tournament import _copy_selected_predictions


@pytest.mark.parametrize("nested", [False, True])
def test_tournament_copy_guard_preserves_directory_source(
    tmp_path: Path,
    nested: bool,
) -> None:
    output_dir = tmp_path / "out"
    target = output_dir / "selected_predictions"
    source = target / "candidate" if nested else target
    source.mkdir(parents=True)
    payload = source / "AA_00.txt"
    payload.write_text("prediction\n", encoding="utf-8")

    with pytest.raises(ValueError, match="copy target"):
        _copy_selected_predictions(source, output_dir)

    assert payload.read_text(encoding="utf-8") == "prediction\n"


def test_tournament_copy_guard_preserves_file_source(tmp_path: Path) -> None:
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    source = output_dir / "selected_predictions.zip"
    source.write_bytes(b"submission")

    with pytest.raises(ValueError, match="copy target"):
        _copy_selected_predictions(source, output_dir)

    assert source.read_bytes() == b"submission"
