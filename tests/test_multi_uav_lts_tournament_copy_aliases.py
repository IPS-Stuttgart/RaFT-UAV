from __future__ import annotations

from pathlib import Path

import pytest

from raft_uav.multi_uav_lts.tournament import _copy_selected_predictions


@pytest.mark.parametrize("nested", [False, True], ids=["target", "nested"])
def test_tournament_copy_preserves_directory_candidate_inside_target(
    tmp_path: Path,
    nested: bool,
) -> None:
    output_dir = tmp_path / "output"
    target = output_dir / "selected_predictions"
    source = target / "candidate" if nested else target
    source.mkdir(parents=True)
    marker = source / "prediction.txt"
    marker.write_text("do not delete\n", encoding="utf-8")

    with pytest.raises(ValueError, match="selected candidate path"):
        _copy_selected_predictions(source, output_dir)

    assert marker.read_text(encoding="utf-8") == "do not delete\n"


def test_tournament_copy_preserves_file_candidate_at_target(tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    source = output_dir / "selected_predictions.tar.gz"
    source.write_bytes(b"archive payload")

    with pytest.raises(ValueError, match="selected candidate path"):
        _copy_selected_predictions(source, output_dir)

    assert source.read_bytes() == b"archive payload"
