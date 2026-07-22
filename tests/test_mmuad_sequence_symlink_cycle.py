from __future__ import annotations

from pathlib import Path

import pytest

from raft_uav.mmuad.sequence import discover_sequence_paths


def _symlink_directory_or_skip(link: Path, target: Path) -> None:
    try:
        link.symlink_to(target, target_is_directory=True)
    except (NotImplementedError, OSError) as exc:
        pytest.skip(f"directory symlinks are unavailable: {exc}")


def test_discover_sequence_paths_breaks_directory_symlink_cycle(tmp_path: Path) -> None:
    sequence = tmp_path / "sequence_001"
    sequence.mkdir()
    (sequence / "candidates.csv").write_text(
        "time_s,x_m,y_m,z_m\n0.0,1.0,2.0,3.0\n",
        encoding="utf-8",
    )
    _symlink_directory_or_skip(sequence / "back_to_root", tmp_path)

    discovered = discover_sequence_paths(tmp_path)

    assert [paths.root for paths in discovered] == [sequence]


def test_discover_sequence_paths_preserves_distinct_symlink_aliases(tmp_path: Path) -> None:
    root = tmp_path / "export"
    root.mkdir()
    target = tmp_path / "actual_sequence"
    target.mkdir()
    (target / "candidates.csv").write_text(
        "time_s,x_m,y_m,z_m\n0.0,1.0,2.0,3.0\n",
        encoding="utf-8",
    )
    _symlink_directory_or_skip(root / "a_alias", target)
    selected_alias = root / "b_alias"
    _symlink_directory_or_skip(selected_alias, target)

    discovered = discover_sequence_paths(root, sequence_glob="b_alias")

    assert [paths.root for paths in discovered] == [selected_alias]
