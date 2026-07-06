from pathlib import Path

from raft_uav.mmuad.inspect import _infer_time_s_from_filename, classify_mmuad_file


def test_hyphenated_frame_numbers_are_not_negative_timestamps() -> None:
    assert _infer_time_s_from_filename(Path("frame-001.25.pcd")) == 1.25
    assert _infer_time_s_from_filename(Path("livox-42.bin")) == 42.0


def test_explicit_leading_negative_timestamp_is_preserved() -> None:
    assert _infer_time_s_from_filename(Path("-1.25.pcd")) == -1.25


def test_classify_mmuad_file_uses_separator_safe_timestamp() -> None:
    category, modality, time_s = classify_mmuad_file(Path("sequence/lidar/frame-042.pcd"))

    assert category == "point_cloud"
    assert modality == "lidar"
    assert time_s == 42.0
