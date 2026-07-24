from pathlib import Path

import numpy as np

from raft_uav.mmuad.io import infer_time_s_from_filename, load_point_cloud_file_as_points


def test_hyphenated_frame_number_is_a_positive_timestamp() -> None:
    assert infer_time_s_from_filename(Path("frame-001.25.pcd")) == 1.25
    assert infer_time_s_from_filename(Path("livox-42.bin")) == 42.0


def test_explicit_negative_and_scientific_timestamps_are_preserved() -> None:
    assert infer_time_s_from_filename(Path("-1.25.pcd")) == -1.25
    assert infer_time_s_from_filename(Path("frame-1e-3.pcd")) == 1.0e-3


def test_point_cloud_loader_uses_separator_safe_filename_time(tmp_path: Path) -> None:
    path = tmp_path / "frame-001.25.npy"
    np.save(path, np.array([[1.0, 2.0, 3.0]], dtype=float))

    points = load_point_cloud_file_as_points(path)

    assert points["time_s"].tolist() == [1.25]
