from __future__ import annotations

import numpy as np

from raft_uav.mmuad import io as mmuad_io


def test_package_init_keeps_single_point_numpy_reader(tmp_path) -> None:
    path = tmp_path / "cloud.npy"
    np.save(path, np.array([1.0, 2.0, 3.0]))

    frame = mmuad_io._read_numpy_point_cloud(path)

    np.testing.assert_allclose(
        frame[["x_m", "y_m", "z_m"]].to_numpy(dtype=float),
        np.array([[1.0, 2.0, 3.0]]),
    )


def test_package_init_keeps_xyz_only_numpy_trajectory_reader(tmp_path) -> None:
    path = tmp_path / "trajectory_12.5.npy"
    np.save(path, np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))

    frame = mmuad_io._read_numpy_trajectory_table(path)

    np.testing.assert_allclose(frame["time_s"].to_numpy(dtype=float), [12.5, 12.5])
    np.testing.assert_allclose(
        frame[["x_m", "y_m", "z_m"]].to_numpy(dtype=float),
        np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
    )


def test_fourth_point_cloud_column_remains_intensity_when_nonconstant(tmp_path) -> None:
    path = tmp_path / "cloud.npy"
    np.save(
        path,
        np.array(
            [
                [1.0, 2.0, 3.0, 0.25],
                [4.0, 5.0, 6.0, 0.75],
            ]
        ),
    )

    frame = mmuad_io._read_numpy_point_cloud(path)

    assert "intensity" in frame.columns
    np.testing.assert_allclose(frame["intensity"].to_numpy(dtype=float), [0.25, 0.75])
