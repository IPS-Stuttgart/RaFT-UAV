from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from raft_uav.mmuad.io import load_point_cloud_file_as_points


_ENV_NAME = "RAFT_UAV_BINARY_POINT_COLUMNS"


def _write_float32_rows(path: Path, rows: np.ndarray) -> None:
    path.write_bytes(np.asarray(rows, dtype="<f4").tobytes())


def _positions(frame) -> np.ndarray:
    return frame[["x_m", "y_m", "z_m"]].to_numpy(dtype=float)


def test_binary_point_cloud_auto_detects_unambiguous_widths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(_ENV_NAME, raising=False)
    xyz = np.arange(9, dtype=float).reshape(3, 3)
    xyz_path = tmp_path / "three-points.bin"
    _write_float32_rows(xyz_path, xyz)
    xyzi = np.arange(8, dtype=float).reshape(2, 4)
    xyzi_path = tmp_path / "two-points.bin"
    _write_float32_rows(xyzi_path, xyzi)

    np.testing.assert_allclose(_positions(load_point_cloud_file_as_points(xyz_path)), xyz)
    np.testing.assert_allclose(
        _positions(load_point_cloud_file_as_points(xyzi_path)),
        xyzi[:, :3],
    )


def test_binary_point_cloud_rejects_ambiguous_generic_width(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(_ENV_NAME, raising=False)
    points = np.arange(12, dtype=float).reshape(4, 3)
    path = tmp_path / "cloud.bin"
    _write_float32_rows(path, points)

    with pytest.raises(ValueError, match="ambiguous between XYZ and XYZI"):
        load_point_cloud_file_as_points(path)


def test_binary_point_cloud_xyz_filename_hint_preserves_all_points(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(_ENV_NAME, raising=False)
    points = np.arange(12, dtype=float).reshape(4, 3)
    path = tmp_path / "cloud.xyz.bin"
    _write_float32_rows(path, points)

    frame = load_point_cloud_file_as_points(path)

    np.testing.assert_allclose(_positions(frame), points)


def test_binary_point_cloud_xyzi_filename_hint_ignores_intensity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(_ENV_NAME, raising=False)
    points = np.array(
        [
            [1.0, 2.0, 3.0, 0.1],
            [4.0, 5.0, 6.0, 0.2],
            [7.0, 8.0, 9.0, 0.3],
        ]
    )
    path = tmp_path / "cloud.xyzi.bin"
    _write_float32_rows(path, points)

    frame = load_point_cloud_file_as_points(path)

    np.testing.assert_allclose(_positions(frame), points[:, :3])


def test_binary_point_cloud_environment_hint_supports_cli_style_loading(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    points = np.arange(12, dtype=float).reshape(4, 3)
    path = tmp_path / "cloud.bin"
    _write_float32_rows(path, points)
    monkeypatch.setenv(_ENV_NAME, "xyz")

    frame = load_point_cloud_file_as_points(path)

    np.testing.assert_allclose(_positions(frame), points)


def test_binary_point_cloud_rejects_conflicting_width_hints(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    points = np.arange(12, dtype=float).reshape(4, 3)
    path = tmp_path / "cloud.xyz.bin"
    _write_float32_rows(path, points)
    monkeypatch.setenv(_ENV_NAME, "4")

    with pytest.raises(ValueError, match="conflicts with"):
        load_point_cloud_file_as_points(path)
