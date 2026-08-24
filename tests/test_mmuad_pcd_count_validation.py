from __future__ import annotations

from pathlib import Path
import struct

import numpy as np
import pytest

from raft_uav.mmuad.io import _parse_pcd_header, load_point_cloud_file_as_points


def _header(*lines: str) -> bytes:
    return ("\n".join(lines) + "\n").encode("ascii")


def test_binary_pcd_rejects_count_length_that_does_not_match_fields(
    tmp_path: Path,
) -> None:
    path = tmp_path / "mismatched-count.pcd"
    header = _header(
        "VERSION .7",
        "FIELDS x y z",
        "SIZE 4 4 4",
        "TYPE F F F",
        "COUNT 1 2",
        "WIDTH 2",
        "HEIGHT 1",
        "POINTS 2",
        "DATA binary",
    )
    # Under the old fallback, COUNT was discarded and these bytes were decoded
    # as two ordinary XYZ rows, silently ignoring the final two float32 values.
    payload = struct.pack("<8f", 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0)
    path.write_bytes(header + payload)

    with pytest.raises(ValueError, match=r"inconsistent FIELDS/COUNT lengths"):
        load_point_cloud_file_as_points(path)


def test_pcd_rejects_nonpositive_explicit_count(tmp_path: Path) -> None:
    path = tmp_path / "zero-count.pcd"
    path.write_bytes(
        _header(
            "VERSION .7",
            "FIELDS x y z",
            "SIZE 4 4 4",
            "TYPE F F F",
            "COUNT 1 0 1",
            "WIDTH 1",
            "HEIGHT 1",
            "POINTS 1",
            "DATA ascii",
            "1 2 3",
        )
    )

    with pytest.raises(ValueError, match=r"COUNT for field 'y' must be a positive integer"):
        load_point_cloud_file_as_points(path)


def test_pcd_omitted_count_keeps_default_scalar_layout(tmp_path: Path) -> None:
    path = tmp_path / "scalar-layout.pcd"
    path.write_bytes(
        _header(
            "VERSION .7",
            "FIELDS x y z",
            "SIZE 4 4 4",
            "TYPE F F F",
            "WIDTH 1",
            "HEIGHT 1",
            "POINTS 1",
            "DATA binary",
        )
        + struct.pack("<3f", 1.25, -2.5, 3.75)
    )

    points = load_point_cloud_file_as_points(path)

    np.testing.assert_allclose(points[["x_m", "y_m", "z_m"]].iloc[0], [1.25, -2.5, 3.75])


def test_ascii_pcd_respects_vector_field_count_offsets(tmp_path: Path) -> None:
    path = tmp_path / "vector-field.pcd"
    path.write_bytes(
        _header(
            "VERSION .7",
            "FIELDS x y descriptor z",
            "SIZE 4 4 4 4",
            "TYPE F F F F",
            "COUNT 1 1 3 1",
            "WIDTH 1",
            "HEIGHT 1",
            "POINTS 1",
            "DATA ascii",
            "1 2 10 11 12 3",
        )
    )

    points = load_point_cloud_file_as_points(path)

    np.testing.assert_allclose(points[["x_m", "y_m", "z_m"]].iloc[0], [1.0, 2.0, 3.0])


def test_pcd_parser_preserves_supported_vector_counts() -> None:
    parsed = _parse_pcd_header(
        "\n".join(
            [
                "FIELDS x descriptor z",
                "SIZE 4 4 4",
                "TYPE F F F",
                "COUNT 1 3 1",
                "WIDTH 1",
                "POINTS 1",
                "DATA binary",
            ]
        )
    )

    assert parsed["count"] == [1, 3, 1]
