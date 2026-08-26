"""Regression tests for affine MMUAD homogeneous calibration matrices."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from raft_uav.mmuad.calibration import (
    calibration_from_mapping,
    load_calibration_auto,
)


def _projective_matrix() -> np.ndarray:
    matrix = np.eye(4)
    matrix[3, 0] = 0.25
    return matrix


def test_mapping_rejects_nonhomogeneous_4x4_transform() -> None:
    payload = {
        "sensors": {
            "radar": {
                "transform_matrix": _projective_matrix().tolist(),
            }
        }
    }

    with pytest.raises(ValueError, match="homogeneous row"):
        calibration_from_mapping(payload)


@pytest.mark.parametrize(
    ("suffix", "delimiter"),
    [
        (".txt", " "),
        (".csv", ","),
    ],
)
def test_single_matrix_file_rejects_nonhomogeneous_last_row(
    tmp_path: Path,
    suffix: str,
    delimiter: str,
) -> None:
    path = tmp_path / f"calibration{suffix}"
    np.savetxt(path, _projective_matrix(), delimiter=delimiter)

    with pytest.raises(ValueError, match="homogeneous row"):
        load_calibration_auto(path)


def test_single_matrix_file_preserves_valid_rigid_transform(tmp_path: Path) -> None:
    matrix = np.eye(4)
    matrix[:3, 3] = [1.5, -2.0, 4.25]
    path = tmp_path / "calibration.txt"
    np.savetxt(path, matrix)

    calibration = load_calibration_auto(path)
    sensor = calibration.get("radar")

    assert sensor is not None
    np.testing.assert_allclose(
        sensor.transform_sensor_to_world.apply(np.zeros(3)),
        matrix[:3, 3],
        rtol=0.0,
        atol=1.0e-12,
    )
