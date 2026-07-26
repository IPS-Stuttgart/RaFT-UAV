from __future__ import annotations

import json

import numpy as np
import pytest

import raft_uav.mmuad.camera as camera


@pytest.mark.parametrize(
    "intrinsics",
    [
        camera.CameraIntrinsics(fx=0.0, fy=2.0, cx=0.0, cy=0.0),
        camera.CameraIntrinsics(fx=2.0, fy=-1.0, cx=0.0, cy=0.0),
        camera.CameraIntrinsics(fx=np.inf, fy=2.0, cx=0.0, cy=0.0),
        camera.CameraIntrinsics(fx=2.0, fy=2.0, cx=np.nan, cy=0.0),
        camera.CameraIntrinsics(fx=True, fy=2.0, cx=0.0, cy=0.0),
        camera.CameraIntrinsics(
            fx=np.array([2.0]),
            fy=2.0,
            cx=0.0,
            cy=0.0,
        ),
    ],
)
def test_backprojection_rejects_invalid_camera_intrinsics(intrinsics) -> None:
    with pytest.raises(
        ValueError,
        match="camera intrinsics must contain finite real scalars",
    ):
        camera.backproject_pixel_to_camera_xyz(1.0, 2.0, 3.0, intrinsics)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("fx", 0.0),
        ("fy", -1.0),
    ],
)
def test_camera_model_loader_rejects_invalid_focal_lengths(
    tmp_path,
    field: str,
    value: float,
) -> None:
    payload = {"fx": 100.0, "fy": 100.0, "cx": 50.0, "cy": 40.0}
    payload[field] = value
    path = tmp_path / "camera.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="fx > 0 and fy > 0"):
        camera.load_camera_models(path)


def test_backprojection_accepts_finite_scalar_like_intrinsics() -> None:
    intrinsics = camera.CameraIntrinsics(
        fx=np.array(2.0),
        fy=np.float32(4.0),
        cx=np.int64(1),
        cy=0.0,
    )

    point = camera.backproject_pixel_to_camera_xyz(3.0, 4.0, 8.0, intrinsics)

    np.testing.assert_allclose(point, np.array([8.0, 8.0, 8.0]))
