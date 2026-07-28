from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from raft_uav.mmuad.calibration import RigidTransform
import raft_uav.mmuad.camera as camera


def _model() -> camera.CameraModel:
    return camera.CameraModel(
        source="camera",
        intrinsics=camera.CameraIntrinsics(fx=2.0, fy=4.0, cx=1.0, cy=2.0),
        transform_camera_to_world=RigidTransform(
            rotation=np.eye(3),
            translation_m=np.zeros(3),
        ),
    )


def test_camera_detection_columns_reject_case_insensitive_collision() -> None:
    frame = pd.DataFrame(
        [[0.0, 3.0, 99.0, 6.0, 2.0]],
        columns=["time_s", "U_PX", "u_Px", "v_px", "depth_m"],
    )

    with pytest.raises(ValueError, match="ambiguous columns"):
        camera.camera_detection_frame_to_candidates(
            frame,
            camera_models={"camera": _model()},
        )


def test_camera_detection_columns_accept_unique_padded_headers() -> None:
    frame = pd.DataFrame(
        {
            " time_s ": [0.0],
            " U_PX ": [3.0],
            " v_px ": [6.0],
            " depth_m ": [2.0],
        }
    )

    candidates = camera.camera_detection_frame_to_candidates(
        frame,
        camera_models={"camera": _model()},
    )

    assert candidates.rows.loc[0, "x_m"] == pytest.approx(2.0)
    assert candidates.rows.loc[0, "y_m"] == pytest.approx(2.0)
    assert candidates.rows.loc[0, "z_m"] == pytest.approx(2.0)
