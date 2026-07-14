from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from raft_uav.mmuad.calibration import RigidTransform
import raft_uav.mmuad.camera as camera


def _model(source: str, translation_x: float) -> camera.CameraModel:
    return camera.CameraModel(
        source=source,
        intrinsics=camera.CameraIntrinsics(fx=1.0, fy=1.0, cx=0.0, cy=0.0),
        transform_camera_to_world=RigidTransform(
            rotation=np.eye(3),
            translation_m=np.array([translation_x, 0.0, 0.0]),
        ),
    )


def test_camera_conversion_prefers_longest_source_prefix() -> None:
    generic = _model("camera", 100.0)
    front = _model("camera_front", 10.0)
    candidates = camera.camera_detection_frame_to_candidates(
        pd.DataFrame(
            {
                "sequence_id": ["seq"],
                "time_s": [0.0],
                "source": ["camera_front_detections"],
                "u_px": [0.0],
                "v_px": [0.0],
                "depth_m": [1.0],
            }
        ),
        camera_models={"camera": generic, "camera_front": front},
    )

    assert float(candidates.rows.loc[0, "x_m"]) == pytest.approx(10.0)


def test_camera_model_lookup_does_not_reverse_prefix_match() -> None:
    front = _model("camera_front", 10.0)
    rear = _model("camera_rear", 20.0)

    assert camera._model_for_source(
        {"camera_front": front, "camera_rear": rear},
        "camera",
    ) is None


def test_camera_model_lookup_preserves_single_model_fallback() -> None:
    front = _model("camera_front", 10.0)

    assert camera._model_for_source({"camera_front": front}, "camera") is front
