from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from raft_uav.mmuad.calibration import RigidTransform
import raft_uav.mmuad.camera as camera


def _model(source: str = "camera") -> camera.CameraModel:
    return camera.CameraModel(
        source=source,
        intrinsics=camera.CameraIntrinsics(fx=100.0, fy=100.0, cx=50.0, cy=40.0),
        transform_camera_to_world=RigidTransform(
            rotation=np.eye(3),
            translation_m=np.zeros(3),
        ),
    )


@pytest.mark.parametrize(
    "depth_m",
    [
        0.0,
        -1.0,
        np.nan,
        np.inf,
        True,
        1.0 + 0.0j,
        np.array([1.0]),
    ],
)
def test_backprojection_rejects_invalid_metric_depth(depth_m) -> None:
    with pytest.raises(ValueError, match="camera depth must be a finite real scalar > 0"):
        camera.backproject_pixel_to_camera_xyz(
            50.0,
            40.0,
            depth_m,
            _model().intrinsics,
        )


def test_camera_candidate_conversion_does_not_silently_drop_nan_depth() -> None:
    frame = pd.DataFrame(
        {
            "sequence_id": ["seq"],
            "time_s": [0.0],
            "source": ["camera"],
            "u_px": [50.0],
            "v_px": [40.0],
            "depth_m": [np.nan],
        }
    )

    with pytest.raises(ValueError, match="camera depth must be a finite real scalar > 0"):
        camera.camera_detection_frame_to_candidates(
            frame,
            camera_models={"camera": _model()},
        )


def test_camera_model_loader_rejects_normalized_source_collisions(tmp_path) -> None:
    entry = {"fx": 100.0, "fy": 100.0, "cx": 50.0, "cy": 40.0}
    path = tmp_path / "cameras.json"
    path.write_text(
        json.dumps(
            {
                "cameras": {
                    "Cam0": entry,
                    " cam0 ": {**entry, "cx": 60.0},
                }
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="ambiguous source names"):
        camera.load_camera_models(path)


def test_camera_model_lookup_rejects_normalized_source_collisions() -> None:
    with pytest.raises(ValueError, match="ambiguous source names"):
        camera._model_for_source(
            {"Cam0": _model("Cam0"), " cam0 ": _model(" cam0 ")},
            "cam0",
        )
