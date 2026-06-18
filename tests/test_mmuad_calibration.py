from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from raft_uav.mmuad.calibration import load_calibration_json


def test_top_level_calibration_mapping_ignores_world_frame_metadata(tmp_path: Path) -> None:
    path = tmp_path / "calibration.json"
    path.write_text(
        json.dumps(
            {
                "world_frame": "leica_world",
                "radar": {
                    "translation_m": [1.0, 2.0, 3.0],
                    "time_offset_s": 0.25,
                },
            }
        ),
        encoding="utf-8",
    )

    calibration = load_calibration_json(path)

    assert calibration.world_frame == "leica_world"
    assert set(calibration.sensors) == {"radar"}
    sensor = calibration.get("radar")
    assert sensor is not None
    assert sensor.time_offset_s == 0.25
    np.testing.assert_allclose(
        sensor.transform_sensor_to_world.translation_m,
        np.array([1.0, 2.0, 3.0]),
    )


def test_calibration_lookup_prefers_most_specific_prefix_match(tmp_path: Path) -> None:
    path = tmp_path / "calibration.json"
    path.write_text(
        json.dumps(
            {
                "world_frame": "leica_world",
                "sensors": {
                    "radar": {"translation_m": [1.0, 0.0, 0.0]},
                    "radar_enhance_pcl": {"translation_m": [10.0, 0.0, 0.0]},
                },
            }
        ),
        encoding="utf-8",
    )

    calibration = load_calibration_json(path)

    sensor = calibration.get("radar_enhance_pcl_clusters")
    assert sensor is not None
    assert sensor.source == "radar_enhance_pcl"
    np.testing.assert_allclose(
        sensor.transform_sensor_to_world.translation_m,
        np.array([10.0, 0.0, 0.0]),
    )
