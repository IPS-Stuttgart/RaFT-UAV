from __future__ import annotations

from pathlib import Path

import pytest

from raft_uav.io.aerpaw import find_rf_sensor_and_radar_root


def test_find_rf_radar_root_recurses_into_underscore_alias(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    expected = dataset_root / "archive" / "RF_Sensor_and_Radar"
    expected.mkdir(parents=True)

    assert find_rf_sensor_and_radar_root(dataset_root) == expected


def test_find_rf_radar_root_rejects_matching_regular_file(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()
    (dataset_root / "RF Sensor and Radar").write_text("not a directory", encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="Could not find RF Sensor and Radar folder"):
        find_rf_sensor_and_radar_root(dataset_root)
