from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from raft_uav.calibration.bundle import (
    CalibrationBundle,
    write_calibration_bundle_manifest,
)


def _object_scalar(value: object) -> np.ndarray:
    boxed = np.empty((), dtype=object)
    boxed[()] = value
    return boxed


@pytest.mark.parametrize(
    "offset_value",
    [
        _object_scalar(np.array(True)),
        _object_scalar(np.array([0.25])),
    ],
)
@pytest.mark.parametrize("field_name", ["rf_time_offset_s", "radar_time_offset_s"])
def test_calibration_bundle_constructor_rejects_nested_pseudo_scalar_offsets(
    tmp_path: Path,
    field_name: str,
    offset_value: object,
) -> None:
    with pytest.raises(ValueError, match=field_name):
        CalibrationBundle(
            path=tmp_path / "bundle.json",
            **{field_name: offset_value},
        )


@pytest.mark.parametrize(
    "offset_value",
    [
        _object_scalar(np.array(False)),
        _object_scalar(np.array([0.5])),
    ],
)
@pytest.mark.parametrize("field_name", ["rf_time_offset_s", "radar_time_offset_s"])
def test_calibration_bundle_writer_rejects_nested_pseudo_scalar_offsets(
    tmp_path: Path,
    field_name: str,
    offset_value: object,
) -> None:
    manifest_path = tmp_path / "bundle.json"

    with pytest.raises(ValueError, match=field_name):
        write_calibration_bundle_manifest(
            manifest_path,
            **{field_name: offset_value},
        )

    assert not manifest_path.exists()


def test_calibration_bundle_rejects_cyclic_object_scalar_offset(tmp_path: Path) -> None:
    cyclic = np.empty((), dtype=object)
    cyclic[()] = cyclic

    with pytest.raises(ValueError, match="rf_time_offset_s"):
        CalibrationBundle(
            path=tmp_path / "bundle.json",
            rf_time_offset_s=cyclic,
        )


def test_calibration_bundle_accepts_recursively_boxed_real_offsets(tmp_path: Path) -> None:
    manifest_path = tmp_path / "bundle.json"
    rf_offset = _object_scalar(_object_scalar(np.float64(0.25)))
    radar_offset = _object_scalar(np.array(0.5))

    bundle = CalibrationBundle(
        path=manifest_path,
        rf_time_offset_s=rf_offset,
        radar_time_offset_s=radar_offset,
    )
    write_calibration_bundle_manifest(
        manifest_path,
        rf_time_offset_s=rf_offset,
        radar_time_offset_s=radar_offset,
    )

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert bundle.rf_time_offset_s == pytest.approx(0.25)
    assert bundle.radar_time_offset_s == pytest.approx(0.5)
    assert payload["time_offsets"] == {"rf": 0.25, "radar": 0.5}
