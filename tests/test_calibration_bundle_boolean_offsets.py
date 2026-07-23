from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from raft_uav.calibration.bundle import (
    CalibrationBundle,
    load_calibration_bundle,
    write_calibration_bundle_manifest,
)


@pytest.mark.parametrize(
    ("offset_payload", "field_name"),
    [
        ({"time_offsets": {"rf": True}}, "rf_time_offset_s"),
        ({"time_offsets": {"radar": False}}, "radar_time_offset_s"),
        ({"time_offsets": {"rf_time_offset_s": True}}, "rf_time_offset_s"),
        ({"time_offsets": {"radar_time_offset_s": False}}, "radar_time_offset_s"),
        ({"rf_time_offset_correction_s": True}, "rf_time_offset_s"),
        ({"radar_time_offset_correction_s": False}, "radar_time_offset_s"),
    ],
)
def test_calibration_bundle_load_rejects_boolean_offsets(
    tmp_path: Path,
    offset_payload: dict[str, object],
    field_name: str,
) -> None:
    manifest_path = tmp_path / "bundle.json"
    manifest_path.write_text(
        json.dumps({"schema_version": 1, **offset_payload}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=field_name):
        load_calibration_bundle(manifest_path)


@pytest.mark.parametrize(
    ("field_name", "offset_value"),
    [
        ("rf_time_offset_s", True),
        ("radar_time_offset_s", np.bool_(False)),
    ],
)
def test_calibration_bundle_writer_rejects_boolean_offsets_without_creating_file(
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


@pytest.mark.parametrize(
    ("field_name", "offset_value"),
    [
        ("rf_time_offset_s", True),
        ("radar_time_offset_s", np.bool_(False)),
    ],
)
def test_calibration_bundle_constructor_rejects_boolean_offsets(
    tmp_path: Path,
    field_name: str,
    offset_value: object,
) -> None:
    with pytest.raises(ValueError, match=field_name):
        CalibrationBundle(
            path=tmp_path / "bundle.json",
            **{field_name: offset_value},
        )
