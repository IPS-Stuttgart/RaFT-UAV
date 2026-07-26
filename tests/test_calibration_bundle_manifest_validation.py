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


@pytest.mark.parametrize("payload", [None, [], 1, "bundle"])
def test_load_rejects_non_mapping_payload(tmp_path: Path, payload: object) -> None:
    manifest_path = tmp_path / "bundle.json"
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="payload must be a mapping"):
        load_calibration_bundle(manifest_path)


@pytest.mark.parametrize("schema_version", [True, 1.5, "1.5", [1], None])
def test_load_rejects_non_exact_schema_version(
    tmp_path: Path,
    schema_version: object,
) -> None:
    manifest_path = tmp_path / "bundle.json"
    manifest_path.write_text(
        json.dumps({"schema_version": schema_version}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="schema_version must be an exact integer scalar"):
        load_calibration_bundle(manifest_path)


@pytest.mark.parametrize("time_offsets", [[], 1, "rf=1"])
def test_load_rejects_non_mapping_time_offsets(
    tmp_path: Path,
    time_offsets: object,
) -> None:
    manifest_path = tmp_path / "bundle.json"
    manifest_path.write_text(
        json.dumps({"schema_version": 1, "time_offsets": time_offsets}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="time_offsets must be a mapping"):
        load_calibration_bundle(manifest_path)


@pytest.mark.parametrize(
    ("offset_payload", "field_name"),
    [
        ({"time_offsets": {"rf": "not-a-number"}}, "rf_time_offset_s"),
        ({"time_offsets": {"radar": float("nan")}}, "radar_time_offset_s"),
        ({"rf_time_offset_correction_s": [1.0]}, "rf_time_offset_s"),
        ({"radar_time_offset_correction_s": {"seconds": 1.0}}, "radar_time_offset_s"),
    ],
)
def test_load_rejects_invalid_offsets(
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


def test_load_preserves_numeric_string_offsets(tmp_path: Path) -> None:
    manifest_path = tmp_path / "bundle.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "1",
                "time_offsets": {"rf": "1.25", "radar": "-0.5"},
            }
        ),
        encoding="utf-8",
    )

    bundle = load_calibration_bundle(manifest_path)

    assert bundle.rf_time_offset_s == pytest.approx(1.25)
    assert bundle.radar_time_offset_s == pytest.approx(-0.5)


def test_constructor_normalizes_scalar_offsets(tmp_path: Path) -> None:
    bundle = CalibrationBundle(
        path=tmp_path / "bundle.json",
        rf_time_offset_s=np.asarray(1.25),
        radar_time_offset_s=np.float64(-0.5),
    )

    assert bundle.rf_time_offset_s == pytest.approx(1.25)
    assert bundle.radar_time_offset_s == pytest.approx(-0.5)


@pytest.mark.parametrize("offset_value", [np.nan, np.inf, "bad", np.array([1.0])])
def test_writer_rejects_invalid_offsets_without_creating_file(
    tmp_path: Path,
    offset_value: object,
) -> None:
    manifest_path = tmp_path / "bundle.json"

    with pytest.raises(ValueError, match="rf_time_offset_s"):
        write_calibration_bundle_manifest(
            manifest_path,
            rf_time_offset_s=offset_value,
        )

    assert not manifest_path.exists()
