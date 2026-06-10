from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from raft_uav.calibration.bundle import load_calibration_bundle, write_calibration_bundle_manifest


def test_calibration_bundle_manifest_writes_json_safe_offsets_and_metadata(tmp_path: Path) -> None:
    manifest_path = tmp_path / "bundle.json"

    write_calibration_bundle_manifest(
        manifest_path,
        rf_time_offset_s=float("nan"),
        radar_time_offset_s=np.float64("inf"),
        metadata={
            "fold": np.int64(2),
            "score": np.nan,
            "source_path": tmp_path / "train.csv",
        },
    )

    payload_text = manifest_path.read_text(encoding="utf-8")
    payload = json.loads(payload_text)
    bundle = load_calibration_bundle(manifest_path)

    assert "NaN" not in payload_text
    assert "Infinity" not in payload_text
    assert payload["time_offsets"] == {"rf": None, "radar": None}
    assert payload["metadata"]["score"] is None
    assert payload["metadata"]["fold"] == 2
    assert payload["metadata"]["source_path"] == str(tmp_path / "train.csv")
    assert bundle.rf_time_offset_s == 0.0
    assert bundle.radar_time_offset_s == 0.0


def test_calibration_bundle_load_treats_legacy_non_finite_offsets_as_missing(
    tmp_path: Path,
) -> None:
    manifest_path = tmp_path / "legacy_bundle.json"
    manifest_path.write_text(
        '{"schema_version": 1, "time_offsets": {"rf": NaN, "radar": Infinity}}',
        encoding="utf-8",
    )

    bundle = load_calibration_bundle(manifest_path)

    assert bundle.rf_time_offset_s == 0.0
    assert bundle.radar_time_offset_s == 0.0
