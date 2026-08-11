from __future__ import annotations

import pytest

from raft_uav.diagnostics.paper_manifest import validate_paper_manifest


def _file_manifest() -> dict[str, dict[str, str]]:
    return {
        "rf": {"path": "rf.csv", "variant": "rerun", "sha256": "rf-hash"},
        "radar": {"path": "radar.json", "variant": "rerun", "sha256": "radar-hash"},
        "truth": {"path": "truth.txt", "variant": "rerun", "sha256": "truth-hash"},
    }


def _origin() -> dict[str, float]:
    return {
        "latitude_deg": 35.0,
        "longitude_deg": -78.0,
        "altitude_m": 100.0,
    }


def test_required_file_hashes_fail_closed_when_digest_is_missing() -> None:
    file_manifest = _file_manifest()
    del file_manifest["rf"]["sha256"]

    with pytest.raises(ValueError, match="missing rf sha256 digest"):
        validate_paper_manifest(
            file_manifest=file_manifest,
            enu_origin_mode="lw1",
            origin=_origin(),
            rf_clock_offset_s=0.0,
            radar_clock_offset_s=0.0,
        )


def test_file_hash_requirement_can_be_disabled_explicitly() -> None:
    file_manifest = _file_manifest()
    del file_manifest["rf"]["sha256"]

    report = validate_paper_manifest(
        file_manifest=file_manifest,
        enu_origin_mode="lw1",
        origin=_origin(),
        rf_clock_offset_s=0.0,
        radar_clock_offset_s=0.0,
        require_file_hashes=False,
    )

    assert report["valid"] is True
    assert report["errors"] == []
