from __future__ import annotations

import json
from pathlib import Path

import pytest

from raft_uav.calibration.bundle import (
    CalibrationBundle,
    load_calibration_bundle,
    write_calibration_bundle_manifest,
)


_INVALID_METADATA = [
    pytest.param(False, id="false"),
    pytest.param(0, id="zero"),
    pytest.param("", id="empty-string"),
    pytest.param([], id="empty-list"),
    pytest.param([["key", "value"]], id="pair-list"),
]


@pytest.mark.parametrize("metadata", _INVALID_METADATA)
def test_load_rejects_non_mapping_metadata(
    tmp_path: Path,
    metadata: object,
) -> None:
    manifest_path = tmp_path / "bundle.json"
    manifest_path.write_text(
        json.dumps({"schema_version": 1, "metadata": metadata}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="metadata must be a mapping or null"):
        load_calibration_bundle(manifest_path)


@pytest.mark.parametrize("metadata", _INVALID_METADATA)
def test_constructor_rejects_non_mapping_metadata(
    tmp_path: Path,
    metadata: object,
) -> None:
    with pytest.raises(ValueError, match="metadata must be a mapping or null"):
        CalibrationBundle(
            path=tmp_path / "bundle.json",
            metadata=metadata,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("metadata", _INVALID_METADATA)
def test_writer_rejects_non_mapping_metadata_without_creating_file(
    tmp_path: Path,
    metadata: object,
) -> None:
    manifest_path = tmp_path / "bundle.json"

    with pytest.raises(ValueError, match="metadata must be a mapping or null"):
        write_calibration_bundle_manifest(
            manifest_path,
            metadata=metadata,  # type: ignore[arg-type]
        )

    assert not manifest_path.exists()


@pytest.mark.parametrize(
    ("include_metadata", "metadata", "expected"),
    [
        pytest.param(False, None, {}, id="missing"),
        pytest.param(True, None, {}, id="null"),
        pytest.param(True, {}, {}, id="empty-mapping"),
        pytest.param(
            True,
            {"fold": "Opt1"},
            {"fold": "Opt1"},
            id="populated-mapping",
        ),
    ],
)
def test_load_accepts_missing_null_and_mapping_metadata(
    tmp_path: Path,
    include_metadata: bool,
    metadata: object,
    expected: dict[str, object],
) -> None:
    payload: dict[str, object] = {"schema_version": 1}
    if include_metadata:
        payload["metadata"] = metadata
    manifest_path = tmp_path / "bundle.json"
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    bundle = load_calibration_bundle(manifest_path)

    assert bundle.metadata == expected
