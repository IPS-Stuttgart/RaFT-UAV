from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from raft_uav.mmuad.submission import (
    OFFICIAL_UPLOAD_MANIFEST_SCHEMA,
    verify_official_upload_manifest,
)


def _write_manifest(
    tmp_path: Path,
    *,
    codabench_upload_ready: Any,
    leaderboard_ready: Any,
) -> Path:
    artifact = tmp_path / "submission.csv"
    artifact.write_text("payload\n", encoding="utf-8")
    payload = artifact.read_bytes()
    manifest = {
        "schema": OFFICIAL_UPLOAD_MANIFEST_SCHEMA,
        "artifact_path": artifact.name,
        "artifact_size_bytes": len(payload),
        "artifact_sha256": hashlib.sha256(payload).hexdigest(),
        "codabench_upload_ready": codabench_upload_ready,
        "leaderboard_ready": leaderboard_ready,
    }
    path = tmp_path / "mmuad_official_upload_manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def test_upload_manifest_rejects_string_readiness_flags(tmp_path: Path) -> None:
    manifest_path = _write_manifest(
        tmp_path,
        codabench_upload_ready="false",
        leaderboard_ready="false",
    )

    summary = verify_official_upload_manifest(manifest_path)

    assert summary["manifest_codabench_upload_ready"] is False
    assert summary["manifest_leaderboard_ready"] is False
    assert summary["valid"] is False
    assert summary["codabench_upload_ready"] is False
    assert any(
        "codabench_upload_ready" in error and "JSON boolean" in error
        for error in summary["errors"]
    )
    assert any(
        "leaderboard_ready" in error and "JSON boolean" in error
        for error in summary["errors"]
    )


def test_upload_manifest_preserves_valid_boolean_readiness_flags(tmp_path: Path) -> None:
    manifest_path = _write_manifest(
        tmp_path,
        codabench_upload_ready=True,
        leaderboard_ready=True,
    )

    summary = verify_official_upload_manifest(manifest_path)

    assert summary["errors"] == []
    assert summary["valid"] is True
    assert summary["manifest_codabench_upload_ready"] is True
    assert summary["manifest_leaderboard_ready"] is True
    assert summary["codabench_upload_ready"] is True
