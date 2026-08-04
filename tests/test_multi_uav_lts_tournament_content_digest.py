from __future__ import annotations

import hashlib
from pathlib import Path

from raft_uav.multi_uav_lts.tournament import (
    _CONTENT_DIGEST_SCHEMA,
    _content_digest,
)


def _legacy_content_digest(path: Path) -> str:
    digest = hashlib.sha256()
    files = tuple(child for child in sorted(path.rglob("*")) if child.is_file())
    for file_path in files:
        relative = file_path.relative_to(path).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(file_path.read_bytes())
    return digest.hexdigest()


def test_content_digest_frames_each_file_payload(tmp_path: Path) -> None:
    left = tmp_path / "left"
    right = tmp_path / "right"
    left.mkdir()
    right.mkdir()

    payload = b"prediction-payload"
    forged_next_header = len(b"b").to_bytes(8, "big") + b"b"
    (left / "a").write_bytes(forged_next_header + payload)
    (right / "a").write_bytes(b"")
    (right / "b").write_bytes(payload)

    assert _legacy_content_digest(left) == _legacy_content_digest(right)

    left_digest, left_bytes = _content_digest(left)
    right_digest, right_bytes = _content_digest(right)

    assert left_digest != right_digest
    assert left_bytes == len(forged_next_header) + len(payload)
    assert right_bytes == len(payload)
    assert _CONTENT_DIGEST_SCHEMA.endswith("-v2")
