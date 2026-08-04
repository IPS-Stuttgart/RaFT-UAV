from __future__ import annotations

import hashlib
from pathlib import Path

from raft_uav.multi_uav_lts.tournament import (
    _CONTENT_DIGEST_SCHEMA,
    _content_digest,
)


def _legacy_content_digest(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    total_bytes = 0
    files = tuple(child for child in sorted(path.rglob("*")) if child.is_file())
    for file_path in files:
        relative = file_path.relative_to(path).as_posix().encode("utf-8")
        payload = file_path.read_bytes()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(payload)
        total_bytes += len(payload)
    return digest.hexdigest(), total_bytes


def test_content_digest_frames_each_file_payload(tmp_path: Path) -> None:
    left = tmp_path / "left"
    right = tmp_path / "right"
    left.mkdir()
    right.mkdir()

    payload = b"prediction-payload"
    header_b = len(b"b").to_bytes(8, "big") + b"b"
    header_c = len(b"c").to_bytes(8, "big") + b"c"
    (left / "a").write_bytes(b"")
    (left / "c").write_bytes(header_b + payload)
    (right / "a").write_bytes(header_c)
    (right / "b").write_bytes(payload)

    left_legacy = _legacy_content_digest(left)
    right_legacy = _legacy_content_digest(right)
    assert left_legacy == right_legacy

    left_digest, left_bytes = _content_digest(left)
    right_digest, right_bytes = _content_digest(right)

    assert left_digest != right_digest
    assert left_bytes == right_bytes == len(header_b) + len(payload)
    assert _CONTENT_DIGEST_SCHEMA.endswith("-v2")
