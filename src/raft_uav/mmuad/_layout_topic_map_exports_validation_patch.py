"""Keep MMUAD layout inspection robust to malformed topic-map exports."""

from __future__ import annotations

from typing import Any

from raft_uav.mmuad import layout as _IMPL

_ORIGINAL_TOPIC_MAP_PAYLOAD_HAS_TRUTH_EXPORT = _IMPL._topic_map_payload_has_truth_export


def _topic_map_payload_has_truth_export(payload: dict[str, Any]) -> bool:
    """Return false when a topic map's exports field is not a list."""

    exports = payload.get("exports", [])
    if not isinstance(exports, list):
        return False
    return _ORIGINAL_TOPIC_MAP_PAYLOAD_HAS_TRUTH_EXPORT(payload)


def install() -> None:
    """Install non-list export validation on the layout helper."""

    _IMPL._topic_map_payload_has_truth_export = _topic_map_payload_has_truth_export
