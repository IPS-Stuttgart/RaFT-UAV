"""Shared guards for the versioned external BoT-SORT source transforms."""

from __future__ import annotations

from typing import Final


_PATCH_MARKER: Final = "RAFT-UAV LTS PATCH v1"


class UpstreamPatchError(RuntimeError):
    """Raised when the upstream checkout does not match the supported layout."""


def _replace_once(text: str, old: str, new: str, *, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise UpstreamPatchError(
            f"cannot patch {label}: expected exactly one upstream anchor, found {count}"
        )
    return text.replace(old, new, 1)


def _replace_span_around_anchor(
    text: str,
    *,
    start_marker: str,
    anchor_marker: str,
    end_marker: str,
    replacement: str,
    label: str,
) -> str:
    """Replace one guarded span while tolerating whitespace-only line changes."""

    if text.count(anchor_marker) != 1:
        raise UpstreamPatchError(
            f"cannot patch {label}: expected exactly one anchor marker"
        )
    anchor = text.index(anchor_marker)
    start = text.rfind(start_marker, 0, anchor)
    if start < 0:
        raise UpstreamPatchError(f"cannot patch {label}: start marker not found")
    end_start = text.find(end_marker, anchor)
    if end_start < 0:
        raise UpstreamPatchError(f"cannot patch {label}: end marker not found")
    end = end_start + len(end_marker)
    return text[:start] + replacement + text[end:]
