"""Collision-safe identifiers for fixed-population parameter sweeps."""

from __future__ import annotations


def claim_config_id(base_id: str, used_ids: set[str]) -> str:
    """Claim a deterministic identifier that is unique within one sweep."""

    candidate = base_id
    suffix = 2
    while candidate in used_ids:
        candidate = f"{base_id}__{suffix}"
        suffix += 1
    used_ids.add(candidate)
    return candidate
