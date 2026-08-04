"""Validate ambiguity counts used by the conservative radar-update policy."""

from __future__ import annotations

from types import ModuleType
from typing import Any


def apply_radar_update_policy_candidate_count_patch(module: ModuleType) -> None:
    """Ignore impossible candidate counts before radar-update classification."""

    if getattr(module, "_candidate_count_validation_patch_applied", False):
        return

    def _first_valid_candidate_count(
        row: Any,
        *names: str,
    ) -> float | None:
        for name in names:
            value = module._finite_float(module._get(row, name))
            if value is not None and value >= 1.0:
                return value
        return None

    def _effective_candidate_count(
        row: Any,
        entropy: float | None,
    ) -> float | None:
        explicit = _first_valid_candidate_count(
            row,
            "association_effective_candidates",
            "association_soft_path_effective_candidates",
        )
        if explicit is not None:
            return explicit
        if entropy is not None and entropy >= 0.0:
            return module._safe_exp_effective_count(entropy)
        return _first_valid_candidate_count(row, "association_soft_path_count")

    module._effective_candidate_count = _effective_candidate_count
    module._candidate_count_validation_patch_applied = True
