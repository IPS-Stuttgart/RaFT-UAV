"""Install duplicate-index safeguards across radar association paths."""

from __future__ import annotations

from importlib import import_module

from raft_uav.baselines._imm_radar_duplicate_index_patch import (
    apply_imm_radar_duplicate_index_patch,
)
from raft_uav.baselines._learned_radar_duplicate_index_patch import (
    apply_learned_radar_duplicate_index_patch,
)
from raft_uav.baselines._radar_candidate_index_patch import (
    apply_radar_candidate_index_patch,
)


def apply_radar_index_patch_bundle() -> None:
    """Install all duplicate-index safeguards idempotently."""

    apply_radar_candidate_index_patch(
        import_module("raft_uav.baselines.radar_association")
    )
    apply_imm_radar_duplicate_index_patch(
        import_module("raft_uav.baselines.imm_radar_association")
    )
    apply_learned_radar_duplicate_index_patch(
        import_module("raft_uav.baselines.learned_radar_association")
    )
