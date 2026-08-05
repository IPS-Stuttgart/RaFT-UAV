from __future__ import annotations

from pyrecest.evaluation import tracking_metrics as upstream
from raft_uav.multi_uav_lts import _clear_identity, _hota


def test_metric_core_is_reexported_from_pyrecest() -> None:
    for name in (
        "HOTA_ALPHAS",
        "HotaCounts",
        "combine_hota",
        "evaluate_hota",
        "finalize_hota",
    ):
        assert getattr(_hota, name) is getattr(upstream, name)
    for name in (
        "ClearCounts",
        "IdentityCounts",
        "combine_clear",
        "combine_identity",
        "evaluate_clear",
        "evaluate_identity",
        "finalize_clear",
        "finalize_identity",
    ):
        assert getattr(_clear_identity, name) is getattr(upstream, name)
