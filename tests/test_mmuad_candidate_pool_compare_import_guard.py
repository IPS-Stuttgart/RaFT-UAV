from __future__ import annotations

from raft_uav.mmuad import candidate_pool_compare
from raft_uav.mmuad import candidate_pool_compare_cli


def test_candidate_pool_compare_package_import_uses_cli_guard() -> None:
    assert candidate_pool_compare.main is candidate_pool_compare_cli.main
