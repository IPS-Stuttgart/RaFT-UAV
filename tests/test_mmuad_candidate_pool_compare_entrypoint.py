from __future__ import annotations

import tomllib
from pathlib import Path


def test_candidate_pool_compare_console_script_uses_fixed_cli() -> None:
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))

    assert (
        pyproject["project"]["scripts"]["raft-uav-mmuad-candidate-pool-compare"]
        == "raft_uav.mmuad.candidate_pool_compare_cli:main"
    )
