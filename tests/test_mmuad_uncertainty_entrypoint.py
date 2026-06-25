from __future__ import annotations

import tomllib
from pathlib import Path


def test_uncertainty_reservoir_entrypoint_is_exposed() -> None:
    scripts = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))["project"][
        "scripts"
    ]

    assert (
        scripts["raft-uav-mmuad-uncertainty-cap-reservoir"]
        == "raft_uav.mmuad.candidate_reservoir_uncertainty:main"
    )


def test_uncertainty_reservoir_entrypoint_target_imports() -> None:
    from raft_uav.mmuad.candidate_reservoir_uncertainty import main

    assert callable(main)
