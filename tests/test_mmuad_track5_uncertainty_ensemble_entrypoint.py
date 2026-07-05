from __future__ import annotations

import tomllib
from pathlib import Path


def test_uncertainty_ensemble_console_script_is_registered() -> None:
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))

    assert (
        pyproject["project"]["scripts"]["raft-uav-mmuad-track5-uncertainty-ensemble"]
        == "raft_uav.mmuad.track5_uncertainty_ensemble:main"
    )
