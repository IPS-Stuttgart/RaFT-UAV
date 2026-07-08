from __future__ import annotations

import tomllib
from pathlib import Path


def _project_scripts() -> dict[str, str]:
    return tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))["project"]["scripts"]


def test_stratified_mixture_submission_has_console_script() -> None:
    scripts = _project_scripts()

    assert (
        scripts["raft-uav-mmuad-stratified-mixture-submission"]
        == "raft_uav.mmuad.stratified_mixture_submission:main"
    )


def test_track5_acceleration_limit_has_console_script() -> None:
    scripts = _project_scripts()

    assert (
        scripts["raft-uav-mmuad-track5-acceleration-limit"]
        == "raft_uav.mmuad.track5_acceleration_limit:main"
    )
