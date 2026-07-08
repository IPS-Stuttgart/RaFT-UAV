from __future__ import annotations

import tomllib
from pathlib import Path


def test_stratified_mixture_submission_has_console_script() -> None:
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    scripts = pyproject["project"]["scripts"]

    assert (
        scripts["raft-uav-mmuad-stratified-mixture-submission"]
        == "raft_uav.mmuad.stratified_mixture_submission:main"
    )


def test_class_probability_context_has_console_script() -> None:
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    scripts = pyproject["project"]["scripts"]

    assert (
        scripts["raft-uav-mmuad-class-prob-context"]
        == "raft_uav.mmuad.class_probability_context_cli:main"
    )
