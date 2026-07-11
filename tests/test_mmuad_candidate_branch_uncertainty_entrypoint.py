from __future__ import annotations

import importlib
import tomllib
from pathlib import Path


SCRIPT_NAME = "raft-uav-mmuad-candidate-branch-uncertainty"
SCRIPT_TARGET = "raft_uav.mmuad.candidate_branch_uncertainty:main"


def _project_scripts() -> dict[str, str]:
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    return dict(pyproject["project"]["scripts"])


def test_candidate_branch_uncertainty_entrypoint_is_exposed() -> None:
    assert _project_scripts()[SCRIPT_NAME] == SCRIPT_TARGET


def test_candidate_branch_uncertainty_entrypoint_target_imports() -> None:
    module_name, function_name = SCRIPT_TARGET.split(":", 1)

    module = importlib.import_module(module_name)

    assert callable(getattr(module, function_name))
