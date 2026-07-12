from __future__ import annotations

import importlib
import tomllib
from pathlib import Path


SCRIPT_NAME = "raft-uav-mmuad-posterior-mass-group-topk"
SCRIPT_TARGET = "raft_uav.mmuad.candidate_mixture_group_mass_topk:main"


def _project_scripts() -> dict[str, str]:
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    return dict(pyproject["project"]["scripts"])


def test_posterior_mass_group_topk_entrypoint_is_exposed() -> None:
    assert _project_scripts()[SCRIPT_NAME] == SCRIPT_TARGET


def test_posterior_mass_group_topk_entrypoint_target_imports() -> None:
    module_name, function_name = SCRIPT_TARGET.split(":", 1)

    module = importlib.import_module(module_name)

    assert callable(getattr(module, function_name))
