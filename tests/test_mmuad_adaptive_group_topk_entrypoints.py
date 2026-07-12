from __future__ import annotations

import importlib
import tomllib
from pathlib import Path

import pytest


ENTRYPOINTS = {
    "raft-uav-mmuad-posterior-mass-group-topk": (
        "raft_uav.mmuad.candidate_mixture_group_mass_topk:main"
    ),
    "raft-uav-mmuad-spatial-posterior-mass-group-topk": (
        "raft_uav.mmuad.candidate_mixture_group_spatial_mass_topk:main"
    ),
    "raft-uav-mmuad-anchor-posterior-mass-group-topk": (
        "raft_uav.mmuad.candidate_mixture_group_anchor_mass_topk:main"
    ),
}


def _project_scripts() -> dict[str, str]:
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    return dict(pyproject["project"]["scripts"])


@pytest.mark.parametrize(("script_name", "script_target"), ENTRYPOINTS.items())
def test_adaptive_group_topk_entrypoint_is_exposed(
    script_name: str,
    script_target: str,
) -> None:
    assert _project_scripts()[script_name] == script_target


@pytest.mark.parametrize("script_target", ENTRYPOINTS.values())
def test_adaptive_group_topk_entrypoint_target_imports(script_target: str) -> None:
    module_name, function_name = script_target.split(":", 1)

    module = importlib.import_module(module_name)

    assert callable(getattr(module, function_name))
