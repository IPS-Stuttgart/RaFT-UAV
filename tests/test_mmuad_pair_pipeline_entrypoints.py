from __future__ import annotations

import importlib
import tomllib
from pathlib import Path


SCRIPT_TARGETS = {
    "raft-uav-mmuad-pair-forward-backward-prior": (
        "raft_uav.mmuad.candidate_pair_forward_backward:main"
    ),
    "raft-uav-mmuad-risk-pair-multistart": (
        "raft_uav.mmuad.candidate_risk_pair_multistart:main"
    ),
}


def _project_scripts() -> dict[str, str]:
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    return dict(pyproject["project"]["scripts"])


def test_pair_pipeline_entrypoints_are_exposed() -> None:
    scripts = _project_scripts()

    assert {name: scripts[name] for name in SCRIPT_TARGETS} == SCRIPT_TARGETS


def test_pair_pipeline_entrypoint_targets_import() -> None:
    for target in SCRIPT_TARGETS.values():
        module_name, function_name = target.split(":", 1)
        module = importlib.import_module(module_name)

        assert callable(getattr(module, function_name))
