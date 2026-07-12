from __future__ import annotations

import importlib
import tomllib
from pathlib import Path


_EXPECTED_ENTRYPOINTS = {
    "raft-uav-mmuad-temporal-consensus": (
        "raft_uav.mmuad.candidate_temporal_consensus:main"
    ),
    "raft-uav-mmuad-temporal-consensus-train-cv": (
        "raft_uav.mmuad.candidate_temporal_consensus_train_cv_cli:main"
    ),
    "raft-uav-mmuad-apply-temporal-consensus-config": (
        "raft_uav.mmuad.candidate_temporal_consensus_train_cv:apply_main"
    ),
    "raft-uav-mmuad-temporal-consensus-assigned": (
        "raft_uav.mmuad.candidate_temporal_consensus_assignment:main"
    ),
}


def _project_scripts() -> dict[str, str]:
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    return dict(pyproject["project"]["scripts"])


def test_temporal_consensus_entrypoints_are_exposed() -> None:
    scripts = _project_scripts()

    for script_name, target in _EXPECTED_ENTRYPOINTS.items():
        assert scripts[script_name] == target


def test_temporal_consensus_entrypoint_targets_import() -> None:
    for target in _EXPECTED_ENTRYPOINTS.values():
        module_name, function_name = target.split(":", 1)
        module = importlib.import_module(module_name)
        assert callable(getattr(module, function_name))
