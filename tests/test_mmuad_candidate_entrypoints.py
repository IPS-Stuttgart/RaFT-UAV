from __future__ import annotations

import importlib
from pathlib import Path
import tomllib


EXPECTED_CANDIDATE_ENTRYPOINTS = {
    "raft-uav-mmuad-candidate-reservoir": "raft_uav.mmuad.candidate_reservoir:main",
    "raft-uav-mmuad-candidate-reservoir-grid": "raft_uav.mmuad.candidate_reservoir_grid:main",
    "raft-uav-mmuad-candidate-mixture-map": "raft_uav.mmuad.candidate_mixture_map_text_cli:main",
    "raft-uav-mmuad-reservoir-mixture-map": "raft_uav.mmuad.candidate_reservoir_mixture_map:main",
    "raft-uav-mmuad-candidate-assignment-diagnostics": (
        "raft_uav.mmuad.candidate_assignment_diagnostics:main"
    ),
    "raft-uav-mmuad-candidate-assignment-blocks": (
        "raft_uav.mmuad.candidate_assignment_blocks:main"
    ),
    "raft-uav-mmuad-candidate-assignment-action-plan": (
        "raft_uav.mmuad.candidate_assignment_action_plan:main"
    ),
    "raft-uav-mmuad-candidate-assignment-branch-summary": (
        "raft_uav.mmuad.candidate_assignment_branch_summary:main"
    ),
}


def test_candidate_diagnostic_entrypoints_are_registered() -> None:
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    scripts = pyproject["project"]["scripts"]

    for name, target in EXPECTED_CANDIDATE_ENTRYPOINTS.items():
        assert scripts[name] == target


def test_candidate_diagnostic_entrypoints_resolve_to_callables() -> None:
    for target in EXPECTED_CANDIDATE_ENTRYPOINTS.values():
        module_name, function_name = target.split(":", 1)
        module = importlib.import_module(module_name)
        assert callable(getattr(module, function_name))
