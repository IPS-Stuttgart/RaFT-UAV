from __future__ import annotations

import importlib
import tomllib
from pathlib import Path


def _project_scripts() -> dict[str, str]:
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    return dict(pyproject["project"]["scripts"])


def test_candidate_pool_compare_entrypoint_is_exposed() -> None:
    assert (
        _project_scripts()["raft-uav-mmuad-candidate-pool-compare"]
        == "raft_uav.mmuad.candidate_pool_compare_cli:main"
    )


def test_candidate_pool_compare_entrypoint_target_imports() -> None:
    module_name, function_name = _project_scripts()[
        "raft-uav-mmuad-candidate-pool-compare"
    ].split(":", 1)

    module = importlib.import_module(module_name)

    assert callable(getattr(module, function_name))
