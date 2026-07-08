from __future__ import annotations

import importlib
import tomllib
from pathlib import Path


LTS_ENTRYPOINTS = {
    "raft-uav-multi-uav-lts": "raft_uav.multi_uav_lts.cli:main",
    "raft-uav-multi-uav-lts-coverage-audit": "raft_uav.multi_uav_lts.coverage_audit:main",
    "raft-uav-multi-uav-lts-duplicate-audit": "raft_uav.multi_uav_lts.duplicate_audit:main",
}


def _project_scripts() -> dict[str, str]:
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    return dict(pyproject["project"]["scripts"])


def test_multi_uav_lts_entrypoints_are_exposed() -> None:
    scripts = _project_scripts()

    for script_name, target in LTS_ENTRYPOINTS.items():
        assert scripts[script_name] == target


def test_multi_uav_lts_entrypoint_targets_import() -> None:
    for target in LTS_ENTRYPOINTS.values():
        module_name, function_name = target.split(":", 1)
        module = importlib.import_module(module_name)

        assert callable(getattr(module, function_name))
