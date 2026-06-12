from __future__ import annotations

import re
import sys
import tomllib
from pathlib import Path


def _project_scripts() -> dict[str, str]:
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    return dict(pyproject["project"]["scripts"])


def _assert_entrypoint_target_imports(script_name: str) -> None:
    module_name, function_name = _project_scripts()[script_name].split(":", 1)

    __import__(module_name)
    module = sys.modules[module_name]

    assert callable(getattr(module, function_name))


def test_documented_nested_lofo_tuning_entrypoint_is_exposed() -> None:
    scripts = _project_scripts()

    assert (
        scripts["raft-uav-nested-lofo-tuning"]
        == "raft_uav.experiments.nested_lofo_tuning:main"
    )


def test_nested_lofo_tuning_entrypoint_target_imports() -> None:
    _assert_entrypoint_target_imports("raft-uav-nested-lofo-tuning")


def test_mmuad_tracking_entrypoint_is_exposed() -> None:
    assert _project_scripts()["raft-uav-mmuad-track"] == "raft_uav.mmuad.cli:main"


def test_mmuad_tracking_entrypoint_target_imports() -> None:
    _assert_entrypoint_target_imports("raft-uav-mmuad-track")


def test_playbook_runnable_commands_are_installed_entrypoints() -> None:
    playbook = Path("docs/results_improvement_playbook.md").read_text(encoding="utf-8")
    documented = set(re.findall(r"`(raft-uav-[a-z0-9][a-z0-9-]*)`", playbook))

    assert documented <= set(_project_scripts())
