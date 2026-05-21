from __future__ import annotations

import sys
import tomllib
from pathlib import Path


def test_documented_nested_lofo_tuning_entrypoint_is_exposed() -> None:
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    scripts = pyproject["project"]["scripts"]

    assert (
        scripts["raft-uav-nested-lofo-tuning"]
        == "raft_uav.experiments.nested_lofo_tuning:main"
    )


def test_nested_lofo_tuning_entrypoint_target_imports() -> None:
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    module_name, function_name = pyproject["project"]["scripts"][
        "raft-uav-nested-lofo-tuning"
    ].split(":", 1)

    __import__(module_name)
    module = sys.modules[module_name]

    assert callable(getattr(module, function_name))
