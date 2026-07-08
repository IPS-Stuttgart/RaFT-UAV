from __future__ import annotations

import importlib
import tomllib
from pathlib import Path


SCRIPT_NAME = "raft-uav-mmuad-class-prob-context"
SCRIPT_TARGET = "raft_uav.mmuad.class_probability_context_cli:main"


def test_mmuad_class_probability_context_entrypoint_is_exposed() -> None:
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))

    assert pyproject["project"]["scripts"][SCRIPT_NAME] == SCRIPT_TARGET


def test_mmuad_class_probability_context_entrypoint_target_imports() -> None:
    module_name, function_name = SCRIPT_TARGET.split(":", 1)

    module = importlib.import_module(module_name)

    assert callable(getattr(module, function_name))
