from __future__ import annotations

import importlib
from pathlib import Path
import tomllib


def test_estimate_sequence_gate_fit_entrypoint_target_imports() -> None:
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    target = pyproject["project"]["scripts"][
        "raft-uav-mmuad-track5-estimate-sequence-gate-fit"
    ]

    assert target == "raft_uav.mmuad.track5_estimate_sequence_gate_fit_text_cli:main"
    module_name, function_name = target.split(":", 1)
    module = importlib.import_module(module_name)

    assert callable(getattr(module, function_name))
