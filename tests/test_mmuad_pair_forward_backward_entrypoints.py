from __future__ import annotations

from importlib import import_module
from pathlib import Path
import tomllib


EXPECTED_SCRIPTS = {
    "raft-uav-mmuad-pair-forward-backward-adaptive": (
        "raft_uav.mmuad.candidate_pair_forward_backward_adaptive:main"
    ),
    "raft-uav-mmuad-pair-forward-backward-agreement": (
        "raft_uav.mmuad.candidate_pair_forward_backward_agreement:main"
    ),
    "raft-uav-mmuad-pair-forward-backward-agreement-adaptive": (
        "raft_uav.mmuad.candidate_pair_forward_backward_agreement_adaptive:main"
    ),
}


def test_pair_forward_backward_variants_are_installed_console_scripts() -> None:
    config = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    scripts = config["project"]["scripts"]

    assert {name: scripts.get(name) for name in EXPECTED_SCRIPTS} == EXPECTED_SCRIPTS

    for target in EXPECTED_SCRIPTS.values():
        module_name, attribute = target.split(":", maxsplit=1)
        assert callable(getattr(import_module(module_name), attribute))
