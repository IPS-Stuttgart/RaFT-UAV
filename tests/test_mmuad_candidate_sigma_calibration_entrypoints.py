from __future__ import annotations

import importlib
import tomllib
from pathlib import Path

import pytest


EXPECTED_CANDIDATE_SIGMA_CALIBRATION_SCRIPTS = {
    "raft-uav-mmuad-fit-candidate-sigma-calibration": "raft_uav.mmuad.candidate_uncertainty_calibration:fit_main",
    "raft-uav-mmuad-apply-candidate-sigma-calibration": "raft_uav.mmuad.candidate_uncertainty_calibration:apply_main",
    "raft-uav-mmuad-candidate-sigma-calibration-train-cv": "raft_uav.mmuad.candidate_uncertainty_calibration_train_cv:main",
}


def _project_scripts() -> dict[str, str]:
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    return dict(pyproject["project"]["scripts"])


@pytest.mark.parametrize(
    ("script_name", "target"),
    sorted(EXPECTED_CANDIDATE_SIGMA_CALIBRATION_SCRIPTS.items()),
)
def test_candidate_sigma_calibration_entrypoints_are_exposed(
    script_name: str,
    target: str,
) -> None:
    scripts = _project_scripts()

    assert scripts[script_name] == target


@pytest.mark.parametrize("target", sorted(EXPECTED_CANDIDATE_SIGMA_CALIBRATION_SCRIPTS.values()))
def test_candidate_sigma_calibration_entrypoint_targets_import(target: str) -> None:
    module_name, function_name = target.split(":", 1)

    module = importlib.import_module(module_name)

    assert callable(getattr(module, function_name))
