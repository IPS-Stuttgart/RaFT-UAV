from __future__ import annotations

import tomllib
from pathlib import Path


RESTORED_MMUAD_ENTRYPOINTS = {
    "raft-uav-mmuad-track5-estimate-ensemble-loso": (
        "raft_uav.mmuad.track5_estimate_ensemble_loso_weight_search:main"
    ),
    "raft-uav-mmuad-track5-sequence-gate-fit": (
        "raft_uav.mmuad.track5_sequence_gate_fit_text_cli:main"
    ),
    "raft-uav-mmuad-track5-speed-limit": "raft_uav.mmuad.track5_speed_limit:main",
    "raft-uav-mmuad-train-sequence-classifier": (
        "raft_uav.mmuad.train_sequence_classifier:main"
    ),
    "raft-uav-mmuad-candidate-assignment-report": (
        "raft_uav.mmuad.candidate_assignment_report:main"
    ),
    "raft-uav-mmuad-fit-source-calibration": "raft_uav.mmuad.source_calibration:fit_main",
    "raft-uav-mmuad-select-train-config": "raft_uav.mmuad.train_selected_config:main",
    "raft-uav-multi-uav-lts": "raft_uav.multi_uav_lts.cli:main",
    "raft-uav-multi-uav-lts-coverage-audit": (
        "raft_uav.multi_uav_lts.coverage_audit:main"
    ),
}


def test_mmuad_entrypoint_registry_extends_past_track5_weight_search() -> None:
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    scripts = pyproject["project"]["scripts"]

    for name, target in RESTORED_MMUAD_ENTRYPOINTS.items():
        assert scripts[name] == target
