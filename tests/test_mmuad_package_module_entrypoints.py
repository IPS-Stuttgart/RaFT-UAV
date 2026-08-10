from __future__ import annotations

import runpy
import sys

import pytest


@pytest.mark.parametrize(
    "module_name",
    [
        "raft_uav.mmuad.candidate_forward_backward",
        "raft_uav.mmuad.candidate_pair_forward_backward",
        "raft_uav.mmuad.candidate_mixture_map_multistart",
        "raft_uav.mmuad.candidate_mixture_map_stratified",
        "raft_uav.mmuad.track5_jerk_limit",
        "raft_uav.mmuad.candidate_mixture_map_grouped",
        "raft_uav.mmuad.track5_segment_source_gate",
        "raft_uav.mmuad.track5_trajectory_smooth",
        "raft_uav.mmuad.track5_class_conditioned_ensemble",
    ],
)
def test_compatibility_packages_support_python_m(
    module_name: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "argv", [module_name, "--help"])

    with pytest.raises(SystemExit) as exc_info:
        runpy.run_module(module_name, run_name="__main__")

    assert exc_info.value.code == 0
