from __future__ import annotations

import runpy
import sys

import pandas as pd
import pytest

from raft_uav.mmuad import candidate_mixture_map as core
from raft_uav.mmuad import candidate_mixture_map_sequence_multistart as sequence_multistart


def _candidate(
    sequence_id: str,
    time_s: float,
    branch: str,
    x_m: float,
) -> dict[str, object]:
    return {
        "sequence_id": sequence_id,
        "time_s": time_s,
        "source": "radar",
        "track_id": f"{sequence_id}-{branch}",
        "candidate_branch": branch,
        "x_m": x_m,
        "y_m": 0.0,
        "z_m": 0.0,
        "ranker_score": 1.0,
        "predicted_sigma_m": 1.0,
    }


def test_sequence_multistart_applies_branch_threshold_per_sequence() -> None:
    rows = [
        _candidate("long", float(index), "long-branch", float(index))
        for index in range(10)
    ]
    rows.append(_candidate("short", 0.0, "short-branch", 100.0))
    candidates = pd.DataFrame.from_records(rows)
    config = sequence_multistart.multistart.CandidateMixtureMultiStartConfig(
        include_score_top1=False,
        include_frame_median=False,
        include_branch_starts=True,
        max_branch_starts=8,
        min_branch_frame_fraction=0.2,
    )

    starts = sequence_multistart.multistart.build_candidate_mixture_initializations(
        candidates,
        mixture_config=core.CandidateMixtureMapConfig(
            top_k=0,
            score_column="ranker_score",
            sigma_column="predicted_sigma_m",
        ),
        multistart_config=config,
    )

    assert 1 / 11 < config.min_branch_frame_fraction
    assert "branch:long-branch" in starts
    assert "branch:short-branch" in starts
    short_start = starts["branch:short-branch"]
    assert short_start is not None
    short_row = short_start.loc[short_start["sequence_id"] == "short"].iloc[0]
    assert short_row["state_x_m"] == pytest.approx(100.0)


def test_sequence_multistart_package_supports_python_m(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module_name = "raft_uav.mmuad.candidate_mixture_map_sequence_multistart"
    monkeypatch.setattr(sys, "argv", [module_name, "--help"])

    with pytest.raises(SystemExit) as exc_info:
        runpy.run_module(module_name, run_name="__main__")

    assert exc_info.value.code == 0
