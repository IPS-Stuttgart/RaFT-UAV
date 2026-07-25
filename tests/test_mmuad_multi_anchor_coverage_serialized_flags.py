from __future__ import annotations

import numpy as np
import pandas as pd

import raft_uav.mmuad.candidate_mixture_group_multi_anchor_coverage as coverage_module
from raft_uav.mmuad.candidate_mixture_group_multi_anchor_coverage import (
    COVERAGE_RESCUED,
    AnchorGroupCoverageConfig,
    _coverage_summary,
)


def test_upstream_selection_normalizes_serialized_anchor_match_flags(monkeypatch) -> None:
    scored = pd.DataFrame(
        {
            "mixture_multi_anchor_left_matched": [
                False,
                True,
                "False",
                " true ",
                "0",
                "1",
                "no",
                "yes",
                np.nan,
            ],
            "unrelated": ["False"] * 9,
        }
    )
    selected = pd.DataFrame({"track_id": ["selected"]})
    anchors = pd.DataFrame({"anchor": ["left"]})
    summary = {"selection": "ok"}

    def fake_selection(*args, **kwargs):
        return scored, selected, anchors, summary

    monkeypatch.setattr(
        coverage_module,
        "_ORIGINAL_MULTI_ANCHOR_SELECTION",
        fake_selection,
    )

    normalized, returned_selected, returned_anchors, returned_summary = (
        coverage_module.select_multi_anchor_posterior_mass_hypothesis_group_topk()
    )

    assert normalized["mixture_multi_anchor_left_matched"].tolist() == [
        False,
        True,
        False,
        True,
        False,
        True,
        False,
        True,
        False,
    ]
    assert normalized["unrelated"].tolist() == ["False"] * 9
    assert returned_selected is selected
    assert returned_anchors is anchors
    assert returned_summary is summary


def test_coverage_summary_parses_serialized_rescue_flags() -> None:
    selected_after = pd.DataFrame(
        {
            "sequence_id": ["seq", "seq", "seq"],
            "time_s": [0.0, 0.0, 0.0],
            "mixture_hypothesis_group": ["base", "false-rescue", "true-rescue"],
            COVERAGE_RESCUED: [False, "False", "True"],
        }
    )

    summary = _coverage_summary(
        {},
        coverage_config=AnchorGroupCoverageConfig(),
        distance_columns=[],
        selected_before=selected_after.iloc[[0]].copy(),
        selected_after=selected_after,
        coverage_frames=pd.DataFrame(),
    )

    coverage = summary["anchor_group_coverage"]
    assert coverage["rescued_candidate_rows"] == 1
    assert coverage["rescued_group_count"] == 1
