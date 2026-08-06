from __future__ import annotations

import numpy as np
import pytest

from raft_uav.diagnostics.paper_parity import build_baseline_paper_parity


def _report(rf_after_nis_rows: object) -> dict[str, object]:
    return build_baseline_paper_parity(
        stage_counts={
            "rf_raw_rows": 206,
            "rf_after_nis_rows": rf_after_nis_rows,
            "radar_raw_target_track_rows": 3106,
            "radar_after_nis_rows": 2403,
            "kf_all_steps_rows": 2655,
            "kf_updated_rows": 2528,
            "kf_coasted_rows": 127,
        },
        rf_rows=0,
        radar_rows=0,
        selected_radar_rows=0,
        posterior_records=0,
        accepted_by_source={},
        rejected_by_source={},
        paper_position_error_3d={},
    )


@pytest.mark.parametrize(
    "value",
    [
        124.6,
        True,
        np.bool_(False),
        np.asarray([125]),
        125 + 0j,
        np.ma.masked,
    ],
)
def test_paper_parity_does_not_round_malformed_stage_counts(value: object) -> None:
    report = _report(value)

    assert report["observed_counts"]["RF after NIS"] is None
    assert report["count_checks"]["RF after NIS"]["matches_reference"] is None
    assert "RF after NIS" in report["missing_reference_counts"]
    assert report["all_count_matches_reference"] is False


@pytest.mark.parametrize(
    "value",
    [
        125,
        125.0,
        "125",
        np.int64(125),
        np.asarray(125),
    ],
)
def test_paper_parity_preserves_exact_integer_like_stage_counts(value: object) -> None:
    report = _report(value)

    assert report["observed_counts"]["RF after NIS"] == 125
    assert report["count_checks"]["RF after NIS"]["matches_reference"] is True
    assert report["missing_reference_counts"] == []
    assert report["all_count_matches_reference"] is True
