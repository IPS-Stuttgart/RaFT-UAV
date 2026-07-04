from __future__ import annotations

from raft_uav.multi_uav_lts.coverage_audit import _count_out_of_range_frame_rows


def test_lts_coverage_counts_rows_when_expected_frame_count_is_zero() -> None:
    text = "1,1,10,20,5,6,0.9,1,1\n"

    assert _count_out_of_range_frame_rows(text, expected_frame_count=0) == 1
