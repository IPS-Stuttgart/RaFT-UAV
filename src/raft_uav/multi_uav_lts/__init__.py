"""Utilities for the Beyond Strong Baseline Multi-UAV Tracking LTS benchmark."""

from __future__ import annotations


def _install_zero_frame_coverage_guard() -> None:
    try:
        from raft_uav.multi_uav_lts import coverage_audit as _coverage_audit
        from raft_uav.multi_uav_lts.cli import _parse_int_like as _parse_int_like
    except Exception:
        return

    def _count_out_of_range_frame_rows(text: str, *, expected_frame_count: int | None) -> int:
        if expected_frame_count is None:
            return 0
        count = 0
        for raw in text.splitlines():
            line = raw.strip()
            if not line:
                continue
            parts = [part.strip() for part in line.split(",")]
            if len(parts) < 1:
                continue
            try:
                frame_id = _parse_int_like(parts[0])
            except ValueError:
                continue
            if frame_id < 1 or frame_id > expected_frame_count:
                count += 1
        return count

    _coverage_audit._count_out_of_range_frame_rows = _count_out_of_range_frame_rows


_install_zero_frame_coverage_guard()
