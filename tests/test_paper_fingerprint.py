from __future__ import annotations

from raft_uav.diagnostics.paper_fingerprint import (
    _fingerprint_run_dir_name,
    _radar_track_selection_orders_to_run,
)
from raft_uav.diagnostics.paper_strict import (
    PAPER_STRICT_RADAR_TRACK_SELECTION_ORDERS,
    PaperStrictConfig,
)


def test_radar_track_selection_orders_to_run_uses_configured_single_order() -> None:
    config = PaperStrictConfig(radar_track_selection_order="range-then-largest-track")

    assert _radar_track_selection_orders_to_run(
        config,
        enumerate_radar_track_selection_orders=False,
    ) == ["range-then-largest-track"]


def test_radar_track_selection_orders_to_run_can_enumerate_all_orders() -> None:
    config = PaperStrictConfig(radar_track_selection_order="range-then-largest-track")

    assert _radar_track_selection_orders_to_run(
        config,
        enumerate_radar_track_selection_orders=True,
    ) == list(PAPER_STRICT_RADAR_TRACK_SELECTION_ORDERS)


def test_fingerprint_run_dir_name_only_includes_enumerated_dimensions() -> None:
    assert _fingerprint_run_dir_name(
        "Opt1",
        variant="auto",
        radar_track_selection_order="raw-track-then-range",
        include_variant=False,
        include_track_order=False,
    ) == "Opt1"
    assert _fingerprint_run_dir_name(
        "Opt1",
        variant="original",
        radar_track_selection_order="range-then-largest-track",
        include_variant=True,
        include_track_order=True,
    ) == "Opt1_original_range-then-largest-track"
