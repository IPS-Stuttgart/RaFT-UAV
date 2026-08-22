from __future__ import annotations

from types import SimpleNamespace

import pytest

from raft_uav.multi_uav_lts import tiled_proposal_baseline as tiled


def test_wrapper_adds_tile_controls_to_upstream_command(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_installer(
        runner,
        *,
        proposal_output_dir,
        proposal_confidence,
        proposal_iou,
        suppress_visualizations,
    ) -> None:
        del proposal_output_dir, proposal_confidence, proposal_iou, suppress_visualizations
        runner._inference_command = lambda *args, **kwargs: ["python", "inference.py"]

    def fake_main(argv) -> int:
        captured["argv"] = tuple(argv)
        runner = SimpleNamespace(_inference_command=lambda: [])
        tiled.baseline._install_proposal_command_wrapper(
            runner,
            proposal_output_dir=tiled.Path("proposals"),
            proposal_confidence=0.001,
            proposal_iou=0.95,
            suppress_visualizations=True,
        )
        captured["command"] = tuple(runner._inference_command())
        return 0

    monkeypatch.setattr(tiled.baseline, "_install_proposal_command_wrapper", fake_installer)
    monkeypatch.setattr(tiled.baseline, "main", fake_main)

    result = tiled.main(
        [
            "--package-only",
            "--proposal-tile-size",
            "800",
            "--proposal-tile-overlap",
            "0.2",
            "--proposal-tile-max-per-frame",
            "123",
        ]
    )

    assert result == 0
    command = captured["command"]
    assert "--proposal-tile-size" in command
    assert command[command.index("--proposal-tile-size") + 1] == "800"
    assert command[command.index("--proposal-tile-overlap") + 1] == "0.2"
    assert command[command.index("--proposal-tile-max-per-frame") + 1] == "123"
    assert "--proposal-tile-size" not in captured["argv"]


@pytest.mark.parametrize("value", ["0", "-1", "1.5", "nan"])
def test_tile_size_validation_rejects_invalid_values(value: str) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        tiled._positive_int(value, name="tile")


@pytest.mark.parametrize("value", ["-0.1", "1", "inf", "nan"])
def test_overlap_validation_rejects_invalid_values(value: str) -> None:
    with pytest.raises(ValueError, match=r"\[0, 1\)"):
        tiled._fraction(value, name="overlap")
