import raft_uav.cli as base_cli
import raft_uav.imm_cli as imm_cli
from raft_uav.io.aerpaw import RADAR_SELECTION_MODES


def test_radar_selection_modes_expose_catprob_all():
    assert "catprob-all" in RADAR_SELECTION_MODES


def test_base_cli_accepts_legacy_catprob_all(monkeypatch, tmp_path):
    captured = {}

    def fake_run_baseline(*args):
        captured["legacy_radar_selection"] = args[5]
        return 0

    monkeypatch.setattr(base_cli, "_run_baseline", fake_run_baseline)

    assert (
        base_cli.main(
            [
                "run-baseline",
                str(tmp_path),
                "--flight",
                "flight",
                "--radar-selection",
                "catprob-all",
            ]
        )
        == 0
    )
    assert captured["legacy_radar_selection"] == "catprob-all"


def test_imm_cli_accepts_catprob_all(monkeypatch, tmp_path):
    captured = {}

    def fake_run_experiment(**kwargs):
        captured.update(kwargs)
        return 0

    monkeypatch.setattr(imm_cli, "run_experiment", fake_run_experiment)

    assert (
        imm_cli.main(
            [str(tmp_path), "--flight", "flight", "--radar-selection", "catprob-all"]
        )
        == 0
    )
    assert captured["radar_selection"] == "catprob-all"
