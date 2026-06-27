from __future__ import annotations

import re
import sys
import tomllib
from pathlib import Path


def _project_scripts() -> dict[str, str]:
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    return dict(pyproject["project"]["scripts"])


def _assert_entrypoint_target_imports(script_name: str) -> None:
    module_name, function_name = _project_scripts()[script_name].split(":", 1)

    __import__(module_name)
    module = sys.modules[module_name]

    assert callable(getattr(module, function_name))


def test_documented_nested_lofo_tuning_entrypoint_is_exposed() -> None:
    scripts = _project_scripts()

    assert (
        scripts["raft-uav-nested-lofo-tuning"]
        == "raft_uav.experiments.nested_lofo_tuning:main"
    )


def test_nested_lofo_tuning_entrypoint_target_imports() -> None:
    _assert_entrypoint_target_imports("raft-uav-nested-lofo-tuning")


def test_mmuad_tracking_entrypoint_is_exposed() -> None:
    assert _project_scripts()["raft-uav-mmuad-track"] == "raft_uav.mmuad.cli:main"


def test_mmuad_tracking_entrypoint_target_imports() -> None:
    _assert_entrypoint_target_imports("raft-uav-mmuad-track")


def test_mmuad_run_entrypoint_is_exposed() -> None:
    assert _project_scripts()["raft-uav-mmuad-run"] == "raft_uav.mmuad.run:main"


def test_mmuad_run_entrypoint_target_imports() -> None:
    _assert_entrypoint_target_imports("raft-uav-mmuad-run")


def test_mmuad_run_entrypoint_help_does_not_require_sequence_root(monkeypatch) -> None:
    from raft_uav.mmuad import run

    forwarded: list[str] = []

    def fake_track_main(argv: list[str] | None = None) -> int:
        forwarded.extend(argv or [])
        return 0

    monkeypatch.setattr(run, "track_main", fake_track_main)

    assert run.main(["--help"]) == 0
    assert forwarded == ["--help"]


def test_mmuad_run_entrypoint_preserves_leading_value_options(monkeypatch) -> None:
    from raft_uav.mmuad import run

    forwarded: list[str] = []

    def fake_track_main(argv: list[str] | None = None) -> int:
        forwarded.extend(argv or [])
        return 0

    monkeypatch.setattr(run, "track_main", fake_track_main)

    assert run.main(["--output-dir", "outputs/mmuad", "data/mmuad"]) == 0
    assert forwarded == [
        "--sequence-root",
        "data/mmuad",
        "--output-dir",
        "outputs/mmuad",
    ]


def test_mmuad_run_entrypoint_drops_separator_before_sequence_root(monkeypatch) -> None:
    from raft_uav.mmuad import run

    forwarded: list[str] = []

    def fake_track_main(argv: list[str] | None = None) -> int:
        forwarded.extend(argv or [])
        return 0

    monkeypatch.setattr(run, "track_main", fake_track_main)

    assert run.main(["--output-dir", "outputs/mmuad", "--", "data/mmuad"]) == 0
    assert forwarded == [
        "--sequence-root",
        "data/mmuad",
        "--output-dir",
        "outputs/mmuad",
    ]


def test_mmuad_run_entrypoint_preserves_all_leading_mmuad_value_options(monkeypatch) -> None:
    from raft_uav.mmuad import run

    forwarded: list[str] = []

    def fake_track_main(argv: list[str] | None = None) -> int:
        forwarded.extend(argv or [])
        return 0

    monkeypatch.setattr(run, "track_main", fake_track_main)

    assert (
        run.main(
            [
                "--candidate-file",
                "external_candidates.csv",
                "--sequence-root-archive-extract-dir",
                "outputs/extracted",
                "--ug2-class-map-file",
                "classes.csv",
                "data/mmuad",
                "--output-dir",
                "outputs/mmuad",
            ]
        )
        == 0
    )
    assert forwarded == [
        "--sequence-root",
        "data/mmuad",
        "--candidate-file",
        "external_candidates.csv",
        "--sequence-root-archive-extract-dir",
        "outputs/extracted",
        "--ug2-class-map-file",
        "classes.csv",
        "--output-dir",
        "outputs/mmuad",
    ]


def test_mmuad_run_entrypoint_preserves_dynamic_extraction_value_options(monkeypatch) -> None:
    from raft_uav.mmuad import run

    forwarded: list[str] = []

    def fake_track_main(argv: list[str] | None = None) -> int:
        forwarded.extend(argv or [])
        return 0

    monkeypatch.setattr(run, "track_main", fake_track_main)

    assert (
        run.main(
            [
                "--point-extraction-mode",
                "dynamic",
                "--dynamic-background-voxel-size-m",
                "1.0",
                "--dynamic-background-min-frame-fraction",
                "0.5",
                "--dynamic-background-min-frames",
                "2",
                "--dynamic-background-neighbor-radius-voxels",
                "1",
                "data/mmuad",
                "--output-dir",
                "outputs/mmuad",
            ]
        )
        == 0
    )
    assert forwarded == [
        "--sequence-root",
        "data/mmuad",
        "--point-extraction-mode",
        "dynamic",
        "--dynamic-background-voxel-size-m",
        "1.0",
        "--dynamic-background-min-frame-fraction",
        "0.5",
        "--dynamic-background-min-frames",
        "2",
        "--dynamic-background-neighbor-radius-voxels",
        "1",
        "--output-dir",
        "outputs/mmuad",
    ]


def test_mmuad_track5_scorecard_entrypoint_is_exposed() -> None:
    assert (
        _project_scripts()["raft-uav-mmuad-track5-scorecard"]
        == "raft_uav.mmuad.track5_scorecard_cli:main"
    )


def test_mmuad_track5_scorecard_entrypoint_target_imports() -> None:
    _assert_entrypoint_target_imports("raft-uav-mmuad-track5-scorecard")


def test_mmuad_track5_sequence_gate_fit_entrypoint_is_exposed() -> None:
    assert (
        _project_scripts()["raft-uav-mmuad-track5-sequence-gate-fit"]
        == "raft_uav.mmuad.track5_sequence_gate_fit:main"
    )


def test_mmuad_track5_sequence_gate_fit_entrypoint_target_imports() -> None:
    _assert_entrypoint_target_imports("raft-uav-mmuad-track5-sequence-gate-fit")


def test_mmuad_train_sequence_classifier_entrypoint_is_exposed() -> None:
    assert (
        _project_scripts()["raft-uav-mmuad-train-sequence-classifier"]
        == "raft_uav.mmuad.train_sequence_classifier:main"
    )


def test_mmuad_train_sequence_classifier_entrypoint_target_imports() -> None:
    _assert_entrypoint_target_imports("raft-uav-mmuad-train-sequence-classifier")


def test_mmuad_sequence_alignment_audit_entrypoint_is_exposed() -> None:
    assert (
        _project_scripts()["raft-uav-mmuad-sequence-alignment-audit"]
        == "raft_uav.mmuad.sequence_alignment_audit:main"
    )


def test_mmuad_sequence_alignment_audit_entrypoint_target_imports() -> None:
    _assert_entrypoint_target_imports("raft-uav-mmuad-sequence-alignment-audit")


def test_mmuad_radar_extraction_audit_entrypoint_is_exposed() -> None:
    assert (
        _project_scripts()["raft-uav-mmuad-radar-extraction-audit"]
        == "raft_uav.mmuad.radar_extraction_audit:main"
    )


def test_mmuad_radar_extraction_audit_entrypoint_target_imports() -> None:
    _assert_entrypoint_target_imports("raft-uav-mmuad-radar-extraction-audit")


def test_mmuad_source_calibration_entrypoints_are_exposed() -> None:
    scripts = _project_scripts()

    assert (
        scripts["raft-uav-mmuad-fit-source-calibration"]
        == "raft_uav.mmuad.source_calibration:fit_main"
    )
    assert (
        scripts["raft-uav-mmuad-apply-source-calibration"]
        == "raft_uav.mmuad.source_calibration:apply_main"
    )


def test_mmuad_source_calibration_entrypoint_targets_import() -> None:
    _assert_entrypoint_target_imports("raft-uav-mmuad-fit-source-calibration")
    _assert_entrypoint_target_imports("raft-uav-mmuad-apply-source-calibration")


def test_mmuad_train_selected_config_entrypoint_is_exposed() -> None:
    assert (
        _project_scripts()["raft-uav-mmuad-select-train-config"]
        == "raft_uav.mmuad.train_selected_config:main"
    )


def test_mmuad_train_selected_config_entrypoint_target_imports() -> None:
    _assert_entrypoint_target_imports("raft-uav-mmuad-select-train-config")


def test_playbook_runnable_commands_are_installed_entrypoints() -> None:
    playbook = Path("docs/results_improvement_playbook.md").read_text(encoding="utf-8")
    documented = set(re.findall(r"`(raft-uav-[a-z0-9][a-z0-9-]*)`", playbook))

    assert documented <= set(_project_scripts())
