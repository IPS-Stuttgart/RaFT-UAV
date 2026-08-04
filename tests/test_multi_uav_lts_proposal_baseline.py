from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from raft_uav.multi_uav_lts import proposal_baseline


def test_wrapper_applies_patches_and_forwards_proposal_controls(
    tmp_path: Path,
    monkeypatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_upstream(root: Path, *, check_only: bool = False):
        captured["upstream"] = (root, check_only)
        return SimpleNamespace(needs_update=check_only)

    def fake_proposal_patch(root: Path, *, check_only: bool = False):
        captured["proposal_patch"] = (root, check_only)
        return SimpleNamespace(needs_update=check_only)

    monkeypatch.setattr(proposal_baseline, "apply_upstream_fixes", fake_upstream)
    monkeypatch.setattr(
        proposal_baseline,
        "write_upstream_summary",
        lambda summary, path: captured.setdefault("upstream_summary", path),
    )
    monkeypatch.setattr(
        proposal_baseline,
        "apply_proposal_export_patch",
        fake_proposal_patch,
    )
    monkeypatch.setattr(
        proposal_baseline,
        "write_patch_summary",
        lambda summary, path: captured.setdefault("proposal_summary", path),
    )

    runner = SimpleNamespace(
        _inference_command=lambda *args, **kwargs: ["python", "tools/inference.py"]
    )

    def fake_main(arguments: list[str]) -> int:
        captured["arguments"] = arguments
        captured["command"] = runner._inference_command(None, None, None)
        return 0

    runner.main = fake_main
    monkeypatch.setattr(proposal_baseline, "_load_official_runner", lambda: runner)

    result = proposal_baseline.main(
        ["--work-root", str(tmp_path), "--no-template", "--dry-run"]
    )

    assert result == 0
    assert captured["upstream"] == (
        tmp_path / "repos" / "YOLOv12-BoT-SORT-ReID" / "BoT-SORT",
        True,
    )
    assert captured["proposal_patch"] == captured["upstream"]
    assert captured["arguments"][-2:] == ["--img-size", "1920"]
    assert captured["command"][-6:] == [
        "--proposal-output-dir",
        str(tmp_path / "outputs" / "proposal_baseline" / "proposals"),
        "--proposal-conf-thres",
        "0.001",
        "--proposal-iou-thres",
        "0.95",
    ]
    assert (
        tmp_path / "outputs" / "proposal_baseline" / "proposal_run_summary.json"
    ).is_file()


def test_package_only_skips_external_source_patches(
    tmp_path: Path,
    monkeypatch,
) -> None:
    def unexpected(*args, **kwargs):
        raise AssertionError("external patches must not run during package-only")

    monkeypatch.setattr(proposal_baseline, "apply_upstream_fixes", unexpected)
    monkeypatch.setattr(
        proposal_baseline,
        "apply_proposal_export_patch",
        unexpected,
    )
    runner = SimpleNamespace(
        _inference_command=lambda *args, **kwargs: [],
        main=lambda arguments: 0,
    )
    monkeypatch.setattr(proposal_baseline, "_load_official_runner", lambda: runner)

    assert (
        proposal_baseline.main(
            ["--work-root", str(tmp_path), "--package-only"]
        )
        == 0
    )


def test_invalid_proposal_threshold_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="proposal_conf_thres"):
        proposal_baseline.main(
            [
                "--work-root",
                str(tmp_path),
                "--proposal-conf-thres",
                "1.1",
            ]
        )
