from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
from types import ModuleType

import pytest

from raft_uav.multi_uav_lts.upstream_patch import PatchedFile, UpstreamPatchReport


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "run_multi_uav_lts_competition_tracker.py"
)


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("competition_runner_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _report(root: Path, *, dry_run: bool = False) -> UpstreamPatchReport:
    return UpstreamPatchReport(
        botsort_root=str(root),
        changed=True,
        dry_run=dry_run,
        files=(
            PatchedFile(
                path="tracker/mc_bot_sort.py",
                action="updated",
                before_sha256="a",
                after_sha256="b",
            ),
        ),
    )


def test_competition_runner_patches_sets_environment_and_delegates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_script()
    captured: dict[str, object] = {}
    work_root = tmp_path / "work"
    botsort_root = work_root / "repos/YOLOv12-BoT-SORT-ReID/BoT-SORT"

    def fake_patch(root: Path, *, dry_run: bool = False) -> UpstreamPatchReport:
        captured["patch_root"] = root
        captured["patch_dry_run"] = dry_run
        return _report(root, dry_run=dry_run)

    def fake_baseline(argv: list[str] | None = None) -> int:
        captured["baseline_argv"] = list(argv or [])
        captured["closed_world"] = os.environ["RAFT_UAV_LTS_CLOSED_WORLD"]
        captured["association_mode"] = os.environ["RAFT_UAV_LTS_ASSOCIATION_MODE"]
        return 0

    monkeypatch.setattr(module, "apply_upstream_tracker_patch", fake_patch)
    monkeypatch.setattr(module, "_load_baseline_main", lambda: fake_baseline)

    result = module.main(
        [
            "--nwd-weight",
            "0.7",
            "--no-closed-world",
            "--coast-frames",
            "2",
            "--work-root",
            str(work_root),
            "--img-size",
            "1920",
        ]
    )

    assert result == 0
    assert captured["patch_root"] == botsort_root
    assert captured["patch_dry_run"] is False
    assert captured["closed_world"] == "false"
    assert captured["association_mode"] == "gated-weighted"
    baseline_argv = captured["baseline_argv"]
    assert isinstance(baseline_argv, list)
    assert baseline_argv[:4] == ["--work-root", str(work_root), "--img-size", "1920"]
    assert "--output-dir" in baseline_argv

    metadata = work_root / "outputs/competition_tracker/competition_tracker_config.json"
    payload = json.loads(metadata.read_text())
    assert payload["environment"]["RAFT_UAV_LTS_NWD_WEIGHT"] == "0.7"
    assert payload["environment"]["RAFT_UAV_LTS_COAST_FRAMES"] == "2"
    assert payload["upstream_patch"]["changed"] is True


def test_competition_runner_verify_only_uses_dry_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_script()
    calls: list[tuple[Path, bool]] = []

    def fake_patch(root: Path, *, dry_run: bool = False) -> UpstreamPatchReport:
        calls.append((root, dry_run))
        return _report(root, dry_run=dry_run)

    monkeypatch.setattr(module, "apply_upstream_tracker_patch", fake_patch)
    monkeypatch.setattr(
        module,
        "_load_baseline_main",
        lambda: pytest.fail("baseline must not run in verify-only mode"),
    )

    result = module.main(
        [
            "--verify-upstream-only",
            "--work-root",
            str(tmp_path),
        ]
    )

    assert result == 0
    assert len(calls) == 1
    assert calls[0][1] is True


def test_competition_runner_can_delegate_without_patching(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_script()
    monkeypatch.setattr(
        module,
        "apply_upstream_tracker_patch",
        lambda *args, **kwargs: pytest.fail("patcher should be disabled"),
    )
    monkeypatch.setattr(module, "_load_baseline_main", lambda: lambda _argv: 0)

    assert (
        module.main(
            [
                "--no-upstream-patch",
                "--work-root",
                str(tmp_path),
            ]
        )
        == 0
    )


def test_competition_runner_rejects_invalid_weights() -> None:
    module = _load_script()
    with pytest.raises(ValueError, match="nwd-weight"):
        module.main(["--nwd-weight", "1.1", "--patch-only"])
