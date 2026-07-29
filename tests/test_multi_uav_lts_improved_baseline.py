from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from raft_uav.multi_uav_lts import improved_baseline
from raft_uav.multi_uav_lts.upstream_fixes import UpstreamFixSummary


def _summary(root: Path, *, check_only: bool) -> UpstreamFixSummary:
    return UpstreamFixSummary(
        botsort_root=str(root),
        check_only=check_only,
        needs_update=check_only,
        changed_file_count=0 if check_only else 2,
        files=(),
    )


def test_improved_runner_applies_fixes_and_defaults_to_1920(
    tmp_path: Path, monkeypatch
) -> None:
    captured: dict[str, object] = {}

    def fake_apply(root: Path, *, check_only: bool = False):
        captured["root"] = root
        captured["check_only"] = check_only
        return _summary(root, check_only=check_only)

    def fake_write(summary: UpstreamFixSummary, path: Path) -> None:
        captured["summary_path"] = path

    def fake_main(arguments: list[str]) -> int:
        captured["arguments"] = arguments
        return 7

    monkeypatch.setattr(improved_baseline, "apply_upstream_fixes", fake_apply)
    monkeypatch.setattr(improved_baseline, "write_summary", fake_write)
    monkeypatch.setattr(
        improved_baseline,
        "_load_official_runner",
        lambda: SimpleNamespace(main=fake_main),
    )

    result = improved_baseline.main(["--work-root", str(tmp_path), "--device", "0"])

    assert result == 7
    assert captured["root"] == (
        tmp_path / "repos" / "YOLOv12-BoT-SORT-ReID" / "BoT-SORT"
    )
    assert captured["check_only"] is False
    assert captured["summary_path"] == (
        tmp_path / "outputs" / "official_baseline_via_first_init"
        / "upstream_fixes_summary.json"
    )
    assert captured["arguments"][-2:] == ["--img-size", "1920"]


def test_dry_run_checks_without_writing_upstream_sources(
    tmp_path: Path, monkeypatch
) -> None:
    captured: dict[str, object] = {}

    def fake_apply(root: Path, *, check_only: bool = False):
        captured["check_only"] = check_only
        return _summary(root, check_only=check_only)

    monkeypatch.setattr(improved_baseline, "apply_upstream_fixes", fake_apply)
    monkeypatch.setattr(improved_baseline, "write_summary", lambda summary, path: None)
    monkeypatch.setattr(
        improved_baseline,
        "_load_official_runner",
        lambda: SimpleNamespace(main=lambda arguments: 0),
    )

    improved_baseline.main(["--work-root", str(tmp_path), "--dry-run"])

    assert captured["check_only"] is True


def test_package_only_skips_upstream_patch(tmp_path: Path, monkeypatch) -> None:
    def unexpected(*args, **kwargs):
        raise AssertionError("upstream patch should not run for package-only")

    captured: dict[str, object] = {}
    monkeypatch.setattr(improved_baseline, "apply_upstream_fixes", unexpected)

    def fake_main(arguments: list[str]) -> int:
        captured["arguments"] = arguments
        return 0

    monkeypatch.setattr(
        improved_baseline,
        "_load_official_runner",
        lambda: SimpleNamespace(main=fake_main),
    )

    result = improved_baseline.main(
        ["--work-root", str(tmp_path), "--package-only", "--img-size", "1600"]
    )

    assert result == 0
    assert captured["arguments"][-2:] == ["--img-size", "1600"]


def test_missing_summary_value_does_not_consume_dry_run(monkeypatch) -> None:
    def unexpected(*args, **kwargs):
        raise AssertionError("upstream patch should not run for malformed arguments")

    monkeypatch.setattr(improved_baseline, "apply_upstream_fixes", unexpected)

    with pytest.raises(ValueError, match="--upstream-fixes-json requires a value"):
        improved_baseline.main(["--upstream-fixes-json", "--dry-run"])


def test_missing_forwarded_option_value_is_rejected() -> None:
    with pytest.raises(ValueError, match="--work-root requires a value"):
        improved_baseline.main(["--work-root", "--package-only"])


def test_empty_inline_wrapper_value_is_rejected() -> None:
    with pytest.raises(ValueError, match="--upstream-fixes-json requires a value"):
        improved_baseline.main(["--upstream-fixes-json="])
