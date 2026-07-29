"""Run the official LTS baseline with RaFT-UAV's deterministic upstream fixes."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

from .upstream_fixes import apply_upstream_fixes, write_summary

REPO_ROOT = Path(__file__).resolve().parents[3]
OFFICIAL_RUNNER = REPO_ROOT / "scripts" / "run_multi_uav_lts_official_baseline.py"
DEFAULT_WORK_ROOT = Path("/mnt/lexar4tb/multi_uav_lts")


def main(argv: list[str] | None = None) -> int:
    forwarded = list(sys.argv[1:] if argv is None else argv)
    skip_fixes = _remove_flag(forwarded, "--skip-upstream-fixes")
    summary_override = _pop_option(forwarded, "--upstream-fixes-json")
    work_root = Path(_option_value(forwarded, "--work-root") or DEFAULT_WORK_ROOT)
    botsort_root = Path(
        _option_value(forwarded, "--botsort-root")
        or work_root / "repos" / "YOLOv12-BoT-SORT-ReID" / "BoT-SORT"
    )
    output_dir = Path(
        _option_value(forwarded, "--output-dir")
        or work_root / "outputs" / "official_baseline_via_first_init"
    )
    dry_run = _has_flag(forwarded, "--dry-run")
    package_only = _has_flag(forwarded, "--package-only")

    if not skip_fixes and not package_only:
        summary = apply_upstream_fixes(botsort_root, check_only=dry_run)
        summary_path = Path(summary_override) if summary_override else (
            output_dir / "upstream_fixes_summary.json"
        )
        write_summary(summary, summary_path)
        print(f"multi_uav_lts_upstream_fixes_json={summary_path}")
        print(f"multi_uav_lts_upstream_fixes_needed={summary.needs_update}")

    if _option_value(forwarded, "--img-size") is None:
        forwarded.extend(["--img-size", "1920"])
    runner = _load_official_runner()
    return int(runner.main(forwarded))


def _load_official_runner() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "_raft_uav_multi_uav_lts_official_runner", OFFICIAL_RUNNER
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load official LTS runner from {OFFICIAL_RUNNER}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _has_flag(arguments: list[str], option: str) -> bool:
    return option in arguments


def _remove_flag(arguments: list[str], option: str) -> bool:
    found = False
    while option in arguments:
        arguments.remove(option)
        found = True
    return found


def _option_value(arguments: list[str], option: str) -> str | None:
    prefix = option + "="
    for index, value in enumerate(arguments):
        if value.startswith(prefix):
            return _validated_option_value(
                value[len(prefix) :], option=option, reject_option_like=False
            )
        if value == option:
            if index + 1 >= len(arguments):
                raise ValueError(f"{option} requires a value")
            return _validated_option_value(
                arguments[index + 1], option=option, reject_option_like=True
            )
    return None


def _pop_option(arguments: list[str], option: str) -> str | None:
    prefix = option + "="
    for index, value in enumerate(arguments):
        if value.startswith(prefix):
            option_value = _validated_option_value(
                value[len(prefix) :], option=option, reject_option_like=False
            )
            arguments.pop(index)
            return option_value
        if value == option:
            if index + 1 >= len(arguments):
                raise ValueError(f"{option} requires a value")
            option_value = _validated_option_value(
                arguments[index + 1], option=option, reject_option_like=True
            )
            arguments.pop(index)
            arguments.pop(index)
            return option_value
    return None


def _validated_option_value(
    value: str, *, option: str, reject_option_like: bool
) -> str:
    if not value or (reject_option_like and value.startswith("-")):
        raise ValueError(f"{option} requires a value")
    return value


if __name__ == "__main__":
    raise SystemExit(main())
