"""Run the seeded LTS baseline while exporting a low-threshold proposal bank."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Callable

from ._records import validate_unit_interval
from .improved_baseline import (
    DEFAULT_WORK_ROOT,
    _has_flag,
    _load_official_runner,
    _option_value,
    _pop_option,
    _remove_flag,
)
from .proposal_export import apply_proposal_export_patch, write_summary as write_patch_summary
from .upstream_fixes import apply_upstream_fixes, write_summary as write_upstream_summary


DEFAULT_OUTPUT_NAME = "proposal_baseline"


def main(argv: list[str] | None = None) -> int:
    forwarded = list(sys.argv[1:] if argv is None else argv)
    help_requested = _has_flag(forwarded, "--help") or _has_flag(forwarded, "-h")
    skip_upstream = _remove_flag(forwarded, "--skip-upstream-fixes")
    skip_proposal_patch = _remove_flag(forwarded, "--skip-proposal-export-patch")
    save_visualizations = _remove_flag(forwarded, "--save-visualizations")
    upstream_summary_override = _pop_option(forwarded, "--upstream-fixes-json")
    proposal_patch_summary_override = _pop_option(
        forwarded,
        "--proposal-export-patch-json",
    )
    proposal_output_override = _pop_option(forwarded, "--proposal-output-dir")
    proposal_confidence_text = _pop_option(forwarded, "--proposal-conf-thres")
    proposal_iou_text = _pop_option(forwarded, "--proposal-iou-thres")
    wrapper_summary_override = _pop_option(forwarded, "--proposal-run-summary-json")

    work_root = Path(_option_value(forwarded, "--work-root") or DEFAULT_WORK_ROOT)
    botsort_root = Path(
        _option_value(forwarded, "--botsort-root")
        or work_root / "repos" / "YOLOv12-BoT-SORT-ReID" / "BoT-SORT"
    )
    output_value = _option_value(forwarded, "--output-dir")
    output_dir = Path(output_value) if output_value else (
        work_root / "outputs" / DEFAULT_OUTPUT_NAME
    )
    if output_value is None:
        forwarded.extend(["--output-dir", str(output_dir)])
    proposal_output_dir = Path(proposal_output_override) if proposal_output_override else (
        output_dir / "proposals"
    )
    proposal_confidence = validate_unit_interval(
        0.001 if proposal_confidence_text is None else proposal_confidence_text,
        name="proposal_conf_thres",
    )
    proposal_iou = validate_unit_interval(
        0.95 if proposal_iou_text is None else proposal_iou_text,
        name="proposal_iou_thres",
    )
    dry_run = _has_flag(forwarded, "--dry-run")
    package_only = _has_flag(forwarded, "--package-only")

    upstream_summary_path = Path(upstream_summary_override) if upstream_summary_override else (
        output_dir / "upstream_fixes_summary.json"
    )
    proposal_patch_summary_path = (
        Path(proposal_patch_summary_override)
        if proposal_patch_summary_override
        else output_dir / "proposal_export_patch_summary.json"
    )
    if not package_only and not help_requested:
        if not skip_upstream:
            upstream_summary = apply_upstream_fixes(
                botsort_root,
                check_only=dry_run,
            )
            write_upstream_summary(upstream_summary, upstream_summary_path)
        if not skip_proposal_patch:
            proposal_patch_summary = apply_proposal_export_patch(
                botsort_root,
                check_only=dry_run,
            )
            write_patch_summary(proposal_patch_summary, proposal_patch_summary_path)

    if _option_value(forwarded, "--img-size") is None:
        forwarded.extend(["--img-size", "1920"])
    runner = _load_official_runner()
    _install_proposal_command_wrapper(
        runner,
        proposal_output_dir=proposal_output_dir,
        proposal_confidence=proposal_confidence,
        proposal_iou=proposal_iou,
        suppress_visualizations=not save_visualizations,
    )
    return_code = int(runner.main(forwarded))
    wrapper_summary_path = Path(wrapper_summary_override) if wrapper_summary_override else (
        output_dir / "proposal_run_summary.json"
    )
    _write_run_summary(
        wrapper_summary_path,
        work_root=work_root,
        botsort_root=botsort_root,
        output_dir=output_dir,
        proposal_output_dir=proposal_output_dir,
        proposal_confidence=proposal_confidence,
        proposal_iou=proposal_iou,
        dry_run=dry_run,
        package_only=package_only,
        save_visualizations=save_visualizations,
        return_code=return_code,
    )
    print(f"multi_uav_lts_proposal_dir={proposal_output_dir}")
    print(f"multi_uav_lts_proposal_run_summary={wrapper_summary_path}")
    return return_code


def _install_proposal_command_wrapper(
    runner: ModuleType,
    *,
    proposal_output_dir: Path,
    proposal_confidence: float,
    proposal_iou: float,
    suppress_visualizations: bool,
) -> None:
    original: Callable[..., list[str]] = runner._inference_command

    def proposal_command(*args, **kwargs) -> list[str]:
        command = list(original(*args, **kwargs))
        command.extend(
            [
                "--proposal-output-dir",
                str(proposal_output_dir),
                "--proposal-conf-thres",
                format(proposal_confidence, ".15g"),
                "--proposal-iou-thres",
                format(proposal_iou, ".15g"),
            ]
        )
        if suppress_visualizations:
            command.append("--nosave")
        return command

    runner._inference_command = proposal_command


def _write_run_summary(
    path: Path,
    *,
    work_root: Path,
    botsort_root: Path,
    output_dir: Path,
    proposal_output_dir: Path,
    proposal_confidence: float,
    proposal_iou: float,
    dry_run: bool,
    package_only: bool,
    save_visualizations: bool,
    return_code: int,
) -> None:
    payload = {
        "schema": "raft-uav-multi-uav-lts-proposal-baseline-v1",
        "work_root": str(work_root),
        "botsort_root": str(botsort_root),
        "output_dir": str(output_dir),
        "proposal_output_dir": str(proposal_output_dir),
        "proposal_confidence_threshold": proposal_confidence,
        "proposal_iou_threshold": proposal_iou,
        "dry_run": dry_run,
        "package_only": package_only,
        "save_visualizations": save_visualizations,
        "return_code": return_code,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    raise SystemExit(main())
