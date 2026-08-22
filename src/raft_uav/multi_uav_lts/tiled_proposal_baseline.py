"""Run the LTS proposal baseline with extra proposal-only tiled passes."""

from __future__ import annotations

import math
import sys
from pathlib import Path

from . import proposal_baseline as baseline
from .improved_baseline import DEFAULT_WORK_ROOT, _has_flag, _option_value, _pop_option
from .proposal_export import apply_proposal_export_patch
from .tiled_proposal_export import apply_tiled_proposal_patch, write_summary
from .upstream_fixes import apply_upstream_fixes

DEFAULT_TILE_SIZE = 960
DEFAULT_TILE_OVERLAP = 0.25
DEFAULT_TILE_MAX_PER_FRAME = 256


def main(argv: list[str] | None = None) -> int:
    forwarded = list(sys.argv[1:] if argv is None else argv)
    help_requested = _has_flag(forwarded, "--help") or _has_flag(forwarded, "-h")
    tile_size = _positive_int(
        _pop_option(forwarded, "--proposal-tile-size") or DEFAULT_TILE_SIZE,
        name="proposal_tile_size",
    )
    tile_overlap = _fraction(
        _pop_option(forwarded, "--proposal-tile-overlap") or DEFAULT_TILE_OVERLAP,
        name="proposal_tile_overlap",
    )
    tile_max = _positive_int(
        _pop_option(forwarded, "--proposal-tile-max-per-frame")
        or DEFAULT_TILE_MAX_PER_FRAME,
        name="proposal_tile_max_per_frame",
    )
    patch_summary_override = _pop_option(forwarded, "--tiled-proposal-patch-json")

    work_root = Path(_option_value(forwarded, "--work-root") or DEFAULT_WORK_ROOT)
    botsort_root = Path(
        _option_value(forwarded, "--botsort-root")
        or work_root / "repos" / "YOLOv12-BoT-SORT-ReID" / "BoT-SORT"
    )
    output_value = _option_value(forwarded, "--output-dir")
    output_dir = (
        Path(output_value)
        if output_value
        else work_root / "outputs" / "tiled_proposal_baseline"
    )
    dry_run = _has_flag(forwarded, "--dry-run")
    package_only = _has_flag(forwarded, "--package-only")

    if not help_requested and not package_only:
        apply_upstream_fixes(botsort_root, check_only=dry_run)
        apply_proposal_export_patch(botsort_root, check_only=dry_run)
        summary = apply_tiled_proposal_patch(botsort_root, check_only=dry_run)
        summary_path = (
            Path(patch_summary_override)
            if patch_summary_override
            else output_dir / "tiled_proposal_patch_summary.json"
        )
        write_summary(summary, summary_path)

    original_installer = baseline._install_proposal_command_wrapper

    def install_with_tiles(
        runner,
        *,
        proposal_output_dir: Path,
        proposal_confidence: float,
        proposal_iou: float,
        suppress_visualizations: bool,
    ) -> None:
        original_installer(
            runner,
            proposal_output_dir=proposal_output_dir,
            proposal_confidence=proposal_confidence,
            proposal_iou=proposal_iou,
            suppress_visualizations=suppress_visualizations,
        )
        original_command = runner._inference_command

        def tiled_command(*args, **kwargs) -> list[str]:
            command = list(original_command(*args, **kwargs))
            command.extend(
                [
                    "--proposal-tile-size",
                    str(tile_size),
                    "--proposal-tile-overlap",
                    format(tile_overlap, ".15g"),
                    "--proposal-tile-max-per-frame",
                    str(tile_max),
                ]
            )
            return command

        runner._inference_command = tiled_command

    baseline._install_proposal_command_wrapper = install_with_tiles
    try:
        return baseline.main(forwarded)
    finally:
        baseline._install_proposal_command_wrapper = original_installer


def _positive_int(value: object, *, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer")
    try:
        parsed_float = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive integer") from exc
    if not math.isfinite(parsed_float) or not parsed_float.is_integer():
        raise ValueError(f"{name} must be a positive integer")
    parsed = int(parsed_float)
    if parsed <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return parsed


def _fraction(value: object, *, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be in [0, 1)")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be in [0, 1)") from exc
    if not math.isfinite(parsed) or not 0.0 <= parsed < 1.0:
        raise ValueError(f"{name} must be in [0, 1)")
    return parsed


if __name__ == "__main__":
    raise SystemExit(main())
