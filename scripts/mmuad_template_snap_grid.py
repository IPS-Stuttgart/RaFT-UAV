#!/usr/bin/env python
"""Sweep official-template snapping policies for MMUAD Track 5 results.

This is a lightweight Codabench-preflight helper. It wraps
``mmuad_snap_official_results_to_template.py`` over resampling/classification
policy combinations, writes one upload-ready artifact bundle per variant, and
optionally scores each variant against a public validation reference. It does
not read hidden-test labels and can be used for both validation diagnostics and
final test-template packaging checks.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
SCRIPT_ROOT = Path(__file__).resolve().parent
for root in (SRC_ROOT, SCRIPT_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from mmuad_snap_official_results_to_template import (  # noqa: E402
    CLASSIFICATION_POLICIES,
    MISSING_POSITION_POLICIES,
    RESAMPLE_METHODS,
    write_template_snapped_submission,
)
from raft_uav.mmuad.submission import (  # noqa: E402
    load_official_track5_results_frame,
    load_official_track5_template_file,
)
from raft_uav.mmuad.track5_scorecard import build_track5_scorecard  # noqa: E402

SUMMARY_CSV = "mmuad_template_snap_grid_summary.csv"
SUMMARY_JSON = "mmuad_template_snap_grid_summary.json"


def run_template_snap_grid(
    *,
    results: pd.DataFrame,
    template: pd.DataFrame,
    output_dir: Path,
    resample_methods: tuple[str, ...] = ("linear", "nearest"),
    max_interpolation_gaps_s: tuple[float | None, ...] = (None,),
    classification_policies: tuple[str, ...] = ("sequence-mode",),
    missing_position_policy: str = "zero",
    truth_path: Path | None = None,
    template_path: Path | None = None,
    class_map_path: Path | None = None,
    timestamp_tolerance_s: float = 1.0e-6,
) -> pd.DataFrame:
    """Run the snapper grid and return a ranked summary table."""

    output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    for method in resample_methods:
        _validate_choice(method, RESAMPLE_METHODS, "resample_method")
        for gap_s in max_interpolation_gaps_s:
            for class_policy in classification_policies:
                _validate_choice(
                    class_policy,
                    CLASSIFICATION_POLICIES,
                    "classification_policy",
                )
                _validate_choice(
                    missing_position_policy,
                    MISSING_POSITION_POLICIES,
                    "missing_position_policy",
                )
                label = _variant_label(method, gap_s, class_policy)
                variant_dir = output_dir / label
                paths = write_template_snapped_submission(
                    results=results,
                    template=template,
                    output_dir=variant_dir,
                    resample_method=method,  # type: ignore[arg-type]
                    max_interpolation_gap_s=gap_s,
                    classification_policy=class_policy,  # type: ignore[arg-type]
                    missing_position_policy=missing_position_policy,  # type: ignore[arg-type]
                )
                validation = json.loads(paths["validation_json"].read_text(encoding="utf-8"))
                manifest = json.loads(paths["manifest_json"].read_text(encoding="utf-8"))
                scorecard_summary: dict[str, Any] = {}
                if truth_path is not None:
                    scorecard = build_track5_scorecard(
                        results_path=paths["official_zip"],
                        truth_path=truth_path,
                        template_path=template_path,
                        class_map_path=class_map_path,
                        require_zip=True,
                        timestamp_tolerance_s=timestamp_tolerance_s,
                    )
                    scorecard_summary = scorecard.summary
                    scorecard_path = variant_dir / "track5_scorecard.json"
                    scorecard_path.write_text(
                        json.dumps(_jsonable(scorecard.summary), indent=2),
                        encoding="utf-8",
                    )
                records.append(
                    _summary_record(
                        label=label,
                        method=method,
                        gap_s=gap_s,
                        class_policy=class_policy,
                        paths=paths,
                        validation=validation,
                        manifest=manifest,
                        scorecard_summary=scorecard_summary,
                    )
                )
    summary = pd.DataFrame.from_records(records)
    summary = _rank_summary(summary)
    summary.to_csv(output_dir / SUMMARY_CSV, index=False)
    (output_dir / SUMMARY_JSON).write_text(
        json.dumps({"rows": _jsonable(summary.to_dict(orient="records"))}, indent=2),
        encoding="utf-8",
    )
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True, help="official CSV/ZIP to snap")
    parser.add_argument(
        "--template",
        type=Path,
        required=True,
        help="official Track 5 template CSV/ZIP",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resample-method", action="append", choices=RESAMPLE_METHODS)
    parser.add_argument(
        "--max-interpolation-gap-s",
        action="append",
        default=["none"],
        help="gap value in seconds, 'none', or comma-separated list; may repeat",
    )
    parser.add_argument(
        "--classification-policy",
        action="append",
        choices=CLASSIFICATION_POLICIES,
    )
    parser.add_argument(
        "--missing-position-policy",
        choices=MISSING_POSITION_POLICIES,
        default="zero",
    )
    parser.add_argument(
        "--truth",
        type=Path,
        help="optional public validation reference for local scoring",
    )
    parser.add_argument("--class-map", type=Path, help="optional sequence-to-class map for scoring")
    parser.add_argument("--timestamp-tolerance-s", type=float, default=1.0e-6)
    parser.add_argument(
        "--require-at-least-one-leaderboard-ready",
        action="store_true",
        help="exit nonzero unless at least one variant passes local upload validation",
    )
    args = parser.parse_args(argv)

    results = load_official_track5_results_frame(args.results)
    template = load_official_track5_template_file(args.template)
    summary = run_template_snap_grid(
        results=results,
        template=template,
        output_dir=args.output_dir,
        resample_methods=tuple(args.resample_method or RESAMPLE_METHODS),
        max_interpolation_gaps_s=_parse_gap_values(args.max_interpolation_gap_s),
        classification_policies=tuple(args.classification_policy or ("sequence-mode",)),
        missing_position_policy=args.missing_position_policy,
        truth_path=args.truth,
        template_path=args.template,
        class_map_path=args.class_map,
        timestamp_tolerance_s=float(args.timestamp_tolerance_s),
    )
    print("mmuad_template_snap_grid=ok")
    print(f"summary_csv={args.output_dir / SUMMARY_CSV}")
    if not summary.empty:
        best = summary.iloc[0]
        print(f"best_variant={best['variant_label']}")
        print(f"best_codabench_upload_ready={best['codabench_upload_ready']}")
        if pd.notna(best.get("pose_mse_loss_m2", np.nan)):
            print(f"best_pose_mse_loss_m2={best['pose_mse_loss_m2']}")
    ready = bool(summary["leaderboard_ready"].any()) if not summary.empty else False
    if args.require_at_least_one_leaderboard_ready and not ready:
        raise SystemExit("no template-snap grid variant was leaderboard-ready")
    return 0


def _summary_record(
    *,
    label: str,
    method: str,
    gap_s: float | None,
    class_policy: str,
    paths: dict[str, Path],
    validation: dict[str, Any],
    manifest: dict[str, Any],
    scorecard_summary: dict[str, Any],
) -> dict[str, Any]:
    public_pooled = (scorecard_summary.get("public_track5") or {}).get("pooled") or {}
    type_accuracy = public_pooled.get("uav_type_accuracy")
    if type_accuracy is None:
        type_accuracy = public_pooled.get("classification_accuracy")
    return {
        "variant_label": label,
        "resample_method": method,
        "max_interpolation_gap_s": gap_s,
        "classification_policy": class_policy,
        "row_count": int(manifest.get("row_count", 0)),
        "template_row_count": int(manifest.get("template_row_count", 0)),
        "source_result_rows": int(manifest.get("source_result_rows", 0)),
        "valid_snapped_rows": int(manifest.get("valid_snapped_rows", 0)),
        "invalid_snapped_rows": int(manifest.get("invalid_snapped_rows", 0)),
        "extrapolated_rows": int(manifest.get("extrapolated_rows", 0)),
        "large_gap_fallback_rows": int(manifest.get("large_gap_fallback_rows", 0)),
        "leaderboard_ready": bool(validation.get("leaderboard_ready", False)),
        "codabench_upload_ready": bool(validation.get("codabench_upload_ready", False)),
        "validation_blocking_reasons": ";".join(
            validation.get("leaderboard_blocking_reasons", []),
        ),
        "pose_mse_loss_m2": _optional_float(public_pooled.get("pose_mse_loss_m2")),
        "pose_rmse_m": _optional_float(public_pooled.get("pose_rmse_m")),
        "pose_mean_error_3d_m": _optional_float(
            public_pooled.get("mean_position_error_3d_m"),
        ),
        "pose_p95_error_3d_m": _optional_float(
            public_pooled.get("p95_position_error_3d_m"),
        ),
        "pose_max_error_3d_m": _optional_float(
            public_pooled.get("max_position_error_3d_m"),
        ),
        "uav_type_accuracy": _optional_float(type_accuracy),
        "official_zip": str(paths["official_zip"]),
        "official_results_csv": str(paths["official_results_csv"]),
        "diagnostics_csv": str(paths["diagnostics_csv"]),
        "validation_json": str(paths["validation_json"]),
        "manifest_json": str(paths["manifest_json"]),
    }


def _rank_summary(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return summary
    rows = summary.copy()
    if rows["pose_mse_loss_m2"].notna().any():
        rows["_pose_sort"] = rows["pose_mse_loss_m2"].fillna(np.inf)
    else:
        rows["_pose_sort"] = np.inf
    rows = rows.sort_values(
        [
            "leaderboard_ready",
            "codabench_upload_ready",
            "_pose_sort",
            "invalid_snapped_rows",
            "large_gap_fallback_rows",
            "extrapolated_rows",
            "variant_label",
        ],
        ascending=[False, False, True, True, True, True, True],
    ).reset_index(drop=True)
    return rows.drop(columns=["_pose_sort"])


def _parse_gap_values(values: list[str]) -> tuple[float | None, ...]:
    parsed: list[float | None] = []
    for value in values:
        for item in str(value).replace(";", ",").split(","):
            text = item.strip().lower()
            if not text:
                continue
            if text in {"none", "null", "na"}:
                parsed.append(None)
            else:
                parsed.append(float(text))
    unique: list[float | None] = []
    seen: set[str] = set()
    for value in parsed or [None]:
        key = "none" if value is None else f"{float(value):.12g}"
        if key not in seen:
            seen.add(key)
            unique.append(value)
    return tuple(unique)


def _variant_label(method: str, gap_s: float | None, class_policy: str) -> str:
    gap_label = "gap_none" if gap_s is None else f"gap_{float(gap_s):.6g}s"
    return "_".join(
        [
            str(method).replace("-", "_"),
            gap_label.replace(".", "p"),
            str(class_policy).replace("-", "_"),
        ],
    )


def _validate_choice(value: str, choices: tuple[str, ...], name: str) -> None:
    if str(value) not in choices:
        raise ValueError(f"{name} must be one of {choices}; got {value!r}")


def _optional_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(number):
        return None
    return number


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
