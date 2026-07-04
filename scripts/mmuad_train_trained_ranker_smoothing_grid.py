#!/usr/bin/env python
"""Run non-learned smoothing/completion over train-trained MMUAD ranker outputs.

This is an experiment runner, not a core method change.  It consumes existing
``mmuad_estimates.csv`` files from train-trained ranker runs, applies a compact
trajectory-completion grid, and writes local Track 5 score summaries.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from raft_uav.mmuad.completion import complete_results_to_truth_timestamps  # noqa: E402
from raft_uav.mmuad.evaluator import evaluate_mmaud_results, load_evaluation_truth_file  # noqa: E402
from raft_uav.mmuad.submission import (  # noqa: E402
    estimates_to_mmaud_results_frame,
    load_official_track5_template_file,
    write_official_mmaud_results_csv,
    write_official_ug2_codabench_zip,
)
from raft_uav.mmuad.trajectory_completion import (  # noqa: E402
    TrajectoryCompletionConfig,
    complete_and_smooth_estimates,
    write_trajectory_completion_diagnostics,
)


DEFAULT_COMPLETION_MODES = ("none", "gap-interpolation", "constant-velocity", "fixed-lag")
DEFAULT_SPEED_GATES_MPS = (10.0, 15.0, 20.0, 30.0)
DEFAULT_SMOOTHING_BLENDS = (0.25, 0.5, 0.75, 1.0)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ranker-output-dir",
        type=Path,
        help="directory containing ranker run subdirectories with mmuad_estimates.csv",
    )
    parser.add_argument(
        "--base-estimates-csv",
        action="append",
        type=Path,
        default=[],
        help="specific mmuad_estimates.csv to post-process; may be repeated",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--val-truth", type=Path)
    parser.add_argument("--val-template", type=Path)
    parser.add_argument("--classification", default="2")
    parser.add_argument("--completion-max-gap-s", type=float, default=3.0)
    parser.add_argument("--fixed-lag-s", type=float, default=3.0)
    parser.add_argument("--speed-gates-mps", nargs="+", type=float, default=list(DEFAULT_SPEED_GATES_MPS))
    parser.add_argument("--smoothing-blends", nargs="+", type=float, default=list(DEFAULT_SMOOTHING_BLENDS))
    parser.add_argument("--completion-modes", nargs="+", default=list(DEFAULT_COMPLETION_MODES))
    parser.add_argument("--completion-max-interpolation-gap-s", type=float, default=1.0)
    parser.add_argument("--timestamp-tolerance-s", type=float, default=1.0e-6)
    args = parser.parse_args(argv)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    val_truth = load_evaluation_truth_file(args.val_truth) if args.val_truth else None
    template_rows = _load_template_rows(args.val_template, val_truth)
    estimate_files = _estimate_files(args)
    if not estimate_files:
        raise SystemExit("no mmuad_estimates.csv inputs found")

    records: list[dict[str, Any]] = []
    for estimate_csv in estimate_files:
        base_estimates = _load_base_estimates(estimate_csv)
        ranker_run = _ranker_run_name(estimate_csv, args.ranker_output_dir)
        for spec in _smoothing_specs(args):
            run_dir = args.output_dir / ranker_run / spec["run"]
            run_dir.mkdir(parents=True, exist_ok=True)
            print(f"smoothing_run={ranker_run}/{spec['run']}", flush=True)
            records.append(
                _run_spec(
                    base_estimates,
                    ranker_run=ranker_run,
                    source_estimates_csv=estimate_csv,
                    run_dir=run_dir,
                    spec=spec,
                    val_truth=val_truth,
                    template_rows=template_rows,
                    args=args,
                )
            )
            _write_summary(args.output_dir, records)

    _write_summary(args.output_dir, records)
    print(f"smoothing_grid_csv={args.output_dir / 'mmuad_train_trained_ranker_smoothing_grid.csv'}")
    print(f"smoothing_grid_json={args.output_dir / 'mmuad_train_trained_ranker_smoothing_grid.json'}")
    return 0


def _estimate_files(args: argparse.Namespace) -> list[Path]:
    files = [Path(path) for path in args.base_estimates_csv]
    if args.ranker_output_dir is not None:
        root = Path(args.ranker_output_dir)
        output_dir = Path(args.output_dir)
        files.extend(
            path
            for path in root.glob("**/mmuad_estimates.csv")
            if path.is_file() and not _is_relative_to(path, output_dir)
        )
    return sorted(dict.fromkeys(files))


def _ranker_run_name(path: Path, root: Path | None) -> str:
    if root is not None:
        try:
            relative = _resolved_path(path).relative_to(_resolved_path(root))
            if len(relative.parts) > 1:
                return str(relative.parts[0])
        except ValueError:
            pass
    return path.parent.name or "direct"


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        _resolved_path(path).relative_to(_resolved_path(root))
    except ValueError:
        return False
    return True


def _resolved_path(path: Path) -> Path:
    return Path(path).expanduser().resolve()


def _load_template_rows(path: Path | None, truth) -> pd.DataFrame | None:
    if path is not None:
        try:
            return load_evaluation_truth_file(path).rows
        except Exception:
            return load_official_track5_template_file(path)
    if truth is not None:
        return truth.rows
    return None


def _smoothing_specs(args: argparse.Namespace) -> tuple[dict[str, Any], ...]:
    specs: list[dict[str, Any]] = []
    for mode in args.completion_modes:
        normalized_mode = str(mode).strip()
        for speed_gate_mps in args.speed_gates_mps:
            for blend in args.smoothing_blends:
                speed = float(speed_gate_mps)
                blend_value = float(blend)
                specs.append(
                    {
                        "run": _spec_run_name(normalized_mode, speed, blend_value),
                        "mode": normalized_mode,
                        "max_gap_s": float(args.completion_max_gap_s),
                        "lag_s": float(args.fixed_lag_s),
                        "blend": blend_value,
                        "speed_gate_mps": speed,
                        "outlier_replacement": "local-linear" if speed > 0.0 else "none",
                    }
                )
    return tuple(specs)


def _spec_run_name(mode: str, speed_gate_mps: float, blend: float) -> str:
    mode_name = mode.replace("-", "_")
    blend_name = f"{blend:.2f}".rstrip("0").rstrip(".").replace(".", "p")
    return f"{mode_name}_speed{speed_gate_mps:g}_blend{blend_name}"


def _load_base_estimates(path: Path) -> pd.DataFrame:
    rows = pd.read_csv(path)
    if "trajectory_completion_filled" in rows.columns:
        filled = _bool_series(rows["trajectory_completion_filled"])
        rows = rows.loc[~filled].copy()
    for state_column, original_column in {
        "state_x_m": "trajectory_original_state_x_m",
        "state_y_m": "trajectory_original_state_y_m",
        "state_z_m": "trajectory_original_state_z_m",
    }.items():
        if original_column in rows.columns:
            rows[state_column] = pd.to_numeric(rows[original_column], errors="coerce")
    return rows.reset_index(drop=True)


def _bool_series(values: pd.Series) -> pd.Series:
    if values.dtype == bool:
        return values.fillna(False)
    return values.astype(str).str.strip().str.lower().isin({"1", "true", "yes"})


def _run_spec(
    base_estimates: pd.DataFrame,
    *,
    ranker_run: str,
    source_estimates_csv: Path,
    run_dir: Path,
    spec: dict[str, Any],
    val_truth,
    template_rows: pd.DataFrame | None,
    args: argparse.Namespace,
) -> dict[str, Any]:
    started = time.time()
    result = complete_and_smooth_estimates(
        base_estimates,
        None if val_truth is None else val_truth.rows,
        config=TrajectoryCompletionConfig(
            mode=spec["mode"],
            max_gap_s=float(spec["max_gap_s"]),
            fixed_lag_s=float(spec["lag_s"]),
            smoothing_blend=float(spec["blend"]),
            include_truth_timestamps=True,
            infer_missing_grid=True,
            speed_gate_mps=float(spec["speed_gate_mps"]),
            outlier_replacement=spec["outlier_replacement"],
        ),
    )
    write_trajectory_completion_diagnostics(result, run_dir)
    result.estimates.to_csv(run_dir / "mmuad_estimates.csv", index=False)
    legacy = estimates_to_mmaud_results_frame(
        result.estimates,
        class_name=str(args.classification),
    )
    legacy = _force_classification(legacy, args.classification)
    legacy.to_csv(run_dir / "mmaud_results_legacy.csv", index=False)

    completed_rows = legacy
    if template_rows is not None:
        completion = complete_results_to_truth_timestamps(
            legacy,
            template_rows,
            max_interpolation_gap_s=args.completion_max_interpolation_gap_s,
            extrapolation="hold",
        )
        completed_rows = completion.rows
        completed_rows = _force_classification(completed_rows, args.classification)
        completed_rows.to_csv(run_dir / "mmaud_results_legacy_completed.csv", index=False)
        completion.diagnostics.to_csv(
            run_dir / "mmuad_official_timestamp_completion_rows.csv",
            index=False,
        )

    official_csv = run_dir / "mmaud_results.csv"
    official_zip = run_dir / "ug2_submission.zip"
    write_official_mmaud_results_csv(
        completed_rows,
        official_csv,
        classification=args.classification,
    )
    write_official_ug2_codabench_zip(
        completed_rows,
        official_zip,
        classification=args.classification,
    )

    record: dict[str, Any] = {
        "ranker_run": ranker_run,
        "smoothing_run": spec["run"],
        "source_estimates_csv": str(source_estimates_csv),
        "mode": spec["mode"],
        "max_gap_s": float(spec["max_gap_s"]),
        "lag_s": float(spec["lag_s"]),
        "blend": float(spec["blend"]),
        "speed_gate_mps": float(spec["speed_gate_mps"]),
        "outlier_replacement": spec["outlier_replacement"],
        "estimates_csv": str(run_dir / "mmuad_estimates.csv"),
        "mmaud_results_csv": str(official_csv),
        "ug2_submission_zip": str(official_zip),
        "elapsed_s": float(time.time() - started),
    }
    if val_truth is not None:
        evaluation = evaluate_mmaud_results(
            completed_rows,
            val_truth.rows,
            metric_protocol="public-track5" if template_rows is not None else "nearest-time",
            timestamp_tolerance_s=args.timestamp_tolerance_s,
        )
        (run_dir / "track5_scorecard_smoothing.json").write_text(
            json.dumps(evaluation["summary"], indent=2),
            encoding="utf-8",
        )
        evaluation["rows"].to_csv(run_dir / "track5_scorecard_rows.csv", index=False)
        pooled = evaluation["summary"].get("pooled", {})
        for key in (
            "pose_mse_loss_m2",
            "rmse_3d_m",
            "mean_3d_m",
            "p95_3d_m",
            "max_3d_m",
            "classification_accuracy",
            "uav_type_accuracy",
        ):
            if key in pooled:
                record[key] = pooled[key]
        record["scorecard_leaderboard_ready"] = evaluation["summary"].get("leaderboard_ready")
    return record


def _force_classification(rows: pd.DataFrame, classification: Any) -> pd.DataFrame:
    out = rows.copy()
    out["uav_type"] = str(classification)
    out["classification"] = str(classification)
    return out


def _write_summary(output_dir: Path, records: list[dict[str, Any]]) -> None:
    frame = pd.DataFrame.from_records(records)
    frame.to_csv(output_dir / "mmuad_train_trained_ranker_smoothing_grid.csv", index=False)
    (output_dir / "mmuad_train_trained_ranker_smoothing_grid.json").write_text(
        json.dumps(records, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
