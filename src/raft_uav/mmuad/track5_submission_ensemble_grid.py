"""Weight search for official MMUAD/UG2+ Track 5 submission ensembles.

``raft-uav-mmuad-ensemble-track5-submissions`` is inference-safe and accepts
explicit weights.  This companion utility uses a local truth/template file to
score a small weight grid over already-generated official submissions, then
writes the best upload-ready ensemble artifact.  It is intended for train-fold
or public-validation diagnostics before freezing weights for hidden-test
submission; truth is used only for grid scoring.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from itertools import product
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from raft_uav.mmuad.evaluator import evaluate_mmaud_results, load_evaluation_truth_file
from raft_uav.mmuad.submission import load_official_track5_template_file
from raft_uav.mmuad.track5_submission_ensemble import SubmissionInput
from raft_uav.mmuad.track5_submission_ensemble import ensemble_track5_submissions
from raft_uav.mmuad.track5_submission_ensemble import parse_submission_input
from raft_uav.mmuad.track5_submission_ensemble import write_track5_submission_ensemble_outputs

GRID_SUMMARY_CSV = "mmuad_track5_submission_ensemble_weight_grid.csv"
GRID_BY_SEQUENCE_CSV = "mmuad_track5_submission_ensemble_weight_grid_by_sequence.csv"
GRID_MANIFEST_JSON = "mmuad_track5_submission_ensemble_weight_grid_manifest.json"
BEST_OUTPUT_DIR = "best_submission_ensemble"


@dataclass(frozen=True)
class SubmissionGridRow:
    """One scored candidate submission-ensemble weight vector."""

    weights: tuple[float, ...]
    class_policy: str
    pose_mse: float
    rmse_m: float
    mean_error_m: float
    p95_error_m: float
    max_error_m: float
    class_accuracy: float | None
    matched_count: int


def generate_simplex_weight_grid(
    n_inputs: int,
    *,
    step: float = 0.25,
    include_singletons: bool = True,
) -> list[tuple[float, ...]]:
    """Return non-negative weight vectors that sum to one on a regular grid."""

    if n_inputs <= 0:
        raise ValueError("n_inputs must be positive")
    if not 0.0 < float(step) <= 1.0:
        raise ValueError("step must be in (0, 1]")
    units = int(round(1.0 / float(step)))
    if not np.isclose(units * float(step), 1.0):
        raise ValueError("step must divide 1.0 exactly, e.g. 0.5, 0.25, 0.1")
    rows: list[tuple[float, ...]] = []
    for values in product(range(units + 1), repeat=n_inputs):
        if sum(values) != units:
            continue
        if not include_singletons and sum(value > 0 for value in values) == 1:
            continue
        rows.append(tuple(float(value) / float(units) for value in values))
    return sorted(set(rows), reverse=True)


def evaluate_submission_ensemble_weight_grid(
    submission_inputs: Iterable[SubmissionInput],
    *,
    truth: pd.DataFrame,
    weight_grid: Iterable[tuple[float, ...]],
    class_policies: Iterable[str] = ("weighted-vote",),
    timestamp_tolerance_s: float = 1.0e-6,
) -> tuple[pd.DataFrame, pd.DataFrame, tuple[float, ...], str]:
    """Score a weight/policy grid over official Track 5 submissions."""

    inputs = tuple(submission_inputs)
    if not inputs:
        raise ValueError("at least one submission input is required")
    policies = _normalize_class_policies(class_policies)
    summary_records: list[dict[str, Any]] = []
    sequence_records: list[dict[str, Any]] = []
    best_row: SubmissionGridRow | None = None
    for class_policy in policies:
        for weights in weight_grid:
            if len(weights) != len(inputs):
                raise ValueError(
                    f"weight vector length {len(weights)} does not match inputs {len(inputs)}"
                )
            weighted_inputs = tuple(
                SubmissionInput(label=item.label, path=item.path, weight=float(weight))
                for item, weight in zip(inputs, weights, strict=True)
            )
            estimates, diagnostics = ensemble_track5_submissions(
                weighted_inputs,
                class_policy=class_policy,
            )
            evaluation = evaluate_mmaud_results(
                _local_results_frame(estimates),
                truth,
                metric_protocol="public-track5",
                timestamp_tolerance_s=float(timestamp_tolerance_s),
            )
            row = _grid_row(weights, class_policy, evaluation)
            summary_records.append(_summary_record(inputs, row, diagnostics=diagnostics))
            sequence_records.extend(_sequence_records(inputs, row.weights, evaluation, row))
            if best_row is None or _row_sort_key(row) < _row_sort_key(best_row):
                best_row = row
    summary = pd.DataFrame.from_records(summary_records).sort_values(
        ["pose_mse", "p95_error_m", "max_error_m"],
        na_position="last",
    )
    by_sequence = pd.DataFrame.from_records(sequence_records)
    if best_row is None:
        raise ValueError("weight grid produced no rows")
    return summary.reset_index(drop=True), by_sequence, best_row.weights, best_row.class_policy


def write_submission_ensemble_weight_grid_outputs(
    *,
    submission_inputs: Iterable[SubmissionInput],
    truth: pd.DataFrame,
    weight_grid: Iterable[tuple[float, ...]],
    output_dir: Path,
    template: pd.DataFrame | None = None,
    class_policies: Iterable[str] = ("weighted-vote",),
    timestamp_tolerance_s: float = 1.0e-6,
) -> dict[str, Path]:
    """Score a grid and write the best official Track 5 ensemble artifact."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    inputs = tuple(submission_inputs)
    summary, by_sequence, best_weights, best_policy = evaluate_submission_ensemble_weight_grid(
        inputs,
        truth=truth,
        weight_grid=weight_grid,
        class_policies=class_policies,
        timestamp_tolerance_s=timestamp_tolerance_s,
    )
    summary_csv = output / GRID_SUMMARY_CSV
    by_sequence_csv = output / GRID_BY_SEQUENCE_CSV
    manifest_json = output / GRID_MANIFEST_JSON
    summary.to_csv(summary_csv, index=False)
    by_sequence.to_csv(by_sequence_csv, index=False)
    best_inputs = tuple(
        SubmissionInput(label=item.label, path=item.path, weight=float(weight))
        for item, weight in zip(inputs, best_weights, strict=True)
    )
    best_estimates, best_diagnostics = ensemble_track5_submissions(
        best_inputs,
        class_policy=best_policy,
    )
    best_paths = write_track5_submission_ensemble_outputs(
        estimates=best_estimates,
        diagnostics=best_diagnostics,
        output_dir=output / BEST_OUTPUT_DIR,
        template=template,
        manifest={
            "source": "submission_ensemble_weight_grid",
            "best_weights": list(best_weights),
            "best_class_policy": best_policy,
        },
    )
    best_summary = summary.iloc[0].to_dict() if not summary.empty else {}
    manifest = {
        "schema": "raft-uav-mmuad-track5-submission-ensemble-weight-grid-v1",
        "submission_inputs": [
            {"label": item.label, "path": str(item.path)} for item in inputs
        ],
        "class_policies": list(_normalize_class_policies(class_policies)),
        "timestamp_tolerance_s": float(timestamp_tolerance_s),
        "grid_row_count": int(len(summary)),
        "best_weights": list(best_weights),
        "best_class_policy": best_policy,
        "best": best_summary,
        "paths": {
            "summary_csv": str(summary_csv),
            "by_sequence_csv": str(by_sequence_csv),
            "best_output_dir": str(output / BEST_OUTPUT_DIR),
            **{f"best_{name}": str(path) for name, path in best_paths.items()},
        },
    }
    manifest_json.write_text(json.dumps(_jsonable(manifest), indent=2), encoding="utf-8")
    return {
        "summary_csv": summary_csv,
        "by_sequence_csv": by_sequence_csv,
        "manifest_json": manifest_json,
        **{f"best_{name}": path for name, path in best_paths.items()},
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="raft-uav-mmuad-track5-submission-ensemble-grid",
        description="score a weight grid over official MMUAD/UG2+ Track 5 submissions",
    )
    parser.add_argument(
        "--submission",
        action="append",
        default=[],
        metavar="LABEL=PATH",
        help="official CSV/ZIP submission; weight in the spec is ignored by the grid",
    )
    parser.add_argument("--truth-csv", type=Path, required=True)
    parser.add_argument("--template", type=Path, help="optional official template for best ZIP validation")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--weight-step", type=float, default=0.25)
    parser.add_argument("--exclude-singletons", action="store_true")
    parser.add_argument(
        "--class-policy",
        action="append",
        choices=("weighted-vote", "first"),
        default=[],
        help="classification ensemble policy to include; may be repeated",
    )
    parser.add_argument("--timestamp-tolerance-s", type=float, default=1.0e-6)
    parser.add_argument("--require-leaderboard-ready", action="store_true")
    args = parser.parse_args(argv)

    if not args.submission:
        parser.error("provide at least one --submission")
    inputs = tuple(_reset_weight(parse_submission_input(value)) for value in args.submission)
    weights = generate_simplex_weight_grid(
        len(inputs),
        step=float(args.weight_step),
        include_singletons=not bool(args.exclude_singletons),
    )
    truth = load_evaluation_truth_file(args.truth_csv).rows
    template = None if args.template is None else load_official_track5_template_file(args.template)
    class_policies = tuple(args.class_policy) or ("weighted-vote",)
    paths = write_submission_ensemble_weight_grid_outputs(
        submission_inputs=inputs,
        truth=truth,
        weight_grid=weights,
        output_dir=args.output_dir,
        template=template,
        class_policies=class_policies,
        timestamp_tolerance_s=float(args.timestamp_tolerance_s),
    )
    manifest = json.loads(paths["manifest_json"].read_text(encoding="utf-8"))
    best_validation = manifest.get("paths", {}).get("best_validation_json")
    print("mmuad_track5_submission_ensemble_grid=ok")
    print(f"grid_rows={manifest['grid_row_count']}")
    print(f"best_weights={manifest['best_weights']}")
    print(f"best_class_policy={manifest['best_class_policy']}")
    best = manifest.get("best", {})
    if best.get("pose_mse") is not None:
        print(f"pose_mse={best['pose_mse']}")
    if best.get("class_accuracy") is not None:
        print(f"class_accuracy={best['class_accuracy']}")
    for name, path in paths.items():
        print(f"{name}={path}")
    if args.require_leaderboard_ready and best_validation:
        validation = json.loads(Path(best_validation).read_text(encoding="utf-8"))
        if not validation.get("leaderboard_ready", False):
            reasons = ", ".join(validation.get("leaderboard_blocking_reasons", [])) or "unknown"
            raise SystemExit(f"best ensemble is not leaderboard-ready: {reasons}")
    return 0


def _reset_weight(item: SubmissionInput) -> SubmissionInput:
    return SubmissionInput(label=item.label, path=item.path, weight=1.0)


def _local_results_frame(estimates: pd.DataFrame) -> pd.DataFrame:
    rows = pd.DataFrame(estimates).copy()
    return pd.DataFrame(
        {
            "sequence_id": rows["sequence_id"].astype(str),
            "timestamp": pd.to_numeric(rows["time_s"], errors="coerce"),
            "x": pd.to_numeric(rows["state_x_m"], errors="coerce"),
            "y": pd.to_numeric(rows["state_y_m"], errors="coerce"),
            "z": pd.to_numeric(rows["state_z_m"], errors="coerce"),
            "uav_type": rows["Classification"].astype(str),
            "score": 1.0,
        }
    )


def _grid_row(
    weights: tuple[float, ...],
    class_policy: str,
    evaluation: dict[str, Any],
) -> SubmissionGridRow:
    summary = evaluation.get("summary", {})
    pooled = summary.get("pooled", {})
    return SubmissionGridRow(
        weights=tuple(float(weight) for weight in weights),
        class_policy=str(class_policy),
        pose_mse=_safe_float(pooled.get("pose_mse_loss_m2")),
        rmse_m=_safe_float(pooled.get("rmse_3d_m")),
        mean_error_m=_safe_float(pooled.get("mean_3d_m")),
        p95_error_m=_safe_float(pooled.get("p95_3d_m")),
        max_error_m=_safe_float(pooled.get("max_3d_m")),
        class_accuracy=_optional_float(pooled.get("classification_accuracy")),
        matched_count=int(summary.get("matched_count", pooled.get("count", 0)) or 0),
    )


def _summary_record(
    inputs: tuple[SubmissionInput, ...],
    row: SubmissionGridRow,
    *,
    diagnostics: pd.DataFrame,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "class_policy": row.class_policy,
        "pose_mse": row.pose_mse,
        "rmse_3d_m": row.rmse_m,
        "mean_3d_m": row.mean_error_m,
        "p95_3d_m": row.p95_error_m,
        "max_3d_m": row.max_error_m,
        "class_accuracy": row.class_accuracy,
        "matched_count": row.matched_count,
        "mean_position_spread_m": _optional_float(diagnostics.get("position_spread_m", pd.Series(dtype=float)).mean()),
    }
    for item, weight in zip(inputs, row.weights, strict=True):
        record[f"weight_{item.label}"] = float(weight)
    return record


def _sequence_records(
    inputs: tuple[SubmissionInput, ...],
    weights: tuple[float, ...],
    evaluation: dict[str, Any],
    row: SubmissionGridRow,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for sequence_id, seq_summary in evaluation.get("summary", {}).get("sequences", {}).items():
        record: dict[str, Any] = {
            "sequence_id": str(sequence_id),
            "class_policy": row.class_policy,
            "pose_mse": _safe_float(seq_summary.get("pose_mse_loss_m2")),
            "rmse_3d_m": _safe_float(seq_summary.get("rmse_3d_m")),
            "p95_3d_m": _safe_float(seq_summary.get("p95_3d_m")),
            "max_3d_m": _safe_float(seq_summary.get("max_3d_m")),
            "class_accuracy": _optional_float(seq_summary.get("classification_accuracy")),
            "matched_count": int(seq_summary.get("matched_count", seq_summary.get("count", 0)) or 0),
        }
        for item, weight in zip(inputs, weights, strict=True):
            record[f"weight_{item.label}"] = float(weight)
        records.append(record)
    return records


def _normalize_class_policies(values: Iterable[str]) -> tuple[str, ...]:
    policies = tuple(dict.fromkeys(str(value) for value in values)) or ("weighted-vote",)
    allowed = {"weighted-vote", "first"}
    invalid = sorted(set(policies).difference(allowed))
    if invalid:
        raise ValueError(f"unsupported class policies: {invalid}")
    return policies


def _row_sort_key(row: SubmissionGridRow) -> tuple[float, float, float]:
    return (row.pose_mse, row.p95_error_m, row.max_error_m)


def _safe_float(value: object) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("inf")
    return out if np.isfinite(out) else float("inf")


def _optional_float(value: object) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if np.isfinite(out) else None


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
