"""Train-fold search for Track 5 spread-guard ensemble settings.

The spread-guard ensemble is inference-safe, but its disagreement threshold and
fallback policy should be selected on labeled train folds rather than tuned on a
leaderboard split.  This helper evaluates a compact grid of thresholds and
fallback policies, writes the best configuration, and can optionally materialize
that best guarded submission for the labeled split.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from raft_uav.mmuad.evaluator import load_evaluation_truth_file
from raft_uav.mmuad.submission import load_official_track5_template_file, load_sequence_class_map
from raft_uav.mmuad.track5_estimate_ensemble import EstimateInput, parse_estimate_spec
from raft_uav.mmuad.track5_estimate_ensemble_spread_guard import FALLBACK_POLICIES
from raft_uav.mmuad.track5_estimate_ensemble_spread_guard import (
    _read_estimate_csv,
    build_spread_guarded_estimate_ensemble,
)
from raft_uav.mmuad.track5_estimate_ensemble_spread_guard import write_spread_guard_outputs

SPREAD_GUARD_GRID_CSV = "mmuad_track5_spread_guard_search_grid.csv"
SPREAD_GUARD_BEST_JSON = "mmuad_track5_spread_guard_best_config.json"
BEST_OUTPUT_DIR = "best_spread_guard_submission"


def search_track5_spread_guard_settings(
    estimate_inputs: Iterable[EstimateInput],
    *,
    template: pd.DataFrame,
    truth: pd.DataFrame,
    spread_thresholds_m: Iterable[float],
    fallback_policies: Iterable[str] = ("max-weight",),
    fallback_labels: Iterable[str] = (),
    max_nearest_time_delta_s: float | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Evaluate spread-guard settings and return grid rows plus best config."""

    input_list = tuple(estimate_inputs)
    if not input_list:
        raise ValueError("at least one estimate input is required")
    loaded = [(item.label, _read_estimate_csv(item.path), float(item.weight)) for item in input_list]
    thresholds = tuple(float(value) for value in spread_thresholds_m)
    if not thresholds:
        raise ValueError("at least one spread threshold is required")
    truth_rows = _normalize_truth_for_exact_template(truth)
    records: list[dict[str, Any]] = []
    for threshold in thresholds:
        _validate_threshold(threshold)
        for policy in fallback_policies:
            if policy not in FALLBACK_POLICIES:
                raise ValueError(f"fallback_policy must be one of: {', '.join(FALLBACK_POLICIES)}")
            labels = (None,) if policy != "label" else tuple(str(label) for label in fallback_labels)
            if not labels:
                raise ValueError("fallback_labels are required when fallback_policy='label'")
            for fallback_label in labels:
                estimates, diagnostics = build_spread_guarded_estimate_ensemble(
                    loaded,
                    template,
                    spread_threshold_m=threshold,
                    fallback_policy=str(policy),
                    fallback_label=fallback_label,
                    max_nearest_time_delta_s=max_nearest_time_delta_s,
                )
                metrics = _score_template_estimates(estimates, truth_rows)
                records.append(
                    {
                        "spread_threshold_m": threshold,
                        "fallback_policy": str(policy),
                        "fallback_label": "" if fallback_label is None else str(fallback_label),
                        "guard_applied_rows": int(estimates["spread_guard_applied"].astype(bool).sum())
                        if "spread_guard_applied" in estimates
                        else 0,
                        "guard_applied_fraction": float(
                            estimates["spread_guard_applied"].astype(bool).mean()
                        )
                        if "spread_guard_applied" in estimates and len(estimates)
                        else 0.0,
                        "mean_position_spread_m": _safe_mean(
                            diagnostics.get("position_spread_m", pd.Series(dtype=float))
                        ),
                        "p95_position_spread_m": _safe_percentile(
                            diagnostics.get("position_spread_m", pd.Series(dtype=float)),
                            95,
                        ),
                        **metrics,
                    }
                )
    grid = pd.DataFrame.from_records(records)
    if grid.empty:
        raise ValueError("spread-guard grid produced no rows")
    best_row = grid.sort_values(["pose_mse_m2", "pose_p95_m", "pose_max_m"], na_position="last").iloc[0]
    best = {
        "schema": "raft-uav-mmuad-track5-spread-guard-search-v1",
        "spread_threshold_m": _jsonable(best_row["spread_threshold_m"]),
        "fallback_policy": str(best_row["fallback_policy"]),
        "fallback_label": str(best_row["fallback_label"]),
        "max_nearest_time_delta_s": max_nearest_time_delta_s,
        "metrics": {
            key: _jsonable(best_row[key])
            for key in (
                "pose_mse_m2",
                "pose_rmse_m",
                "pose_mean_m",
                "pose_p95_m",
                "pose_max_m",
                "matched_rows",
                "guard_applied_rows",
                "guard_applied_fraction",
            )
            if key in best_row.index
        },
        "estimate_inputs": [asdict(item) | {"path": str(item.path)} for item in input_list],
    }
    return grid, _jsonable(best)


def write_spread_guard_search_outputs(
    *,
    estimate_inputs: Iterable[EstimateInput],
    template: pd.DataFrame,
    truth: pd.DataFrame,
    output_dir: Path,
    spread_thresholds_m: Iterable[float],
    fallback_policies: Iterable[str] = ("max-weight",),
    fallback_labels: Iterable[str] = (),
    max_nearest_time_delta_s: float | None = None,
    write_best_submission: bool = False,
    class_map: dict[str, str] | None = None,
    default_classification: int | str = 0,
) -> dict[str, Path]:
    """Run the grid and write search artifacts and optional best submission."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    input_list = list(estimate_inputs)
    grid, best = search_track5_spread_guard_settings(
        input_list,
        template=template,
        truth=truth,
        spread_thresholds_m=spread_thresholds_m,
        fallback_policies=fallback_policies,
        fallback_labels=fallback_labels,
        max_nearest_time_delta_s=max_nearest_time_delta_s,
    )
    paths = {
        "grid_csv": output / SPREAD_GUARD_GRID_CSV,
        "best_config_json": output / SPREAD_GUARD_BEST_JSON,
    }
    grid.to_csv(paths["grid_csv"], index=False)
    paths["best_config_json"].write_text(json.dumps(_jsonable(best), indent=2), encoding="utf-8")
    if write_best_submission:
        fallback_label = str(best["fallback_label"]) or None
        best_paths = write_spread_guard_outputs(
            estimate_inputs=input_list,
            template=template,
            output_dir=output / BEST_OUTPUT_DIR,
            spread_threshold_m=float(best["spread_threshold_m"]),
            fallback_policy=str(best["fallback_policy"]),
            fallback_label=fallback_label,
            max_nearest_time_delta_s=max_nearest_time_delta_s,
            class_map=class_map or {},
            default_classification=default_classification,
        )
        paths.update({f"best_{name}": path for name, path in best_paths.items()})
    return paths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m raft_uav.mmuad.track5_spread_guard_search",
        description="select Track 5 spread-guard ensemble settings on a labeled split",
    )
    parser.add_argument("--estimate-csv", action="append", default=[], metavar="LABEL=PATH[@WEIGHT]")
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--truth-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--spread-threshold-m",
        action="append",
        default=[],
        help="spread threshold in meters; may be repeated or comma-separated",
    )
    parser.add_argument(
        "--fallback-policy",
        action="append",
        default=[],
        choices=FALLBACK_POLICIES,
        help="fallback policy to test; may be repeated",
    )
    parser.add_argument(
        "--fallback-label",
        action="append",
        default=[],
        help="fallback label to test when fallback-policy=label",
    )
    parser.add_argument("--max-nearest-time-delta-s", type=float)
    parser.add_argument("--write-best-submission", action="store_true")
    parser.add_argument("--class-map", type=Path)
    parser.add_argument("--default-classification", default="0")
    args = parser.parse_args(argv)

    if not args.estimate_csv:
        parser.error("provide at least one --estimate-csv LABEL=PATH[@WEIGHT]")
    thresholds = _parse_float_list(args.spread_threshold_m)
    if not thresholds:
        parser.error("provide at least one --spread-threshold-m")
    policies = tuple(args.fallback_policy) or ("max-weight",)
    estimates = [parse_estimate_spec(value) for value in args.estimate_csv]
    template = load_official_track5_template_file(args.template)
    truth = load_evaluation_truth_file(args.truth_csv).rows
    class_map = load_sequence_class_map(args.class_map) if args.class_map is not None else {}
    paths = write_spread_guard_search_outputs(
        estimate_inputs=estimates,
        template=template,
        truth=truth,
        output_dir=args.output_dir,
        spread_thresholds_m=thresholds,
        fallback_policies=policies,
        fallback_labels=tuple(args.fallback_label),
        max_nearest_time_delta_s=args.max_nearest_time_delta_s,
        write_best_submission=args.write_best_submission,
        class_map=class_map,
        default_classification=args.default_classification,
    )
    print("mmuad_track5_spread_guard_search=ok")
    for name, path in paths.items():
        print(f"{name}={path}")
    return 0


def _parse_float_list(values: Iterable[str]) -> tuple[float, ...]:
    parsed: list[float] = []
    for value in values:
        for part in str(value).replace(";", ",").split(","):
            text = part.strip()
            if text:
                parsed.append(float(text))
    return tuple(parsed)


def _validate_threshold(value: float) -> None:
    if not np.isfinite(value) or value < 0.0:
        raise ValueError(f"spread threshold must be finite and non-negative: {value}")


def _normalize_truth_for_exact_template(truth: pd.DataFrame) -> pd.DataFrame:
    rows = pd.DataFrame(truth).copy()
    if rows.empty:
        return pd.DataFrame(columns=["sequence_id", "time_s", "x_m", "y_m", "z_m"])
    rows["sequence_id"] = rows["sequence_id"].astype(str)
    for column in ("time_s", "x_m", "y_m", "z_m"):
        rows[column] = pd.to_numeric(rows[column], errors="coerce")
    rows["_time_key"] = _time_key(rows["time_s"])
    finite = np.isfinite(rows[["time_s", "x_m", "y_m", "z_m"]].to_numpy(float)).all(axis=1)
    return rows.loc[finite, ["sequence_id", "_time_key", "x_m", "y_m", "z_m"]].copy()


def _score_template_estimates(estimates: pd.DataFrame, truth: pd.DataFrame) -> dict[str, Any]:
    rows = pd.DataFrame(estimates).copy()
    if rows.empty or truth.empty:
        return _empty_metrics()
    rows["sequence_id"] = rows["sequence_id"].astype(str)
    rows["_time_key"] = _time_key(pd.to_numeric(rows["time_s"], errors="coerce"))
    merged = rows.merge(truth, on=["sequence_id", "_time_key"], how="inner", suffixes=("", "_truth"))
    if merged.empty:
        return _empty_metrics()
    estimated_xyz = merged[["state_x_m", "state_y_m", "state_z_m"]].to_numpy(float)
    truth_xyz = merged[["x_m", "y_m", "z_m"]].to_numpy(float)
    finite = np.isfinite(estimated_xyz).all(axis=1) & np.isfinite(truth_xyz).all(axis=1)
    if not finite.any():
        return _empty_metrics()
    errors = np.linalg.norm(estimated_xyz[finite] - truth_xyz[finite], axis=1)
    squared = errors**2
    return {
        "matched_rows": int(len(errors)),
        "pose_mse_m2": float(np.mean(squared)),
        "pose_rmse_m": float(np.sqrt(np.mean(squared))),
        "pose_mean_m": float(np.mean(errors)),
        "pose_p95_m": float(np.percentile(errors, 95)),
        "pose_max_m": float(np.max(errors)),
    }


def _empty_metrics() -> dict[str, Any]:
    return {
        "matched_rows": 0,
        "pose_mse_m2": np.nan,
        "pose_rmse_m": np.nan,
        "pose_mean_m": np.nan,
        "pose_p95_m": np.nan,
        "pose_max_m": np.nan,
    }


def _time_key(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    rounded = np.round(numeric.to_numpy(float), 9)
    return pd.Series(
        [f"{value:.9f}" if np.isfinite(value) else "" for value in rounded],
        index=values.index,
        dtype="string",
    )


def _safe_mean(values: pd.Series) -> float | None:
    numeric = pd.to_numeric(values, errors="coerce")
    numeric = numeric[np.isfinite(numeric.to_numpy(float))]
    if numeric.empty:
        return None
    return float(np.mean(numeric.to_numpy(float)))


def _safe_percentile(values: pd.Series, percentile: float) -> float | None:
    numeric = pd.to_numeric(values, errors="coerce")
    numeric = numeric[np.isfinite(numeric.to_numpy(float))]
    if numeric.empty:
        return None
    return float(np.percentile(numeric.to_numpy(float), percentile))


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
