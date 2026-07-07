"""Train-time weight search for MMUAD Track 5 estimate ensembles.

The Track 5 estimate ensemble command intentionally requires explicit weights so
hidden-test inference stays non-oracle.  This helper provides the complementary
train/validation diagnostic: sweep a small convex weight grid against an
available truth file, write the best weight config, and optionally materialize an
official-style ZIP for the best row.

Use this on train folds or public-validation diagnostics only.  Do not select
hidden-test ensemble weights with hidden-test feedback.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from raft_uav.mmuad.evaluator import load_evaluation_truth_file
from raft_uav.mmuad.submission import load_official_track5_template_file, load_sequence_class_map
from raft_uav.mmuad.track5_estimate_ensemble import EstimateInput
from raft_uav.mmuad.track5_estimate_ensemble import build_track5_estimate_ensemble
from raft_uav.mmuad.track5_estimate_ensemble import parse_estimate_spec
from raft_uav.mmuad.track5_estimate_ensemble import write_track5_estimate_ensemble_outputs

WEIGHT_GRID_CSV = "mmuad_track5_estimate_weight_search.csv"
BEST_CONFIG_JSON = "mmuad_track5_estimate_weight_search_best_config.json"
BEST_ESTIMATES_CSV = "mmuad_track5_estimate_weight_search_best_estimates.csv"
BEST_DIAGNOSTICS_CSV = "mmuad_track5_estimate_weight_search_best_diagnostics.csv"
PROVENANCE_JSON = "mmuad_track5_estimate_weight_search_provenance.json"


def generate_simplex_weight_grid(count: int, *, step: float = 0.25) -> list[tuple[float, ...]]:
    """Generate non-negative weights summing to one on a regular simplex grid."""

    if count <= 0:
        raise ValueError("count must be positive")
    step = float(step)
    if not np.isfinite(step) or step <= 0.0 or step > 1.0:
        raise ValueError("step must be in (0, 1]")
    denominator_float = 1.0 / step
    denominator = int(round(denominator_float))
    if not np.isclose(denominator_float, denominator):
        raise ValueError("step must evenly divide 1.0, e.g. 0.5, 0.25, 0.1")
    allocations: list[tuple[int, ...]] = []
    _integer_simplex_allocations(count, denominator, prefix=(), out=allocations)
    return [tuple(value / denominator for value in allocation) for allocation in allocations]


def search_estimate_ensemble_weights(
    estimate_inputs: Iterable[EstimateInput],
    *,
    truth: pd.DataFrame,
    template: pd.DataFrame | None = None,
    class_map: dict[str, str] | None = None,
    default_classification: int | str = 0,
    step: float = 0.25,
    max_nearest_time_delta_s: float | None = None,
    aggregation_policy: str = "weighted-mean",
    trim_fraction: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, float]]:
    """Sweep convex ensemble weights and return summary plus best artifacts."""

    inputs = tuple(estimate_inputs)
    if not inputs:
        raise ValueError("at least one estimate input is required")
    template_rows = _template_rows(template if template is not None else truth)
    truth_rows = _truth_rows(truth)
    loaded = [(item.label, pd.read_csv(item.path), item.weight) for item in inputs]
    class_map = class_map or {}
    records: list[dict[str, Any]] = []
    best_key: tuple[float, float, float] | None = None
    best_weights: dict[str, float] = {}
    best_estimates = pd.DataFrame()
    best_diagnostics = pd.DataFrame()
    for weights in generate_simplex_weight_grid(len(inputs), step=step):
        weighted_inputs = [
            (label, estimates, float(weight))
            for (label, estimates, _), weight in zip(loaded, weights, strict=True)
        ]
        estimates, diagnostics = build_track5_estimate_ensemble(
            weighted_inputs,
            template_rows,
            max_nearest_time_delta_s=max_nearest_time_delta_s,
            aggregation_policy=aggregation_policy,
            trim_fraction=trim_fraction,
        )
        metrics = _score_estimates_against_truth(
            estimates,
            truth_rows,
            class_map=class_map,
            default_classification=default_classification,
        )
        weight_map = {item.label: float(weight) for item, weight in zip(inputs, weights, strict=True)}
        record = {
            "weights_json": json.dumps(weight_map, sort_keys=True),
            "active_weight_count": int(sum(weight > 0.0 for weight in weights)),
            **{f"weight_{item.label}": float(weight) for item, weight in zip(inputs, weights, strict=True)},
            **metrics,
        }
        records.append(record)
        key = (
            float(metrics.get("pose_mse", np.inf)),
            float(metrics.get("p95_3d_m", np.inf)),
            float(metrics.get("max_3d_m", np.inf)),
        )
        if best_key is None or key < best_key:
            best_key = key
            best_weights = weight_map
            best_estimates = estimates
            best_diagnostics = diagnostics
    summary = pd.DataFrame.from_records(records).sort_values(
        ["pose_mse", "p95_3d_m", "max_3d_m"],
        na_position="last",
    )
    return summary.reset_index(drop=True), best_estimates, best_diagnostics, best_weights


def write_estimate_weight_search_outputs(
    *,
    estimate_inputs: Iterable[EstimateInput],
    truth: pd.DataFrame,
    output_dir: Path,
    template: pd.DataFrame | None = None,
    class_map: dict[str, str] | None = None,
    default_classification: int | str = 0,
    step: float = 0.25,
    max_nearest_time_delta_s: float | None = None,
    aggregation_policy: str = "weighted-mean",
    trim_fraction: float = 0.2,
    write_best_submission: bool = False,
) -> dict[str, Path]:
    """Run the grid and write search, config, and optional submission artifacts."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    inputs = tuple(estimate_inputs)
    summary, best_estimates, best_diagnostics, best_weights = search_estimate_ensemble_weights(
        inputs,
        truth=truth,
        template=template,
        class_map=class_map,
        default_classification=default_classification,
        step=step,
        max_nearest_time_delta_s=max_nearest_time_delta_s,
        aggregation_policy=aggregation_policy,
        trim_fraction=trim_fraction,
    )
    paths = {
        "summary_csv": output / WEIGHT_GRID_CSV,
        "best_config_json": output / BEST_CONFIG_JSON,
        "best_estimates_csv": output / BEST_ESTIMATES_CSV,
        "best_diagnostics_csv": output / BEST_DIAGNOSTICS_CSV,
        "provenance_json": output / PROVENANCE_JSON,
    }
    summary.to_csv(paths["summary_csv"], index=False)
    best_estimates.to_csv(paths["best_estimates_csv"], index=False)
    best_diagnostics.to_csv(paths["best_diagnostics_csv"], index=False)
    best_payload = {
        "schema": "raft-uav-mmuad-track5-estimate-weight-search-v1",
        "weights": best_weights,
        "step": float(step),
        "aggregation_policy": aggregation_policy,
        "trim_fraction": float(trim_fraction),
        "best_metrics": summary.iloc[0].to_dict() if not summary.empty else {},
    }
    paths["best_config_json"].write_text(json.dumps(_jsonable(best_payload), indent=2), encoding="utf-8")
    provenance = {
        "estimate_inputs": [
            {"label": item.label, "path": str(item.path), "initial_weight": float(item.weight)}
            for item in inputs
        ],
        "step": float(step),
        "max_nearest_time_delta_s": max_nearest_time_delta_s,
        "aggregation_policy": aggregation_policy,
        "trim_fraction": float(trim_fraction),
        "grid_rows": int(len(summary)),
        "paths": {name: str(path) for name, path in paths.items()},
    }
    paths["provenance_json"].write_text(json.dumps(_jsonable(provenance), indent=2), encoding="utf-8")
    if write_best_submission:
        weighted_inputs = [
            EstimateInput(label=item.label, path=item.path, weight=best_weights[item.label])
            for item in inputs
        ]
        submission_paths = write_track5_estimate_ensemble_outputs(
            estimate_inputs=weighted_inputs,
            template=_template_rows(template if template is not None else truth),
            output_dir=output / "best_submission",
            class_map=class_map,
            default_classification=default_classification,
            max_nearest_time_delta_s=max_nearest_time_delta_s,
            aggregation_policy=aggregation_policy,
            trim_fraction=trim_fraction,
        )
        paths.update({f"best_submission_{name}": path for name, path in submission_paths.items()})
    return paths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="raft-uav-mmuad-track5-estimate-weight-search",
        description="sweep train-selected weights for MMUAD Track 5 estimate ensembles",
    )
    parser.add_argument("--estimate-csv", action="append", default=[], metavar="LABEL=PATH")
    parser.add_argument("--truth-file", type=Path, required=True)
    parser.add_argument("--template", type=Path)
    parser.add_argument("--class-map", type=Path)
    parser.add_argument("--default-classification", default="0")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--step", type=float, default=0.25)
    parser.add_argument("--max-nearest-time-delta-s", type=float)
    parser.add_argument(
        "--aggregation-policy",
        choices=("weighted-mean", "weighted-median", "trimmed-mean"),
        default="weighted-mean",
    )
    parser.add_argument("--trim-fraction", type=float, default=0.2)
    parser.add_argument("--write-best-submission", action="store_true")
    args = parser.parse_args(argv)

    if not args.estimate_csv:
        parser.error("provide at least one --estimate-csv LABEL=PATH")
    inputs = [parse_estimate_spec(value) for value in args.estimate_csv]
    truth = load_evaluation_truth_file(args.truth_file).rows
    template = load_official_track5_template_file(args.template) if args.template is not None else truth
    class_map = load_sequence_class_map(args.class_map) if args.class_map is not None else {}
    paths = write_estimate_weight_search_outputs(
        estimate_inputs=inputs,
        truth=truth,
        template=template,
        class_map=class_map,
        default_classification=args.default_classification,
        output_dir=args.output_dir,
        step=float(args.step),
        max_nearest_time_delta_s=args.max_nearest_time_delta_s,
        aggregation_policy=args.aggregation_policy,
        trim_fraction=float(args.trim_fraction),
        write_best_submission=bool(args.write_best_submission),
    )
    print("mmuad_track5_estimate_weight_search=ok")
    for name, path in paths.items():
        print(f"{name}={path}")
    return 0


def _integer_simplex_allocations(
    count: int,
    total: int,
    *,
    prefix: tuple[int, ...],
    out: list[tuple[int, ...]],
) -> None:
    if count == 1:
        out.append((*prefix, total))
        return
    for value in range(total + 1):
        _integer_simplex_allocations(count - 1, total - value, prefix=(*prefix, value), out=out)


def _score_estimates_against_truth(
    estimates: pd.DataFrame,
    truth: pd.DataFrame,
    *,
    class_map: dict[str, str],
    default_classification: int | str,
) -> dict[str, Any]:
    rows = estimates.merge(
        truth[["sequence_id", "time_s", "x_m", "y_m", "z_m"] + _truth_class_columns(truth)],
        on=["sequence_id", "time_s"],
        how="inner",
        suffixes=("", "_truth"),
    )
    if rows.empty:
        return {
            "matched_rows": 0,
            "pose_mse": np.nan,
            "rmse_3d_m": np.nan,
            "mean_3d_m": np.nan,
            "p95_3d_m": np.nan,
            "max_3d_m": np.nan,
            "classification_accuracy": np.nan,
        }
    estimate_xyz = rows[["state_x_m", "state_y_m", "state_z_m"]].to_numpy(float)
    truth_xyz = rows[["x_m", "y_m", "z_m"]].to_numpy(float)
    errors = np.linalg.norm(estimate_xyz - truth_xyz, axis=1)
    truth_class = _truth_class_series(rows)
    if truth_class is not None:
        predicted_class = rows["sequence_id"].astype(str).map(
            {str(key): str(value) for key, value in class_map.items()}
        ).fillna(str(default_classification))
        class_acc = float((predicted_class.astype(str) == truth_class.astype(str)).mean())
    else:
        class_acc = np.nan
    return {
        "matched_rows": int(len(rows)),
        "pose_mse": float(np.mean(errors**2)),
        "rmse_3d_m": float(np.sqrt(np.mean(errors**2))),
        "mean_3d_m": float(np.mean(errors)),
        "p95_3d_m": float(np.percentile(errors, 95)),
        "max_3d_m": float(np.max(errors)),
        "classification_accuracy": class_acc,
    }


def _truth_rows(truth: pd.DataFrame) -> pd.DataFrame:
    rows = pd.DataFrame(truth).copy()
    rows["sequence_id"] = rows["sequence_id"].astype(str)
    for column in ("time_s", "x_m", "y_m", "z_m"):
        rows[column] = pd.to_numeric(rows[column], errors="coerce")
    return rows.sort_values(["sequence_id", "time_s"]).reset_index(drop=True)


def _template_rows(template: pd.DataFrame) -> pd.DataFrame:
    rows = pd.DataFrame(template).copy()
    if "Sequence" in rows.columns and "Timestamp" in rows.columns:
        return rows
    if "sequence_id" in rows.columns and "time_s" in rows.columns:
        out = rows.copy()
        out["Sequence"] = out["sequence_id"]
        out["Timestamp"] = out["time_s"]
        if "Position" not in out.columns:
            out["Position"] = "(0,0,0)"
        if "Classification" not in out.columns:
            out["Classification"] = 0
        return out[["Sequence", "Timestamp", "Position", "Classification"]]
    return rows


def _truth_class_columns(truth: pd.DataFrame) -> list[str]:
    return [column for column in ("class_name", "uav_type", "class_id", "Classification") if column in truth.columns]


def _truth_class_series(rows: pd.DataFrame) -> pd.Series | None:
    for column in ("class_name", "uav_type", "class_id", "Classification"):
        if column in rows.columns:
            return rows[column].astype(str)
    return None


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
