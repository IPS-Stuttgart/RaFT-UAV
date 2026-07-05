"""Leave-one-sequence-out weight search for Track 5 estimate ensembles.

The same-split estimate-ensemble weight search is useful for diagnostics, but
leaderboard-facing weights should be selected without using the held-out
sequence.  This module performs a sequence-level LOSO protocol on a labeled split:
for each sequence, choose ensemble weights on all other sequences, evaluate those
weights on the held-out sequence, and write both fold diagnostics and a final
full-train weight config for validation/test use.
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
from raft_uav.mmuad.track5_estimate_ensemble import EstimateInput, build_track5_estimate_ensemble
from raft_uav.mmuad.track5_estimate_ensemble import parse_estimate_spec
from raft_uav.mmuad.track5_estimate_ensemble import write_track5_estimate_ensemble_outputs
from raft_uav.mmuad.track5_estimate_ensemble_weight_search import search_track5_estimate_ensemble_weights

LOSO_FOLD_SUMMARY_CSV = "mmuad_track5_ensemble_loso_fold_summary.csv"
LOSO_PREDICTIONS_CSV = "mmuad_track5_ensemble_loso_predictions.csv"
LOSO_SUMMARY_JSON = "mmuad_track5_ensemble_loso_summary.json"
FULL_WEIGHTS_JSON = "mmuad_track5_ensemble_loso_full_weights.json"
FULL_OUTPUT_DIR = "full_train_weighted_ensemble"


def run_track5_estimate_ensemble_loso_weight_search(
    estimate_inputs: Iterable[EstimateInput],
    *,
    template: pd.DataFrame,
    truth: pd.DataFrame,
    weight_step: float = 0.1,
    aggregation_policy: str = "weighted-mean",
    trim_fraction: float = 0.2,
    max_nearest_time_delta_s: float | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], dict[str, Any]]:
    """Run LOSO ensemble weight selection and return diagnostics.

    Returns ``(fold_summary, loso_predictions, loso_summary, full_weight_config)``.
    ``full_weight_config`` is fit on the complete labeled split and is the config
    intended for downstream validation/test application.
    """

    inputs = tuple(estimate_inputs)
    if not inputs:
        raise ValueError("at least one estimate input is required")
    template_rows = _normalize_template_rows(template)
    truth_rows = _normalize_truth_rows(truth)
    sequences = sorted(set(truth_rows["sequence_id"].astype(str)))
    if not sequences:
        raise ValueError("truth contains no finite sequences")
    loaded = {item.label: pd.read_csv(item.path) for item in inputs}
    fold_records: list[dict[str, Any]] = []
    prediction_parts: list[pd.DataFrame] = []
    for held_out_sequence in sequences:
        train_template = template_rows.loc[template_rows["sequence_id"] != held_out_sequence]
        train_truth = truth_rows.loc[truth_rows["sequence_id"] != held_out_sequence]
        holdout_template = template_rows.loc[template_rows["sequence_id"] == held_out_sequence]
        holdout_truth = truth_rows.loc[truth_rows["sequence_id"] == held_out_sequence]
        if train_truth.empty or holdout_template.empty or holdout_truth.empty:
            continue
        _, fold_best = search_track5_estimate_ensemble_weights(
            inputs,
            template=train_template,
            truth=train_truth,
            weight_step=float(weight_step),
            aggregation_policy=aggregation_policy,
            trim_fraction=float(trim_fraction),
            max_nearest_time_delta_s=max_nearest_time_delta_s,
        )
        weight_map = {str(label): float(weight) for label, weight in fold_best["weights"].items()}
        weighted_loaded = [
            (item.label, loaded[item.label], float(weight_map[item.label])) for item in inputs
        ]
        holdout_estimates, holdout_diag = build_track5_estimate_ensemble(
            weighted_loaded,
            holdout_template,
            max_nearest_time_delta_s=max_nearest_time_delta_s,
            aggregation_policy=aggregation_policy,
            trim_fraction=float(trim_fraction),
        )
        metrics = _score_estimates(holdout_estimates, holdout_truth)
        prediction = holdout_estimates.copy()
        prediction["loso_held_out_sequence"] = held_out_sequence
        for label, weight in weight_map.items():
            prediction[f"loso_weight_{label}"] = weight
        prediction_parts.append(prediction)
        fold_record = {
            "held_out_sequence": held_out_sequence,
            "train_sequence_count": int(train_truth["sequence_id"].nunique()),
            "holdout_row_count": int(len(holdout_template)),
            "aggregation_policy": aggregation_policy,
            "trim_fraction": float(trim_fraction),
            "weight_step": float(weight_step),
            "valid_input_count_mean": _safe_mean(holdout_diag.get("valid_input_count", pd.Series(dtype=float))),
            **metrics,
        }
        for label, weight in weight_map.items():
            fold_record[f"weight_{label}"] = weight
        fold_records.append(fold_record)

    fold_summary = pd.DataFrame.from_records(fold_records)
    predictions = (
        pd.concat(prediction_parts, ignore_index=True, sort=False)
        if prediction_parts
        else pd.DataFrame(columns=["sequence_id", "time_s", "state_x_m", "state_y_m", "state_z_m"])
    )
    pooled_metrics = _score_estimates(predictions, truth_rows)
    _, full_best = search_track5_estimate_ensemble_weights(
        inputs,
        template=template_rows,
        truth=truth_rows,
        weight_step=float(weight_step),
        aggregation_policy=aggregation_policy,
        trim_fraction=float(trim_fraction),
        max_nearest_time_delta_s=max_nearest_time_delta_s,
    )
    loso_summary = {
        "schema": "raft-uav-mmuad-track5-estimate-ensemble-loso-weight-search-v1",
        "protocol": "leave-one-sequence-out",
        "sequence_count": int(len(sequences)),
        "fold_count": int(len(fold_summary)),
        "aggregation_policy": aggregation_policy,
        "trim_fraction": float(trim_fraction),
        "weight_step": float(weight_step),
        "max_nearest_time_delta_s": max_nearest_time_delta_s,
        "loso_metrics": pooled_metrics,
        "mean_fold_mse_m2": _safe_mean(fold_summary.get("pose_mse_m2", pd.Series(dtype=float))),
        "full_train_weights": full_best.get("weights", {}),
        "estimate_inputs": [asdict(item) | {"path": str(item.path)} for item in inputs],
    }
    return fold_summary, predictions, _jsonable(loso_summary), _jsonable(full_best)


def write_loso_weight_search_outputs(
    *,
    estimate_inputs: Iterable[EstimateInput],
    template: pd.DataFrame,
    truth: pd.DataFrame,
    output_dir: Path,
    weight_step: float = 0.1,
    aggregation_policy: str = "weighted-mean",
    trim_fraction: float = 0.2,
    max_nearest_time_delta_s: float | None = None,
    write_full_train_submission: bool = False,
    class_map: dict[str, str] | None = None,
    default_classification: int | str = 0,
) -> dict[str, Path]:
    """Run LOSO search and write fold, prediction, summary, and optional ZIP artifacts."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    input_list = list(estimate_inputs)
    fold_summary, predictions, loso_summary, full_weights = run_track5_estimate_ensemble_loso_weight_search(
        input_list,
        template=template,
        truth=truth,
        weight_step=float(weight_step),
        aggregation_policy=aggregation_policy,
        trim_fraction=float(trim_fraction),
        max_nearest_time_delta_s=max_nearest_time_delta_s,
    )
    paths = {
        "fold_summary_csv": output / LOSO_FOLD_SUMMARY_CSV,
        "predictions_csv": output / LOSO_PREDICTIONS_CSV,
        "summary_json": output / LOSO_SUMMARY_JSON,
        "full_weights_json": output / FULL_WEIGHTS_JSON,
    }
    fold_summary.to_csv(paths["fold_summary_csv"], index=False)
    predictions.to_csv(paths["predictions_csv"], index=False)
    paths["summary_json"].write_text(json.dumps(_jsonable(loso_summary), indent=2), encoding="utf-8")
    paths["full_weights_json"].write_text(json.dumps(_jsonable(full_weights), indent=2), encoding="utf-8")
    if write_full_train_submission:
        weight_map = {str(label): float(weight) for label, weight in full_weights["weights"].items()}
        best_inputs = [EstimateInput(item.label, item.path, weight_map[item.label]) for item in input_list]
        best_paths = write_track5_estimate_ensemble_outputs(
            estimate_inputs=best_inputs,
            template=template,
            output_dir=output / FULL_OUTPUT_DIR,
            class_map=class_map or {},
            default_classification=default_classification,
            max_nearest_time_delta_s=max_nearest_time_delta_s,
            aggregation_policy=aggregation_policy,
            trim_fraction=float(trim_fraction),
        )
        paths.update({f"full_train_{name}": path for name, path in best_paths.items()})
    return paths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="raft-uav-mmuad-track5-ensemble-loso-weight-search",
        description="select Track 5 estimate-ensemble weights with leave-one-sequence-out CV",
    )
    parser.add_argument(
        "--estimate-csv",
        action="append",
        default=[],
        metavar="LABEL=PATH",
        help="estimate trajectory to include; may be repeated; input weights are ignored",
    )
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--truth-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--weight-step", type=float, default=0.1)
    parser.add_argument("--aggregation-policy", default="weighted-mean")
    parser.add_argument("--trim-fraction", type=float, default=0.2)
    parser.add_argument("--max-nearest-time-delta-s", type=float)
    parser.add_argument("--write-full-train-submission", action="store_true")
    parser.add_argument("--class-map", type=Path)
    parser.add_argument("--default-classification", default="0")
    args = parser.parse_args(argv)

    if not args.estimate_csv:
        parser.error("provide at least one --estimate-csv")
    estimates = [parse_estimate_spec(value) for value in args.estimate_csv]
    template = load_official_track5_template_file(args.template)
    truth = load_evaluation_truth_file(args.truth_csv).rows
    class_map = load_sequence_class_map(args.class_map) if args.class_map is not None else {}
    paths = write_loso_weight_search_outputs(
        estimate_inputs=estimates,
        template=template,
        truth=truth,
        output_dir=args.output_dir,
        weight_step=float(args.weight_step),
        aggregation_policy=str(args.aggregation_policy),
        trim_fraction=float(args.trim_fraction),
        max_nearest_time_delta_s=args.max_nearest_time_delta_s,
        write_full_train_submission=bool(args.write_full_train_submission),
        class_map=class_map,
        default_classification=args.default_classification,
    )
    print("mmuad_track5_ensemble_loso_weight_search=ok")
    for name, path in paths.items():
        print(f"{name}={path}")
    return 0


def _normalize_template_rows(template: pd.DataFrame) -> pd.DataFrame:
    rows = pd.DataFrame(template).copy()
    seq_col = _first_present(rows, ("sequence_id", "Sequence", "sequence", "seq"))
    time_col = _first_present(rows, ("time_s", "Timestamp", "timestamp", "timestamp_s", "time"))
    if seq_col is None or time_col is None:
        raise ValueError("template must contain sequence and timestamp columns")
    out = pd.DataFrame(
        {
            "sequence_id": rows[seq_col].astype(str),
            "time_s": pd.to_numeric(rows[time_col], errors="coerce"),
        }
    )
    finite = np.isfinite(out["time_s"].to_numpy(float))
    return out.loc[finite].sort_values(["sequence_id", "time_s"]).reset_index(drop=True)


def _normalize_truth_rows(truth: pd.DataFrame) -> pd.DataFrame:
    rows = pd.DataFrame(truth).copy()
    if rows.empty:
        return pd.DataFrame(columns=["sequence_id", "time_s", "x_m", "y_m", "z_m"])
    rows["sequence_id"] = rows["sequence_id"].astype(str)
    for column in ("time_s", "x_m", "y_m", "z_m"):
        rows[column] = pd.to_numeric(rows[column], errors="coerce")
    finite = np.isfinite(rows[["time_s", "x_m", "y_m", "z_m"]].to_numpy(float)).all(axis=1)
    return rows.loc[finite, ["sequence_id", "time_s", "x_m", "y_m", "z_m"]].reset_index(drop=True)


def _score_estimates(estimates: pd.DataFrame, truth: pd.DataFrame) -> dict[str, Any]:
    truth_rows = _normalize_truth_rows(truth).copy()
    if estimates.empty or truth_rows.empty:
        return _empty_metrics()
    truth_rows["_time_key"] = _time_key(truth_rows["time_s"])
    rows = pd.DataFrame(estimates).copy()
    rows["sequence_id"] = rows["sequence_id"].astype(str)
    rows["_time_key"] = _time_key(pd.to_numeric(rows["time_s"], errors="coerce"))
    merged = rows.merge(truth_rows, on=["sequence_id", "_time_key"], how="inner")
    if merged.empty:
        return _empty_metrics()
    est = merged[["state_x_m", "state_y_m", "state_z_m"]].to_numpy(float)
    ref = merged[["x_m", "y_m", "z_m"]].to_numpy(float)
    finite = np.isfinite(est).all(axis=1) & np.isfinite(ref).all(axis=1)
    if not finite.any():
        return _empty_metrics()
    errors = np.linalg.norm(est[finite] - ref[finite], axis=1)
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
    return pd.to_numeric(values, errors="coerce").round(9).astype(str)


def _first_present(rows: pd.DataFrame, names: tuple[str, ...]) -> str | None:
    lower = {str(column).lower(): str(column) for column in rows.columns}
    for name in names:
        if name in rows.columns:
            return name
        found = lower.get(name.lower())
        if found is not None:
            return found
    return None


def _safe_mean(values: pd.Series) -> float | None:
    numeric = pd.to_numeric(values, errors="coerce")
    numeric = numeric[np.isfinite(numeric.to_numpy(float))]
    if numeric.empty:
        return None
    return float(np.mean(numeric.to_numpy(float)))


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
