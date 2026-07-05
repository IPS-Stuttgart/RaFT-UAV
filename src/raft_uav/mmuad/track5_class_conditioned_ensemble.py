"""Class-conditioned Track 5 estimate ensembling.

MMUAD pose pipelines can have class-specific error modes: a reservoir mixture,
calibrated branch, or tracker variant may be better for one UAV type than
another.  This module selects per-class estimate-ensemble weights on a labeled
split and applies the fixed class-conditioned weights to validation/test
estimates using a sequence-to-class map.  The apply path is inference-safe: it
uses only supplied estimates, an official timestamp template, and fixed class
labels/predictions.
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
from raft_uav.mmuad.submission import (
    load_official_track5_template_file,
    load_sequence_class_map,
    parse_official_sequence_cell,
    validate_official_track5_submission,
    write_official_mmaud_results_csv,
    write_official_ug2_codabench_zip,
)
from raft_uav.mmuad.track5_estimate_ensemble import (
    ENSEMBLE_POLICIES,
    EstimateInput,
    build_track5_estimate_ensemble,
    parse_estimate_spec,
)

CLASS_WEIGHT_GRID_CSV = "mmuad_track5_class_ensemble_weight_grid.csv"
CLASS_WEIGHTS_JSON = "mmuad_track5_class_ensemble_weights.json"
CLASS_ENSEMBLE_ESTIMATES_CSV = "mmuad_track5_class_ensemble_estimates.csv"
CLASS_ENSEMBLE_DIAGNOSTICS_CSV = "mmuad_track5_class_ensemble_diagnostics.csv"
CLASS_ENSEMBLE_MANIFEST_JSON = "mmuad_track5_class_ensemble_manifest.json"
VALIDATION_JSON = "mmuad_track5_class_ensemble_validation.json"
VALIDATION_ROWS_CSV = "mmuad_track5_class_ensemble_validation_rows.csv"
OFFICIAL_RESULTS_CSV = "mmaud_results.csv"
OFFICIAL_ZIP = "ug2_submission.zip"


def search_class_conditioned_ensemble_weights(
    estimate_inputs: Iterable[EstimateInput],
    *,
    template: pd.DataFrame,
    truth: pd.DataFrame,
    class_map: dict[str, str],
    weight_step: float = 0.1,
    aggregation_policy: str = "weighted-mean",
    trim_fraction: float = 0.2,
    max_nearest_time_delta_s: float | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Select global and per-class weights on a labeled Track 5 split."""

    inputs = tuple(estimate_inputs)
    if not inputs:
        raise ValueError("at least one estimate input is required")
    loaded = [(item.label, pd.read_csv(item.path), 1.0) for item in inputs]
    template_rows = _normalize_template_rows(template)
    truth_rows = _normalize_truth_rows(truth)
    class_map = _normalize_sequence_class_map(class_map)

    records: list[dict[str, Any]] = []
    global_grid, global_best = _search_weights_for_template(
        loaded,
        inputs=inputs,
        template=template_rows,
        truth=truth_rows,
        class_label="__global__",
        weight_step=weight_step,
        aggregation_policy=aggregation_policy,
        trim_fraction=trim_fraction,
        max_nearest_time_delta_s=max_nearest_time_delta_s,
    )
    records.extend(global_grid)
    class_weights: dict[str, dict[str, float]] = {}
    class_metrics: dict[str, dict[str, Any]] = {"__global__": global_best["metrics"]}
    for class_label in sorted(set(class_map.values())):
        sequences = {seq for seq, label in class_map.items() if str(label) == str(class_label)}
        subset = template_rows.loc[template_rows["sequence_id"].astype(str).isin(sequences)]
        if subset.empty:
            continue
        grid_rows, best = _search_weights_for_template(
            loaded,
            inputs=inputs,
            template=subset,
            truth=truth_rows,
            class_label=str(class_label),
            weight_step=weight_step,
            aggregation_policy=aggregation_policy,
            trim_fraction=trim_fraction,
            max_nearest_time_delta_s=max_nearest_time_delta_s,
        )
        records.extend(grid_rows)
        class_weights[str(class_label)] = best["weights"]
        class_metrics[str(class_label)] = best["metrics"]
    grid = pd.DataFrame.from_records(records)
    config = {
        "schema": "raft-uav-mmuad-track5-class-conditioned-ensemble-v1",
        "aggregation_policy": aggregation_policy,
        "trim_fraction": float(trim_fraction),
        "weight_step": float(weight_step),
        "global_weights": global_best["weights"],
        "class_weights": class_weights,
        "metrics": class_metrics,
        "estimate_inputs": [asdict(item) | {"path": str(item.path)} for item in inputs],
    }
    return grid, _jsonable(config)


def build_class_conditioned_estimate_ensemble(
    estimate_inputs: Iterable[EstimateInput],
    *,
    template: pd.DataFrame,
    class_map: dict[str, str],
    weight_config: dict[str, Any],
    aggregation_policy: str | None = None,
    trim_fraction: float | None = None,
    max_nearest_time_delta_s: float | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply fixed global/per-class weights and return estimates/diagnostics."""

    inputs = tuple(estimate_inputs)
    if not inputs:
        raise ValueError("at least one estimate input is required")
    loaded = {item.label: pd.read_csv(item.path) for item in inputs}
    template_rows = _normalize_template_rows(template)
    class_map = _normalize_sequence_class_map(class_map)
    global_weights = _normalized_weight_map(weight_config.get("global_weights", {}), inputs)
    class_weights_raw = weight_config.get("class_weights", {})
    if not isinstance(class_weights_raw, dict):
        raise ValueError("weight config class_weights must be an object")
    policy = aggregation_policy or str(weight_config.get("aggregation_policy", "weighted-mean"))
    trim = float(trim_fraction if trim_fraction is not None else weight_config.get("trim_fraction", 0.2))
    estimate_parts: list[pd.DataFrame] = []
    diagnostic_parts: list[pd.DataFrame] = []
    for class_label, subset in _template_groups_by_class(template_rows, class_map):
        weights = _normalized_weight_map(class_weights_raw.get(class_label, global_weights), inputs)
        weighted_inputs = [(item.label, loaded[item.label], weights[item.label]) for item in inputs]
        estimates, diagnostics = build_track5_estimate_ensemble(
            weighted_inputs,
            subset,
            max_nearest_time_delta_s=max_nearest_time_delta_s,
            aggregation_policy=policy,
            trim_fraction=trim,
        )
        estimates["class_conditioned_ensemble_class"] = str(class_label)
        estimates["class_conditioned_weight_source"] = (
            "class" if class_label in class_weights_raw else "global"
        )
        diagnostics["class_conditioned_ensemble_class"] = str(class_label)
        diagnostics["class_conditioned_weights_json"] = json.dumps(weights, sort_keys=True)
        estimate_parts.append(estimates)
        diagnostic_parts.append(diagnostics)
    estimates_all = _concat(estimate_parts)
    diagnostics_all = _concat(diagnostic_parts)
    return (
        estimates_all.sort_values(["sequence_id", "time_s"]).reset_index(drop=True),
        diagnostics_all.sort_values(["sequence_id", "time_s"]).reset_index(drop=True),
    )


def write_class_conditioned_ensemble_outputs(
    *,
    estimate_inputs: Iterable[EstimateInput],
    template: pd.DataFrame,
    class_map: dict[str, str],
    weight_config: dict[str, Any],
    output_dir: Path,
    default_classification: int | str = 0,
    max_nearest_time_delta_s: float | None = None,
) -> dict[str, Path]:
    """Write class-conditioned estimate ensemble and official Track 5 artifacts."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    estimates, diagnostics = build_class_conditioned_estimate_ensemble(
        estimate_inputs,
        template=template,
        class_map=class_map,
        weight_config=weight_config,
        max_nearest_time_delta_s=max_nearest_time_delta_s,
    )
    paths = {
        "estimates_csv": output / CLASS_ENSEMBLE_ESTIMATES_CSV,
        "diagnostics_csv": output / CLASS_ENSEMBLE_DIAGNOSTICS_CSV,
        "official_results_csv": output / OFFICIAL_RESULTS_CSV,
        "official_zip": output / OFFICIAL_ZIP,
        "validation_json": output / VALIDATION_JSON,
        "validation_rows_csv": output / VALIDATION_ROWS_CSV,
        "manifest_json": output / CLASS_ENSEMBLE_MANIFEST_JSON,
    }
    estimates.to_csv(paths["estimates_csv"], index=False)
    diagnostics.to_csv(paths["diagnostics_csv"], index=False)
    write_official_mmaud_results_csv(
        estimates,
        paths["official_results_csv"],
        classification=default_classification,
        class_map=class_map,
        invalid_row_policy="raise",
    )
    write_official_ug2_codabench_zip(
        estimates,
        paths["official_zip"],
        classification=default_classification,
        class_map=class_map,
        invalid_row_policy="raise",
    )
    validation = validate_official_track5_submission(
        paths["official_zip"],
        template=template,
        require_zip=True,
    )
    paths["validation_json"].write_text(
        json.dumps(_jsonable(validation.summary), indent=2),
        encoding="utf-8",
    )
    validation.rows.to_csv(paths["validation_rows_csv"], index=False)
    manifest = {
        "schema": "raft-uav-mmuad-track5-class-conditioned-ensemble-output-v1",
        "weight_config_schema": weight_config.get("schema"),
        "class_weights": weight_config.get("class_weights", {}),
        "global_weights": weight_config.get("global_weights", {}),
        "row_count": int(len(estimates)),
        "class_map_sequence_count": int(len(class_map)),
        "leaderboard_ready": bool(validation.summary.get("leaderboard_ready", False)),
        "codabench_upload_ready": bool(validation.summary.get("codabench_upload_ready", False)),
        "paths": {name: str(path) for name, path in paths.items() if name != "manifest_json"},
    }
    paths["manifest_json"].write_text(json.dumps(_jsonable(manifest), indent=2), encoding="utf-8")
    return paths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="raft-uav-mmuad-track5-class-ensemble",
        description="search/apply class-conditioned Track 5 estimate ensemble weights",
    )
    parser.add_argument("--estimate-csv", action="append", default=[], metavar="LABEL=PATH")
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--class-map", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--truth-csv", type=Path)
    parser.add_argument("--weights-json", type=Path)
    parser.add_argument("--weight-step", type=float, default=0.1)
    parser.add_argument("--aggregation-policy", choices=ENSEMBLE_POLICIES, default="weighted-mean")
    parser.add_argument("--trim-fraction", type=float, default=0.2)
    parser.add_argument("--max-nearest-time-delta-s", type=float)
    parser.add_argument("--write-submission", action="store_true")
    parser.add_argument("--default-classification", default="0")
    parser.add_argument("--require-leaderboard-ready", action="store_true")
    args = parser.parse_args(argv)
    should_write_submission = (
        bool(args.write_submission)
        or args.weights_json is not None
        or bool(args.require_leaderboard_ready)
    )

    if not args.estimate_csv:
        parser.error("provide at least one --estimate-csv LABEL=PATH")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    estimates = [parse_estimate_spec(value) for value in args.estimate_csv]
    template = load_official_track5_template_file(args.template)
    class_map = load_sequence_class_map(args.class_map)
    paths: dict[str, Path] = {}
    if args.weights_json is not None:
        weight_config = json.loads(args.weights_json.read_text(encoding="utf-8"))
    elif args.truth_csv is not None:
        truth = load_evaluation_truth_file(args.truth_csv).rows
        grid, weight_config = search_class_conditioned_ensemble_weights(
            estimates,
            template=template,
            truth=truth,
            class_map=class_map,
            weight_step=float(args.weight_step),
            aggregation_policy=args.aggregation_policy,
            trim_fraction=float(args.trim_fraction),
            max_nearest_time_delta_s=args.max_nearest_time_delta_s,
        )
        paths["weight_grid_csv"] = output / CLASS_WEIGHT_GRID_CSV
        paths["weights_json"] = output / CLASS_WEIGHTS_JSON
        grid.to_csv(paths["weight_grid_csv"], index=False)
        paths["weights_json"].write_text(
            json.dumps(_jsonable(weight_config), indent=2),
            encoding="utf-8",
        )
    else:
        parser.error("provide --truth-csv to search weights or --weights-json to apply")

    if should_write_submission:
        paths.update(
            write_class_conditioned_ensemble_outputs(
                estimate_inputs=estimates,
                template=template,
                class_map=class_map,
                weight_config=weight_config,
                output_dir=output / "class_conditioned_submission",
                default_classification=args.default_classification,
                max_nearest_time_delta_s=args.max_nearest_time_delta_s,
            )
        )
    print("mmuad_track5_class_ensemble=ok")
    for name, path in paths.items():
        print(f"{name}={path}")
    if args.require_leaderboard_ready:
        if "validation_json" not in paths:
            raise SystemExit("class-conditioned ensemble readiness check produced no validation output")
        validation = json.loads(paths["validation_json"].read_text(encoding="utf-8"))
        if not validation.get("leaderboard_ready", False):
            reasons = ", ".join(validation.get("leaderboard_blocking_reasons", [])) or "unknown"
            raise SystemExit(f"class-conditioned ensemble is not leaderboard-ready: {reasons}")
    return 0


def _search_weights_for_template(
    loaded: list[tuple[str, pd.DataFrame, float]],
    *,
    inputs: tuple[EstimateInput, ...],
    template: pd.DataFrame,
    truth: pd.DataFrame,
    class_label: str,
    weight_step: float,
    aggregation_policy: str,
    trim_fraction: float,
    max_nearest_time_delta_s: float | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for weights in _simplex_weight_grid(len(inputs), step=float(weight_step)):
        weighted_inputs = [
            (label, rows, float(weight))
            for (label, rows, _), weight in zip(loaded, weights, strict=True)
        ]
        estimates, diagnostics = build_track5_estimate_ensemble(
            weighted_inputs,
            template,
            max_nearest_time_delta_s=max_nearest_time_delta_s,
            aggregation_policy=aggregation_policy,
            trim_fraction=trim_fraction,
        )
        metrics = _score_template_estimates(estimates, truth)
        record = {
            "class_label": class_label,
            "template_rows": int(len(template)),
            "valid_input_count_mean": _safe_mean(diagnostics.get("valid_input_count", pd.Series(dtype=float))),
            **metrics,
        }
        for item, weight in zip(inputs, weights, strict=True):
            record[f"weight_{item.label}"] = float(weight)
        records.append(record)
    grid = pd.DataFrame.from_records(records)
    best_row = grid.sort_values(["pose_mse_m2", "pose_p95_m", "pose_max_m"]).iloc[0]
    best = {
        "class_label": class_label,
        "weights": {item.label: float(best_row[f"weight_{item.label}"]) for item in inputs},
        "metrics": {
            key: _jsonable(best_row[key])
            for key in ("matched_rows", "pose_mse_m2", "pose_rmse_m", "pose_mean_m", "pose_p95_m", "pose_max_m")
            if key in best_row.index
        },
    }
    return records, best


def _template_groups_by_class(
    template: pd.DataFrame,
    class_map: dict[str, str],
) -> list[tuple[str, pd.DataFrame]]:
    rows = template.copy()
    rows["_class_label"] = [class_map.get(str(seq), "__global__") for seq in rows["sequence_id"]]
    return [(str(label), group.drop(columns=["_class_label"])) for label, group in rows.groupby("_class_label", sort=True)]


def _normalized_weight_map(raw: Any, inputs: tuple[EstimateInput, ...]) -> dict[str, float]:
    if not isinstance(raw, dict) or not raw:
        raw = {item.label: 1.0 for item in inputs}
    weights = {item.label: float(raw.get(item.label, 0.0)) for item in inputs}
    total = sum(max(value, 0.0) for value in weights.values())
    if total <= 0.0:
        raise ValueError("weight map must contain at least one positive weight")
    return {label: max(value, 0.0) / total for label, value in weights.items()}


def _simplex_weight_grid(count: int, *, step: float) -> list[tuple[float, ...]]:
    if count == 1:
        return [(1.0,)]
    units = int(round(1.0 / step))
    if units <= 0 or not np.isclose(units * step, 1.0, atol=1.0e-9):
        raise ValueError("weight_step must divide 1.0 evenly")
    return [tuple(value / units for value in row) for row in _simplex_integer_grid(count, units)]


def _simplex_integer_grid(count: int, total: int) -> list[tuple[int, ...]]:
    if count == 1:
        return [(total,)]
    rows: list[tuple[int, ...]] = []
    for value in range(total + 1):
        for tail in _simplex_integer_grid(count - 1, total - value):
            rows.append((value, *tail))
    return rows


def _normalize_template_rows(template: pd.DataFrame) -> pd.DataFrame:
    rows = pd.DataFrame(template).copy()
    sequence_column = _first_present(rows, ("sequence_id", "Sequence", "sequence", "seq"))
    time_column = _first_present(rows, ("time_s", "Timestamp", "timestamp", "timestamp_s", "time"))
    if sequence_column is None or time_column is None:
        raise ValueError("template must contain sequence and timestamp columns")
    out = pd.DataFrame(
        {
            "sequence_id": rows[sequence_column].map(_sequence_id_or_none),
            "time_s": pd.to_numeric(rows[time_column], errors="coerce"),
        }
    )
    finite = out["sequence_id"].notna() & np.isfinite(out["time_s"].to_numpy(float))
    return out.loc[finite].sort_values(["sequence_id", "time_s"]).reset_index(drop=True)


def _normalize_truth_rows(truth: pd.DataFrame) -> pd.DataFrame:
    rows = pd.DataFrame(truth).copy()
    rows["sequence_id"] = rows["sequence_id"].map(_sequence_id_or_none)
    for column in ("time_s", "x_m", "y_m", "z_m"):
        rows[column] = pd.to_numeric(rows[column], errors="coerce")
    rows["_time_key"] = _time_key(rows["time_s"])
    finite = rows["sequence_id"].notna() & np.isfinite(
        rows[["time_s", "x_m", "y_m", "z_m"]].to_numpy(float)
    ).all(axis=1)
    return rows.loc[finite, ["sequence_id", "_time_key", "x_m", "y_m", "z_m"]].copy()


def _score_template_estimates(estimates: pd.DataFrame, truth: pd.DataFrame) -> dict[str, Any]:
    rows = pd.DataFrame(estimates).copy()
    rows["sequence_id"] = rows["sequence_id"].map(_sequence_id_or_none)
    rows["_time_key"] = _time_key(pd.to_numeric(rows["time_s"], errors="coerce"))
    merged = rows.merge(truth, on=["sequence_id", "_time_key"], how="inner")
    if merged.empty:
        return _empty_metrics()
    estimate_xyz = merged[["state_x_m", "state_y_m", "state_z_m"]].to_numpy(float)
    truth_xyz = merged[["x_m", "y_m", "z_m"]].to_numpy(float)
    finite = np.isfinite(estimate_xyz).all(axis=1) & np.isfinite(truth_xyz).all(axis=1)
    if not finite.any():
        return _empty_metrics()
    errors = np.linalg.norm(estimate_xyz[finite] - truth_xyz[finite], axis=1)
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


def _concat(parts: list[pd.DataFrame]) -> pd.DataFrame:
    return pd.concat(parts, ignore_index=True, sort=False) if parts else pd.DataFrame()


def _time_key(values: pd.Series) -> pd.Series:
    return pd.to_numeric(values, errors="coerce").round(9).astype(str)


def _safe_mean(values: pd.Series) -> float | None:
    numeric = pd.to_numeric(values, errors="coerce")
    numeric = numeric[np.isfinite(numeric.to_numpy(float))]
    return None if numeric.empty else float(np.mean(numeric.to_numpy(float)))


def _first_present(rows: pd.DataFrame, names: tuple[str, ...]) -> str | None:
    lower = {str(column).lower(): str(column) for column in rows.columns}
    for name in names:
        if name in rows.columns:
            return name
        found = lower.get(name.lower())
        if found is not None:
            return found
    return None


def _normalize_sequence_class_map(class_map: dict[str, str]) -> dict[str, str]:
    normalized: dict[str, str] = {}
    for key, value in class_map.items():
        sequence_id = _sequence_id_or_none(key)
        if sequence_id is not None:
            normalized[sequence_id] = str(value)
    return normalized


def _sequence_id_or_none(value: object) -> str | None:
    try:
        return parse_official_sequence_cell(value)
    except ValueError:
        return None


def _safe_label(value: object) -> str:
    # Class maps are keyed by official sequence ids, not by filenames. Preserve
    # internal spaces and path-like separators so they continue to match template rows.
    text = str(value).strip()
    return text or "sequence"


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
