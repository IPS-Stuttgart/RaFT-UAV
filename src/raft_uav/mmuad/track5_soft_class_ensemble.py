"""Soft class-conditioned Track 5 estimate ensembling.

The hard class-conditioned ensemble selects one set of pose-pipeline weights from
a predicted class label.  This module keeps the same train-selected class weight
configuration but blends the class-specific pose ensembles by sequence-level class
probabilities.  The apply path is inference-safe: it uses fixed weights, predicted
class probabilities, estimate CSVs, and the official timestamp template only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from raft_uav.mmuad.submission import (
    load_official_track5_template_file,
    load_sequence_class_map,
    validate_official_track5_submission,
    write_official_mmaud_results_csv,
    write_official_ug2_codabench_zip,
)
from raft_uav.mmuad.track5_estimate_ensemble import EstimateInput
from raft_uav.mmuad.track5_estimate_ensemble import build_track5_estimate_ensemble
from raft_uav.mmuad.track5_estimate_ensemble import parse_estimate_spec

SOFT_CLASS_ESTIMATES_CSV = "mmuad_track5_soft_class_ensemble_estimates.csv"
SOFT_CLASS_DIAGNOSTICS_CSV = "mmuad_track5_soft_class_ensemble_diagnostics.csv"
SOFT_CLASS_MANIFEST_JSON = "mmuad_track5_soft_class_ensemble_manifest.json"
SOFT_CLASS_VALIDATION_JSON = "mmuad_track5_soft_class_ensemble_validation.json"
SOFT_CLASS_VALIDATION_ROWS_CSV = "mmuad_track5_soft_class_ensemble_validation_rows.csv"
OFFICIAL_RESULTS_CSV = "mmaud_results.csv"
OFFICIAL_ZIP = "ug2_submission.zip"
SEQUENCE_ALIASES = ("sequence_id", "Sequence", "sequence", "seq")
PREDICTED_CLASS_ALIASES = ("predicted_class", "Classification", "uav_type", "class_id")
XYZ_COLUMNS = ("state_x_m", "state_y_m", "state_z_m")


def build_soft_class_conditioned_estimate_ensemble(
    estimate_inputs: Iterable[EstimateInput],
    *,
    template: pd.DataFrame,
    class_probabilities: pd.DataFrame,
    weight_config: dict[str, Any],
    aggregation_policy: str | None = None,
    trim_fraction: float | None = None,
    max_nearest_time_delta_s: float | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return estimates blended by sequence-level class probabilities."""

    inputs = tuple(estimate_inputs)
    if not inputs:
        raise ValueError("at least one estimate input is required")
    template_rows = _normalize_template_rows(template)
    probability_rows = _normalize_probability_rows(class_probabilities)
    global_weights = _normalized_weight_map(_global_weights(weight_config), inputs)
    class_weight_config = weight_config.get("class_weights", {})
    if class_weight_config is None:
        class_weight_config = {}
    if not isinstance(class_weight_config, dict):
        raise ValueError("weight config class_weights must be an object")
    class_labels = _class_labels(probability_rows, class_weight_config)
    loaded = {item.label: pd.read_csv(item.path) for item in inputs}
    policy = str(aggregation_policy or weight_config.get("aggregation_policy", "weighted-mean"))
    trim = _select_trim_fraction(trim_fraction, weight_config)

    global_estimates, _ = build_track5_estimate_ensemble(
        [(item.label, loaded[item.label], global_weights[item.label]) for item in inputs],
        template_rows,
        max_nearest_time_delta_s=max_nearest_time_delta_s,
        aggregation_policy=policy,
        trim_fraction=trim,
    )
    class_estimates: dict[str, pd.DataFrame] = {}
    for label in class_labels:
        weights = _normalized_weight_map(class_weight_config.get(label, global_weights), inputs)
        estimates, _ = build_track5_estimate_ensemble(
            [(item.label, loaded[item.label], weights[item.label]) for item in inputs],
            template_rows,
            max_nearest_time_delta_s=max_nearest_time_delta_s,
            aggregation_policy=policy,
            trim_fraction=trim,
        )
        class_estimates[label] = estimates

    global_lookup = _estimate_lookup(global_estimates)
    class_lookups = {label: _estimate_lookup(frame) for label, frame in class_estimates.items()}
    probability_lookup = {
        str(row["sequence_id"]): row.to_dict() for _, row in probability_rows.iterrows()
    }
    estimate_records: list[dict[str, Any]] = []
    diagnostic_records: list[dict[str, Any]] = []
    for _, template_row in template_rows.iterrows():
        sequence_id = str(template_row["sequence_id"])
        time_s = float(template_row["time_s"])
        key = _row_key(sequence_id, time_s)
        probabilities = _probabilities_for_sequence(
            probability_lookup.get(sequence_id),
            class_labels=class_labels,
        )
        class_probability_available = probabilities is not None
        if probabilities is None:
            xyz = global_lookup.get(key, _nan_xyz()).copy()
            effective_labels = ["__global__"]
            effective_probability_sum = 1.0 if np.isfinite(xyz).all() else 0.0
            entropy = np.nan
            prob_json = "{}"
        else:
            xyz, effective_labels, effective_probability_sum = _blend_class_estimates(
                key,
                probabilities,
                class_lookups,
                global_lookup,
            )
            entropy = _entropy(probabilities.values())
            prob_json = json.dumps(probabilities, sort_keys=True)
        estimate_records.append(
            {
                "sequence_id": sequence_id,
                "time_s": time_s,
                "source": "track5-soft-class-ensemble",
                "track_id": "track5-soft-class-ensemble",
                "state_x_m": float(xyz[0]) if np.isfinite(xyz[0]) else np.nan,
                "state_y_m": float(xyz[1]) if np.isfinite(xyz[1]) else np.nan,
                "state_z_m": float(xyz[2]) if np.isfinite(xyz[2]) else np.nan,
                "soft_class_probability_available": bool(class_probability_available),
                "soft_class_probability_entropy": entropy,
                "soft_class_effective_probability_sum": effective_probability_sum,
                "soft_class_labels_used": ";".join(effective_labels),
            }
        )
        diagnostic_records.append(
            {
                "sequence_id": sequence_id,
                "time_s": time_s,
                "class_probability_available": bool(class_probability_available),
                "class_probabilities_json": prob_json,
                "probability_entropy": entropy,
                "effective_probability_sum": effective_probability_sum,
                "labels_used": ";".join(effective_labels),
            }
        )
    estimates = pd.DataFrame.from_records(estimate_records)
    diagnostics = pd.DataFrame.from_records(diagnostic_records)
    return estimates, diagnostics


def write_soft_class_conditioned_ensemble_outputs(
    *,
    estimate_inputs: Iterable[EstimateInput],
    template: pd.DataFrame,
    class_probabilities: pd.DataFrame,
    weight_config: dict[str, Any],
    output_dir: Path,
    class_map: dict[str, str] | None = None,
    default_classification: int | str = 0,
    aggregation_policy: str | None = None,
    trim_fraction: float | None = None,
    max_nearest_time_delta_s: float | None = None,
) -> dict[str, Path]:
    """Write soft class-conditioned estimates and official Track 5 artifacts."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    input_list = list(estimate_inputs)
    estimates, diagnostics = build_soft_class_conditioned_estimate_ensemble(
        input_list,
        template=template,
        class_probabilities=class_probabilities,
        weight_config=weight_config,
        aggregation_policy=aggregation_policy,
        trim_fraction=trim_fraction,
        max_nearest_time_delta_s=max_nearest_time_delta_s,
    )
    paths = {
        "estimates_csv": output / SOFT_CLASS_ESTIMATES_CSV,
        "diagnostics_csv": output / SOFT_CLASS_DIAGNOSTICS_CSV,
        "official_results_csv": output / OFFICIAL_RESULTS_CSV,
        "official_zip": output / OFFICIAL_ZIP,
        "validation_json": output / SOFT_CLASS_VALIDATION_JSON,
        "validation_rows_csv": output / SOFT_CLASS_VALIDATION_ROWS_CSV,
        "manifest_json": output / SOFT_CLASS_MANIFEST_JSON,
    }
    estimates.to_csv(paths["estimates_csv"], index=False)
    diagnostics.to_csv(paths["diagnostics_csv"], index=False)
    write_official_mmaud_results_csv(
        estimates,
        paths["official_results_csv"],
        classification=default_classification,
        class_map=class_map or {},
        invalid_row_policy="raise",
    )
    write_official_ug2_codabench_zip(
        estimates,
        paths["official_zip"],
        classification=default_classification,
        class_map=class_map or {},
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
    probability_rows = _normalize_probability_rows(class_probabilities)
    manifest = {
        "schema": "raft-uav-mmuad-track5-soft-class-ensemble-v1",
        "weight_config_schema": weight_config.get("schema"),
        "class_weights": weight_config.get("class_weights", {}),
        "global_weights": _global_weights(weight_config),
        "estimate_inputs": [
            {"label": item.label, "path": str(item.path), "weight": float(item.weight)}
            for item in input_list
        ],
        "class_probability_sequence_count": int(len(probability_rows)),
        "row_count": int(len(estimates)),
        "probability_available_rows": int(estimates["soft_class_probability_available"].astype(bool).sum()),
        "mean_probability_entropy": _safe_mean(diagnostics.get("probability_entropy", pd.Series(dtype=float))),
        "leaderboard_ready": bool(validation.summary.get("leaderboard_ready", False)),
        "codabench_upload_ready": bool(validation.summary.get("codabench_upload_ready", False)),
        "paths": {name: str(path) for name, path in paths.items() if name != "manifest_json"},
    }
    paths["manifest_json"].write_text(json.dumps(_jsonable(manifest), indent=2), encoding="utf-8")
    return paths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="raft-uav-mmuad-track5-soft-class-ensemble",
        description="blend class-conditioned Track 5 pose ensembles by class probabilities",
    )
    parser.add_argument("--estimate-csv", action="append", default=[], metavar="LABEL=PATH")
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--class-probabilities-csv", type=Path, required=True)
    parser.add_argument("--weight-config-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--class-map", type=Path)
    parser.add_argument("--default-classification", default="0")
    parser.add_argument("--aggregation-policy")
    parser.add_argument("--trim-fraction", type=float)
    parser.add_argument("--max-nearest-time-delta-s", type=float)
    parser.add_argument("--require-leaderboard-ready", action="store_true")
    args = parser.parse_args(argv)

    if not args.estimate_csv:
        parser.error("provide at least one --estimate-csv LABEL=PATH")
    estimate_inputs = [parse_estimate_spec(value) for value in args.estimate_csv]
    template = load_official_track5_template_file(args.template)
    probabilities = pd.read_csv(args.class_probabilities_csv)
    weight_config = json.loads(args.weight_config_json.read_text(encoding="utf-8"))
    class_map = load_sequence_class_map(args.class_map) if args.class_map is not None else {}
    paths = write_soft_class_conditioned_ensemble_outputs(
        estimate_inputs=estimate_inputs,
        template=template,
        class_probabilities=probabilities,
        weight_config=weight_config,
        output_dir=args.output_dir,
        class_map=class_map,
        default_classification=args.default_classification,
        aggregation_policy=args.aggregation_policy,
        trim_fraction=args.trim_fraction,
        max_nearest_time_delta_s=args.max_nearest_time_delta_s,
    )
    validation = json.loads(Path(paths["validation_json"]).read_text(encoding="utf-8"))
    print("mmuad_track5_soft_class_ensemble=ok")
    for name, path in paths.items():
        print(f"{name}={path}")
    print(f"leaderboard_ready={validation.get('leaderboard_ready')}")
    print(f"codabench_upload_ready={validation.get('codabench_upload_ready')}")
    if args.require_leaderboard_ready and not validation.get("leaderboard_ready", False):
        reasons = ", ".join(validation.get("leaderboard_blocking_reasons", [])) or "unknown"
        raise SystemExit(f"soft class ensemble is not leaderboard-ready: {reasons}")
    return 0


def _normalize_template_rows(template: pd.DataFrame) -> pd.DataFrame:
    rows = pd.DataFrame(template).copy()
    sequence_column = _first_present(rows, SEQUENCE_ALIASES)
    time_column = _first_present(rows, ("time_s", "Timestamp", "timestamp", "timestamp_s", "time"))
    if sequence_column is None or time_column is None:
        raise ValueError("template must contain sequence and timestamp columns")
    out = pd.DataFrame(
        {
            "sequence_id": rows[sequence_column].astype(str),
            "time_s": pd.to_numeric(rows[time_column], errors="coerce"),
        }
    )
    finite = np.isfinite(out["time_s"].to_numpy(float))
    return out.loc[finite].sort_values(["sequence_id", "time_s"]).reset_index(drop=True)


def _normalize_probability_rows(probabilities: pd.DataFrame) -> pd.DataFrame:
    rows = pd.DataFrame(probabilities).copy()
    if rows.empty:
        return pd.DataFrame(columns=["sequence_id"])
    sequence_column = _first_present(rows, SEQUENCE_ALIASES)
    if sequence_column is None:
        raise ValueError("class probabilities must contain sequence_id/Sequence")
    out = pd.DataFrame({"sequence_id": rows[sequence_column].astype(str)})
    found_probability = False
    for label in _official_class_labels():
        column = _probability_column(rows, label)
        if column is not None:
            out[f"class_prob_{label}"] = pd.to_numeric(rows[column], errors="coerce")
            found_probability = True
    if not found_probability:
        predicted_column = _first_present(rows, PREDICTED_CLASS_ALIASES)
        if predicted_column is None:
            raise ValueError("class probabilities need probability columns or predicted_class")
        predicted = rows[predicted_column].astype(str)
        for label in _official_class_labels():
            out[f"class_prob_{label}"] = (predicted == label).astype(float)
    out = out.groupby("sequence_id", as_index=False).mean(numeric_only=True)
    return _normalize_probability_mass(out)


def _probability_column(rows: pd.DataFrame, label: str) -> str | None:
    return _first_present(
        rows,
        (
            f"class_prob_{label}",
            f"image_class_prob_{label}",
            f"predicted_probability_{label}",
            f"class_probability_{label}",
            f"probability_{label}",
            f"p_class_{label}",
        ),
    )


def _normalize_probability_mass(rows: pd.DataFrame) -> pd.DataFrame:
    out = rows.copy()
    columns = [column for column in out.columns if str(column).startswith("class_prob_")]
    for column in columns:
        out[column] = pd.to_numeric(out[column], errors="coerce").clip(lower=0.0).fillna(0.0)
    totals = out[columns].sum(axis=1) if columns else pd.Series(0.0, index=out.index)
    if columns:
        for column in columns:
            out[column] = np.where(totals > 0.0, out[column] / totals, 0.0)
    return out


def _class_labels(probability_rows: pd.DataFrame, class_weights: dict[str, Any]) -> list[str]:
    labels = {str(key) for key in class_weights}
    labels.update(
        str(column).removeprefix("class_prob_")
        for column in probability_rows.columns
        if str(column).startswith("class_prob_")
    )
    labels.update(_official_class_labels())
    return sorted(labels)


def _global_weights(config: dict[str, Any]) -> dict[str, Any]:
    if isinstance(config.get("global_weights"), dict):
        return dict(config["global_weights"])
    if isinstance(config.get("weights"), dict):
        return dict(config["weights"])
    raise ValueError("weight config must contain global_weights or weights")


def _normalized_weight_map(raw: Any, inputs: tuple[EstimateInput, ...]) -> dict[str, float]:
    if not isinstance(raw, dict):
        raise ValueError("weights must be an object")
    labels = [item.label for item in inputs]
    weights: dict[str, float] = {}
    for label in labels:
        value = raw.get(label, 0.0)
        weight = float(value)
        if not np.isfinite(weight) or weight < 0.0:
            raise ValueError(f"weight for {label!r} must be finite and non-negative")
        weights[label] = weight
    total = sum(weights.values())
    if total <= 0.0:
        raise ValueError("weight sum must be positive")
    return {label: value / total for label, value in weights.items()}


def _estimate_lookup(estimates: pd.DataFrame) -> dict[tuple[str, str], np.ndarray]:
    lookup: dict[tuple[str, str], np.ndarray] = {}
    for _, row in pd.DataFrame(estimates).iterrows():
        key = _row_key(str(row["sequence_id"]), float(row["time_s"]))
        lookup[key] = row[list(XYZ_COLUMNS)].to_numpy(float)
    return lookup


def _probabilities_for_sequence(
    row: dict[str, Any] | None,
    *,
    class_labels: list[str],
) -> dict[str, float] | None:
    if row is None:
        return None
    probs = {
        label: float(pd.to_numeric(pd.Series([row.get(f"class_prob_{label}", 0.0)]), errors="coerce").iloc[0])
        for label in class_labels
    }
    probs = {label: value for label, value in probs.items() if np.isfinite(value) and value > 0.0}
    total = sum(probs.values())
    if total <= 0.0:
        return None
    return {label: value / total for label, value in probs.items()}


def _blend_class_estimates(
    key: tuple[str, str],
    probabilities: dict[str, float],
    class_lookups: dict[str, dict[tuple[str, str], np.ndarray]],
    global_lookup: dict[tuple[str, str], np.ndarray],
) -> tuple[np.ndarray, list[str], float]:
    weighted_sum = np.zeros(3, dtype=float)
    total = 0.0
    labels: list[str] = []
    for label, probability in probabilities.items():
        xyz = class_lookups.get(label, global_lookup).get(key, _nan_xyz())
        if np.isfinite(xyz).all():
            weighted_sum += float(probability) * xyz
            total += float(probability)
            labels.append(str(label))
    if total <= 0.0:
        return global_lookup.get(key, _nan_xyz()).copy(), ["__global__"], 0.0
    return weighted_sum / total, labels, total


def _select_trim_fraction(override: float | None, weight_config: dict[str, Any]) -> float:
    value = override if override is not None else weight_config.get("trim_fraction", 0.2)
    trim = float(value)
    if not np.isfinite(trim) or not 0.0 <= trim < 0.5:
        raise ValueError("trim_fraction must be finite and in [0, 0.5)")
    return trim


def _entropy(values: Iterable[float]) -> float:
    arr = np.asarray([value for value in values if value > 0.0], dtype=float)
    if arr.size == 0:
        return np.nan
    arr = arr / float(np.sum(arr))
    return float(-np.sum(arr * np.log(arr)))


def _safe_mean(values: pd.Series) -> float | None:
    numeric = pd.to_numeric(values, errors="coerce")
    numeric = numeric[np.isfinite(numeric.to_numpy(float))]
    if numeric.empty:
        return None
    return float(np.mean(numeric.to_numpy(float)))


def _row_key(sequence_id: str, time_s: float) -> tuple[str, str]:
    return str(sequence_id), f"{float(time_s):.9f}"


def _nan_xyz() -> np.ndarray:
    return np.asarray([np.nan, np.nan, np.nan], dtype=float)


def _first_present(rows: pd.DataFrame, names: tuple[str, ...]) -> str | None:
    lower = {str(column).lower(): str(column) for column in rows.columns}
    for name in names:
        if name in rows.columns:
            return name
        found = lower.get(name.lower())
        if found is not None:
            return found
    return None


def _official_class_labels() -> tuple[str, ...]:
    return ("0", "1", "2", "3")


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
