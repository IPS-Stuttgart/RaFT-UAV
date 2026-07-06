"""Weighted Track 5 estimate ensembling for MMUAD leaderboard submissions.

Different MMUAD pose pipelines often make partially independent errors
(candidate-mixture settings, reservoir settings, calibration branches, and
tracker variants).  This module combines multiple estimate trajectories after
resampling each one onto the official Track 5 timestamp template.  It is
inference-safe: the template contributes only requested Sequence/Timestamp rows,
and weights are supplied explicitly or selected upstream on train folds.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from raft_uav.mmuad.submission import (
    load_official_track5_template_file,
    load_sequence_class_map,
    parse_official_sequence_cell,
    validate_official_track5_submission,
    write_official_mmaud_results_csv,
    write_official_ug2_codabench_zip,
)
from raft_uav.mmuad.track5_template_resample import resample_estimates_to_track5_template


ENSEMBLED_ESTIMATES_CSV = "mmuad_track5_ensemble_estimates.csv"
ENSEMBLE_DIAGNOSTICS_CSV = "mmuad_track5_ensemble_diagnostics.csv"
ENSEMBLE_MANIFEST_JSON = "mmuad_track5_ensemble_manifest.json"
VALIDATION_JSON = "mmuad_track5_ensemble_validation.json"
VALIDATION_ROWS_CSV = "mmuad_track5_ensemble_validation_rows.csv"
OFFICIAL_RESULTS_CSV = "mmaud_results.csv"
OFFICIAL_ZIP = "ug2_submission.zip"
ENSEMBLE_POLICIES = ("weighted-mean", "weighted-median", "trimmed-mean")
WEIGHT_MISSING_POLICIES = ("error", "keep", "zero")
# Resampling copies requested template timestamps into each candidate row. Match those
# rows with an absolute tolerance only; NumPy's default relative tolerance is unsafe
# for epoch-style timestamps because seconds-scale differences can compare close.
TEMPLATE_TIME_MATCH_ATOL_S = 1.0e-9


@dataclass(frozen=True)
class EstimateInput:
    """One estimate trajectory input for a Track 5 ensemble."""

    label: str
    path: Path
    weight: float = 1.0


def parse_estimate_spec(value: str) -> EstimateInput:
    """Parse ``LABEL=PATH`` or ``LABEL=PATH@WEIGHT`` estimate specs."""

    if "=" not in value:
        path = Path(value)
        label = _safe_label(path.stem)
        return EstimateInput(
            label=label,
            path=path,
            weight=_validate_ensemble_weight(1.0, label=label),
        )
    label_text, path_weight = value.split("=", 1)
    label = _safe_label(label_text)
    weight = 1.0
    path_text = path_weight
    if "@" in path_weight:
        path_text, weight_text = path_weight.rsplit("@", 1)
        weight = float(weight_text)
    return EstimateInput(
        label=label,
        path=Path(path_text),
        weight=_validate_ensemble_weight(weight, label=label),
    )


def load_estimate_weight_config(path: Path) -> dict[str, float]:
    """Load a train-selected ensemble-weight mapping from JSON.

    The preferred schema is ``{"weights": {"label": weight, ...}}``, matching
    the output of the Track 5 weight-search helper.  A direct ``{"label":
    weight}`` mapping is accepted for small hand-authored configs.
    """

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("ensemble weight config must be a JSON object")
    raw_weights = payload.get("weights", payload)
    if not isinstance(raw_weights, dict):
        raise ValueError("ensemble weight config must contain a weights object")
    weights: dict[str, float] = {}
    for label, value in raw_weights.items():
        safe_label = _safe_label(str(label))
        weights[safe_label] = _validate_ensemble_weight(float(value), label=safe_label)
    if not weights:
        raise ValueError("ensemble weight config contains no weights")
    return weights


def apply_estimate_weight_config(
    estimate_inputs: Iterable[EstimateInput],
    weights: dict[str, float],
    *,
    missing_policy: str = "error",
) -> list[EstimateInput]:
    """Return estimate inputs with weights replaced by a config mapping."""

    if missing_policy not in WEIGHT_MISSING_POLICIES:
        raise ValueError(f"unsupported weight missing policy: {missing_policy}")
    safe_weights = {_safe_label(label): float(weight) for label, weight in weights.items()}
    out: list[EstimateInput] = []
    missing: list[str] = []
    for item in estimate_inputs:
        label = _safe_label(item.label)
        if label in safe_weights:
            weight = _validate_ensemble_weight(safe_weights[label], label=label)
        elif missing_policy == "keep":
            weight = item.weight
            missing.append(label)
        elif missing_policy == "zero":
            weight = 0.0
            missing.append(label)
        else:
            missing.append(label)
            continue
        out.append(EstimateInput(label=label, path=item.path, weight=weight))
    if missing and missing_policy == "error":
        raise ValueError(f"missing ensemble weights for labels: {sorted(missing)}")
    unused = sorted(set(safe_weights).difference(_safe_label(item.label) for item in estimate_inputs))
    if unused:
        raise ValueError(f"ensemble weight config has unused labels: {unused}")
    return out


def _validate_ensemble_weight(weight: float, *, label: str) -> float:
    weight = float(weight)
    if not np.isfinite(weight) or weight < 0.0:
        raise ValueError(
            f"estimate weight must be finite and non-negative for {label}: {weight}"
        )
    return weight


def build_track5_estimate_ensemble(
    estimate_inputs: Iterable[tuple[str, pd.DataFrame, float]],
    template: pd.DataFrame,
    *,
    max_nearest_time_delta_s: float | None = None,
    aggregation_policy: str = "weighted-mean",
    trim_fraction: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return ensembled estimates and per-template-row diagnostics.

    Each estimate input is first interpolated to the template with
    :func:`resample_estimates_to_track5_template`.  The final position is a
    row-wise aggregate over finite, valid resampled trajectories.  ``weighted-mean``
    keeps the original behavior, while ``weighted-median`` and ``trimmed-mean``
    provide leaderboard-safe robust alternatives for ensembling partially
    independent pose pipelines with occasional outlier trajectories.
    """

    if aggregation_policy not in ENSEMBLE_POLICIES:
        raise ValueError(f"unsupported aggregation_policy: {aggregation_policy}")
    if not 0.0 <= float(trim_fraction) < 0.5:
        raise ValueError("trim_fraction must be in [0, 0.5)")

    template_rows = _normalize_template_rows(template)
    if template_rows.empty:
        empty = pd.DataFrame(columns=["sequence_id", "time_s", "state_x_m", "state_y_m", "state_z_m"])
        return empty, pd.DataFrame()

    resampled_parts: list[pd.DataFrame] = []
    input_summaries: list[dict[str, Any]] = []
    for label, estimates, weight in estimate_inputs:
        label = _safe_label(label)
        weight = _validate_ensemble_weight(weight, label=label)
        resampled, diagnostics = resample_estimates_to_track5_template(
            estimates,
            template_rows,
            max_nearest_time_delta_s=max_nearest_time_delta_s,
        )
        part = resampled[["sequence_id", "time_s", "state_x_m", "state_y_m", "state_z_m"]].copy()
        part["ensemble_label"] = label
        part["ensemble_weight"] = weight
        part["ensemble_valid"] = _finite_xyz(part) & resampled.get(
            "template_resample_valid",
            pd.Series(True, index=resampled.index),
        ).astype(bool)
        if "template_nearest_time_delta_s" in resampled.columns:
            part["template_nearest_time_delta_s"] = pd.to_numeric(
                resampled["template_nearest_time_delta_s"],
                errors="coerce",
            )
        resampled_parts.append(part)
        input_summaries.append(
            {
                "label": label,
                "weight": weight,
                "input_estimate_rows": int(len(estimates)),
                "template_rows": int(len(template_rows)),
                "valid_resampled_rows": int(part["ensemble_valid"].sum()),
                "mean_abs_nearest_time_delta_s": _safe_mean_abs(
                    diagnostics.get("nearest_time_delta_s", pd.Series(dtype=float))
                ),
            }
        )
    if not resampled_parts:
        raise ValueError("at least one estimate input is required")
    stacked = pd.concat(resampled_parts, ignore_index=True, sort=False)
    records: list[dict[str, Any]] = []
    diagnostics_records: list[dict[str, Any]] = []
    for _, template_row in template_rows.iterrows():
        sequence_id = str(template_row["sequence_id"])
        time_s = float(template_row["time_s"])
        rows = stacked.loc[
            (stacked["sequence_id"].astype(str) == sequence_id)
            & _template_time_matches(stacked["time_s"], time_s)
        ]
        valid = rows.loc[rows["ensemble_valid"].astype(bool) & (rows["ensemble_weight"] > 0.0)]
        if valid.empty:
            xyz = np.asarray([np.nan, np.nan, np.nan], dtype=float)
            total_weight = 0.0
            labels = ""
            spread = np.nan
        else:
            weights = valid["ensemble_weight"].to_numpy(float)
            xyz_values = valid[["state_x_m", "state_y_m", "state_z_m"]].to_numpy(float)
            total_weight = float(np.sum(weights))
            xyz = _aggregate_xyz(
                xyz_values,
                weights,
                policy=aggregation_policy,
                trim_fraction=float(trim_fraction),
            )
            labels = ";".join(valid["ensemble_label"].astype(str).tolist())
            spread = _weighted_spread_m(xyz_values, weights, xyz)
        records.append(
            {
                "sequence_id": sequence_id,
                "time_s": time_s,
                "state_x_m": float(xyz[0]) if np.isfinite(xyz[0]) else np.nan,
                "state_y_m": float(xyz[1]) if np.isfinite(xyz[1]) else np.nan,
                "state_z_m": float(xyz[2]) if np.isfinite(xyz[2]) else np.nan,
                "track5_ensemble": True,
                "ensemble_source_count": int(len(valid)),
                "ensemble_weight_sum": total_weight,
                "ensemble_labels": labels,
                "ensemble_policy": aggregation_policy,
                "ensemble_position_spread_m": spread,
            }
        )
        diagnostics_records.append(
            {
                "sequence_id": sequence_id,
                "time_s": time_s,
                "candidate_input_count": int(len(rows)),
                "valid_input_count": int(len(valid)),
                "weight_sum": total_weight,
                "labels": labels,
                "ensemble_policy": aggregation_policy,
                "position_spread_m": spread,
            }
        )
    ensemble = pd.DataFrame.from_records(records)
    diagnostics = pd.DataFrame.from_records(diagnostics_records)
    diagnostics.attrs["input_summaries"] = input_summaries
    return ensemble, diagnostics


def write_track5_estimate_ensemble_outputs(
    *,
    estimate_inputs: Iterable[EstimateInput],
    template: pd.DataFrame,
    output_dir: Path,
    class_map: dict[str, str] | None = None,
    default_classification: int | str = 0,
    max_nearest_time_delta_s: float | None = None,
    aggregation_policy: str = "weighted-mean",
    trim_fraction: float = 0.2,
) -> dict[str, Path]:
    """Write ensemble estimates, official CSV/ZIP, validation, and manifest."""

    estimate_input_list = list(estimate_inputs)
    loaded_inputs = [
        (item.label, pd.read_csv(item.path), float(item.weight)) for item in estimate_input_list
    ]
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    ensemble, diagnostics = build_track5_estimate_ensemble(
        loaded_inputs,
        template,
        max_nearest_time_delta_s=max_nearest_time_delta_s,
        aggregation_policy=aggregation_policy,
        trim_fraction=trim_fraction,
    )
    paths = {
        "ensemble_estimates_csv": output / ENSEMBLED_ESTIMATES_CSV,
        "diagnostics_csv": output / ENSEMBLE_DIAGNOSTICS_CSV,
        "official_results_csv": output / OFFICIAL_RESULTS_CSV,
        "official_zip": output / OFFICIAL_ZIP,
        "validation_json": output / VALIDATION_JSON,
        "validation_rows_csv": output / VALIDATION_ROWS_CSV,
        "manifest_json": output / ENSEMBLE_MANIFEST_JSON,
    }
    ensemble.to_csv(paths["ensemble_estimates_csv"], index=False)
    diagnostics.to_csv(paths["diagnostics_csv"], index=False)
    class_map = class_map or {}
    write_official_mmaud_results_csv(
        ensemble,
        paths["official_results_csv"],
        classification=default_classification,
        class_map=class_map,
        invalid_row_policy="raise",
    )
    write_official_ug2_codabench_zip(
        ensemble,
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
        "schema": "raft-uav-mmuad-track5-estimate-ensemble-v1",
        "estimate_inputs": [
            {"label": item.label, "path": str(item.path), "weight": float(item.weight)}
            for item in estimate_input_list
        ],
        "input_summaries": diagnostics.attrs.get("input_summaries", []),
        "row_count": int(len(ensemble)),
        "valid_ensemble_rows": int(_finite_xyz(ensemble).sum()),
        "max_nearest_time_delta_s": max_nearest_time_delta_s,
        "aggregation_policy": aggregation_policy,
        "trim_fraction": float(trim_fraction),
        "mean_position_spread_m": _safe_mean(diagnostics.get("position_spread_m", pd.Series(dtype=float))),
        "p95_position_spread_m": _safe_percentile(
            diagnostics.get("position_spread_m", pd.Series(dtype=float)),
            95,
        ),
        "leaderboard_ready": bool(validation.summary.get("leaderboard_ready", False)),
        "codabench_upload_ready": bool(validation.summary.get("codabench_upload_ready", False)),
        "paths": {name: str(path) for name, path in paths.items() if name != "manifest_json"},
    }
    paths["manifest_json"].write_text(json.dumps(_jsonable(manifest), indent=2), encoding="utf-8")
    return paths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="raft-uav-mmuad-track5-estimate-ensemble",
        description="ensemble MMUAD Track 5 estimate trajectories on an official template",
    )
    parser.add_argument(
        "--estimate-csv",
        action="append",
        default=[],
        metavar="LABEL=PATH[@WEIGHT]",
        help="estimate trajectory to include; may be repeated",
    )
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--class-map", type=Path)
    parser.add_argument("--default-classification", default="0")
    parser.add_argument("--max-nearest-time-delta-s", type=float)
    parser.add_argument("--aggregation-policy", choices=ENSEMBLE_POLICIES, default="weighted-mean")
    parser.add_argument("--trim-fraction", type=float, default=0.2)
    parser.add_argument(
        "--weights-json",
        type=Path,
        help="JSON config with a weights mapping, e.g. output from train-fold weight search",
    )
    parser.add_argument(
        "--weight-missing-policy",
        choices=WEIGHT_MISSING_POLICIES,
        default="error",
        help="how to handle estimate labels missing from --weights-json",
    )
    parser.add_argument("--require-leaderboard-ready", action="store_true")
    args = parser.parse_args(argv)

    if not args.estimate_csv:
        parser.error("provide at least one --estimate-csv LABEL=PATH[@WEIGHT]")
    estimate_inputs = [parse_estimate_spec(spec) for spec in args.estimate_csv]
    if args.weights_json is not None:
        estimate_inputs = apply_estimate_weight_config(
            estimate_inputs,
            load_estimate_weight_config(args.weights_json),
            missing_policy=args.weight_missing_policy,
        )
    template = load_official_track5_template_file(args.template)
    class_map = load_sequence_class_map(args.class_map) if args.class_map is not None else {}
    paths = write_track5_estimate_ensemble_outputs(
        estimate_inputs=estimate_inputs,
        template=template,
        output_dir=args.output_dir,
        class_map=class_map,
        default_classification=args.default_classification,
        max_nearest_time_delta_s=args.max_nearest_time_delta_s,
        aggregation_policy=args.aggregation_policy,
        trim_fraction=float(args.trim_fraction),
    )
    summary = json.loads(paths["validation_json"].read_text(encoding="utf-8"))
    print("mmuad_track5_estimate_ensemble=ok")
    for name, path in paths.items():
        print(f"{name}={path}")
    print(f"leaderboard_ready={summary.get('leaderboard_ready')}")
    print(f"codabench_upload_ready={summary.get('codabench_upload_ready')}")
    if args.require_leaderboard_ready and not summary.get("leaderboard_ready", False):
        reasons = ", ".join(summary.get("leaderboard_blocking_reasons", [])) or "unknown"
        raise SystemExit(f"ensemble upload is not leaderboard-ready: {reasons}")
    return 0


def _aggregate_xyz(
    xyz: np.ndarray,
    weights: np.ndarray,
    *,
    policy: str,
    trim_fraction: float,
) -> np.ndarray:
    if policy == "weighted-mean":
        return np.sum(weights[:, None] * xyz, axis=0) / float(np.sum(weights))
    if policy == "weighted-median":
        return np.asarray([_weighted_median(xyz[:, axis], weights) for axis in range(3)])
    if policy == "trimmed-mean":
        return np.asarray([_trimmed_weighted_mean(xyz[:, axis], weights, trim_fraction) for axis in range(3)])
    raise ValueError(f"unsupported aggregation policy: {policy}")


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values)
    sorted_values = values[order]
    sorted_weights = weights[order]
    cumulative = np.cumsum(sorted_weights)
    cutoff = 0.5 * float(np.sum(sorted_weights))
    return float(sorted_values[int(np.searchsorted(cumulative, cutoff, side="left"))])


def _trimmed_weighted_mean(values: np.ndarray, weights: np.ndarray, trim_fraction: float) -> float:
    if len(values) <= 2 or trim_fraction <= 0.0:
        return float(np.sum(weights * values) / np.sum(weights))
    order = np.argsort(values)
    sorted_values = values[order]
    sorted_weights = weights[order]
    trim = int(np.floor(float(trim_fraction) * len(values)))
    if trim > 0 and len(values) - 2 * trim > 0:
        sorted_values = sorted_values[trim:-trim]
        sorted_weights = sorted_weights[trim:-trim]
    return float(np.sum(sorted_weights * sorted_values) / np.sum(sorted_weights))


def _normalize_template_rows(template: pd.DataFrame) -> pd.DataFrame:
    rows = pd.DataFrame(template).copy()
    sequence_column = _first_present(rows, ("sequence_id", "Sequence", "sequence", "seq"))
    time_column = _first_present(rows, ("time_s", "Timestamp", "timestamp", "timestamp_s", "time"))
    if sequence_column is None or time_column is None:
        raise ValueError("template must contain sequence and timestamp columns")
    out = pd.DataFrame(
        {
            "sequence_id": rows[sequence_column].map(_template_sequence_or_none),
            "time_s": pd.to_numeric(rows[time_column], errors="coerce"),
        }
    )
    finite = out["sequence_id"].notna() & np.isfinite(out["time_s"].to_numpy(float))
    return out.loc[finite].sort_values(["sequence_id", "time_s"]).reset_index(drop=True)


def _template_sequence_or_none(value: Any) -> str | None:
    try:
        return parse_official_sequence_cell(value)
    except ValueError:
        return None


def _template_time_matches(values: pd.Series, target: float) -> np.ndarray:
    numeric = pd.to_numeric(values, errors="coerce").to_numpy(float)
    return np.isclose(numeric, float(target), rtol=0.0, atol=TEMPLATE_TIME_MATCH_ATOL_S)


def _finite_xyz(rows: pd.DataFrame) -> pd.Series:
    if rows.empty:
        return pd.Series(dtype=bool)
    xyz = rows[["state_x_m", "state_y_m", "state_z_m"]].apply(pd.to_numeric, errors="coerce")
    return pd.Series(np.isfinite(xyz.to_numpy(float)).all(axis=1), index=rows.index)


def _safe_mean_abs(values: pd.Series) -> float | None:
    numeric = pd.to_numeric(values, errors="coerce")
    numeric = numeric[np.isfinite(numeric.to_numpy(float))]
    if numeric.empty:
        return None
    return float(np.mean(np.abs(numeric.to_numpy(float))))


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


def _weighted_spread_m(xyz: np.ndarray, weights: np.ndarray, center: np.ndarray) -> float:
    if len(xyz) == 0:
        return np.nan
    distances = np.linalg.norm(xyz - center[None, :], axis=1)
    return float(np.sum(weights * distances) / np.sum(weights))


def _first_present(rows: pd.DataFrame, names: tuple[str, ...]) -> str | None:
    lower = {str(column).lower(): str(column) for column in rows.columns}
    for name in names:
        if name in rows.columns:
            return name
        found = lower.get(name.lower())
        if found is not None:
            return found
    return None


def _safe_label(value: str) -> str:
    label = str(value).strip().replace(" ", "_").replace("/", "_").replace("\\", "_")
    return label or "estimate"


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
