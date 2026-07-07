"""Consensus-weighted Track 5 estimate ensembling for MMUAD/Codabench.

The plain estimate ensemble uses fixed weights for each pose pipeline.  This
module adds an inference-safe adaptive variant: after resampling every estimate
trajectory onto the official Track 5 template, it builds a robust consensus
center per timestamp and downweights trajectories that are far from that
consensus.  This is useful for leaderboard runs where one pose branch has
occasional large outliers but remains valuable elsewhere.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
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
from raft_uav.mmuad.track5_estimate_ensemble import EstimateInput, parse_estimate_spec
from raft_uav.mmuad.track5_template_resample import resample_estimates_to_track5_template

ENSEMBLED_ESTIMATES_CSV = "mmuad_track5_consensus_ensemble_estimates.csv"
ENSEMBLE_DIAGNOSTICS_CSV = "mmuad_track5_consensus_ensemble_diagnostics.csv"
ENSEMBLE_MANIFEST_JSON = "mmuad_track5_consensus_ensemble_manifest.json"
VALIDATION_JSON = "mmuad_track5_consensus_ensemble_validation.json"
VALIDATION_ROWS_CSV = "mmuad_track5_consensus_ensemble_validation_rows.csv"
OFFICIAL_RESULTS_CSV = "mmaud_results.csv"
OFFICIAL_ZIP = "ug2_submission.zip"
CENTER_POLICIES = ("weighted-median", "weighted-mean")
ADAPTIVE_WEIGHT_POLICIES = ("inverse-distance", "gaussian")


def build_consensus_estimate_ensemble(
    estimate_inputs: Iterable[tuple[str, pd.DataFrame, float]],
    template: pd.DataFrame,
    *,
    center_policy: str = "weighted-median",
    adaptive_weight_policy: str = "inverse-distance",
    distance_floor_m: float = 1.0,
    distance_power: float = 1.0,
    gaussian_scale_m: float = 10.0,
    max_nearest_time_delta_s: float | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return consensus-weighted estimates and diagnostics.

    All estimate trajectories are first interpolated to the official template.
    For each requested timestamp, a robust consensus center is computed.  Input
    weights are then adaptively scaled by each trajectory's distance from that
    center before a final weighted mean is formed.
    """

    if center_policy not in CENTER_POLICIES:
        raise ValueError(f"center_policy must be one of: {', '.join(CENTER_POLICIES)}")
    if adaptive_weight_policy not in ADAPTIVE_WEIGHT_POLICIES:
        raise ValueError(
            f"adaptive_weight_policy must be one of: {', '.join(ADAPTIVE_WEIGHT_POLICIES)}"
        )
    if distance_floor_m <= 0.0 or not np.isfinite(distance_floor_m):
        raise ValueError("distance_floor_m must be positive and finite")
    if distance_power <= 0.0 or not np.isfinite(distance_power):
        raise ValueError("distance_power must be positive and finite")
    if gaussian_scale_m <= 0.0 or not np.isfinite(gaussian_scale_m):
        raise ValueError("gaussian_scale_m must be positive and finite")

    template_rows = _normalize_template_rows(template)
    loaded: list[pd.DataFrame] = []
    input_summaries: list[dict[str, Any]] = []
    for label, estimates, weight in estimate_inputs:
        label = _safe_label(label)
        weight = _validate_weight(float(weight), label=label)
        resampled, diagnostics = resample_estimates_to_track5_template(
            estimates,
            template_rows,
            max_nearest_time_delta_s=max_nearest_time_delta_s,
        )
        part = resampled[["sequence_id", "time_s", "state_x_m", "state_y_m", "state_z_m"]].copy()
        part["ensemble_label"] = label
        part["base_weight"] = weight
        part["valid"] = _finite_xyz(part) & resampled.get(
            "template_resample_valid",
            pd.Series(True, index=resampled.index),
        ).astype(bool)
        loaded.append(part)
        input_summaries.append(
            {
                "label": label,
                "weight": weight,
                "input_estimate_rows": int(len(estimates)),
                "template_rows": int(len(template_rows)),
                "valid_resampled_rows": int(part["valid"].sum()),
                "mean_abs_nearest_time_delta_s": _safe_mean_abs(
                    diagnostics.get("nearest_time_delta_s", pd.Series(dtype=float))
                ),
            }
        )
    if not loaded:
        raise ValueError("at least one estimate input is required")
    stacked = pd.concat(loaded, ignore_index=True, sort=False)
    records: list[dict[str, Any]] = []
    diagnostics_records: list[dict[str, Any]] = []
    for _, template_row in template_rows.iterrows():
        sequence_id = str(template_row["sequence_id"])
        time_s = float(template_row["time_s"])
        group = stacked.loc[
            (stacked["sequence_id"].astype(str) == sequence_id)
            & _time_matches(stacked["time_s"], time_s)
        ]
        valid = group.loc[group["valid"].astype(bool) & (group["base_weight"] > 0.0)].copy()
        if valid.empty:
            xyz = np.asarray([np.nan, np.nan, np.nan], dtype=float)
            center = np.asarray([np.nan, np.nan, np.nan], dtype=float)
            spread = np.nan
            mean_consensus_distance = np.nan
            weight_sum = 0.0
            labels = ""
        else:
            xyz_values = valid[["state_x_m", "state_y_m", "state_z_m"]].to_numpy(float)
            base_weights = valid["base_weight"].to_numpy(float)
            center = _consensus_center(xyz_values, base_weights, policy=center_policy)
            distances = np.linalg.norm(xyz_values - center[None, :], axis=1)
            adaptive_weights = _adaptive_weights(
                base_weights,
                distances,
                policy=adaptive_weight_policy,
                distance_floor_m=distance_floor_m,
                distance_power=distance_power,
                gaussian_scale_m=gaussian_scale_m,
            )
            weight_sum = float(np.sum(adaptive_weights))
            xyz = np.sum(adaptive_weights[:, None] * xyz_values, axis=0) / weight_sum
            spread = _weighted_spread_m(xyz_values, adaptive_weights, xyz)
            mean_consensus_distance = float(np.sum(base_weights * distances) / np.sum(base_weights))
            labels = ";".join(valid["ensemble_label"].astype(str).tolist())
            valid["consensus_distance_m"] = distances
            valid["adaptive_weight"] = adaptive_weights
        records.append(
            {
                "sequence_id": sequence_id,
                "time_s": time_s,
                "state_x_m": float(xyz[0]) if np.isfinite(xyz[0]) else np.nan,
                "state_y_m": float(xyz[1]) if np.isfinite(xyz[1]) else np.nan,
                "state_z_m": float(xyz[2]) if np.isfinite(xyz[2]) else np.nan,
                "track5_consensus_ensemble": True,
                "ensemble_source_count": int(len(valid)),
                "adaptive_weight_sum": weight_sum,
                "ensemble_labels": labels,
                "consensus_center_x_m": float(center[0]) if np.isfinite(center[0]) else np.nan,
                "consensus_center_y_m": float(center[1]) if np.isfinite(center[1]) else np.nan,
                "consensus_center_z_m": float(center[2]) if np.isfinite(center[2]) else np.nan,
                "ensemble_position_spread_m": spread,
                "mean_consensus_distance_m": mean_consensus_distance,
            }
        )
        diagnostics_records.append(
            {
                "sequence_id": sequence_id,
                "time_s": time_s,
                "candidate_input_count": int(len(group)),
                "valid_input_count": int(len(valid)),
                "adaptive_weight_sum": weight_sum,
                "labels": labels,
                "position_spread_m": spread,
                "mean_consensus_distance_m": mean_consensus_distance,
                "max_consensus_distance_m": float(valid["consensus_distance_m"].max())
                if not valid.empty and "consensus_distance_m" in valid
                else np.nan,
            }
        )
    ensemble = pd.DataFrame.from_records(records)
    diagnostics = pd.DataFrame.from_records(diagnostics_records)
    diagnostics.attrs["input_summaries"] = input_summaries
    return ensemble, diagnostics


def write_consensus_estimate_ensemble_outputs(
    *,
    estimate_inputs: Iterable[EstimateInput],
    template: pd.DataFrame,
    output_dir: Path,
    class_map: dict[str, str] | None = None,
    default_classification: int | str = 0,
    center_policy: str = "weighted-median",
    adaptive_weight_policy: str = "inverse-distance",
    distance_floor_m: float = 1.0,
    distance_power: float = 1.0,
    gaussian_scale_m: float = 10.0,
    max_nearest_time_delta_s: float | None = None,
) -> dict[str, Path]:
    """Write consensus ensemble estimates, official CSV/ZIP, and validation artifacts."""

    input_list = list(estimate_inputs)
    loaded_inputs = [(item.label, pd.read_csv(item.path), float(item.weight)) for item in input_list]
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    ensemble, diagnostics = build_consensus_estimate_ensemble(
        loaded_inputs,
        template,
        center_policy=center_policy,
        adaptive_weight_policy=adaptive_weight_policy,
        distance_floor_m=distance_floor_m,
        distance_power=distance_power,
        gaussian_scale_m=gaussian_scale_m,
        max_nearest_time_delta_s=max_nearest_time_delta_s,
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
    write_official_mmaud_results_csv(
        ensemble,
        paths["official_results_csv"],
        classification=default_classification,
        class_map=class_map or {},
        invalid_row_policy="raise",
    )
    write_official_ug2_codabench_zip(
        ensemble,
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
    manifest = {
        "schema": "raft-uav-mmuad-track5-consensus-estimate-ensemble-v1",
        "estimate_inputs": [asdict(item) | {"path": str(item.path)} for item in input_list],
        "input_summaries": diagnostics.attrs.get("input_summaries", []),
        "row_count": int(len(ensemble)),
        "valid_ensemble_rows": int(_finite_xyz(ensemble).sum()),
        "center_policy": center_policy,
        "adaptive_weight_policy": adaptive_weight_policy,
        "distance_floor_m": float(distance_floor_m),
        "distance_power": float(distance_power),
        "gaussian_scale_m": float(gaussian_scale_m),
        "max_nearest_time_delta_s": max_nearest_time_delta_s,
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
        prog="raft-uav-mmuad-track5-consensus-ensemble",
        description="consensus-weight MMUAD Track 5 estimate trajectories on an official template",
    )
    parser.add_argument("--estimate-csv", action="append", default=[], metavar="LABEL=PATH[@WEIGHT]")
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--class-map", type=Path)
    parser.add_argument("--default-classification", default="0")
    parser.add_argument("--center-policy", choices=CENTER_POLICIES, default="weighted-median")
    parser.add_argument(
        "--adaptive-weight-policy",
        choices=ADAPTIVE_WEIGHT_POLICIES,
        default="inverse-distance",
    )
    parser.add_argument("--distance-floor-m", type=float, default=1.0)
    parser.add_argument("--distance-power", type=float, default=1.0)
    parser.add_argument("--gaussian-scale-m", type=float, default=10.0)
    parser.add_argument("--max-nearest-time-delta-s", type=float)
    parser.add_argument("--require-leaderboard-ready", action="store_true")
    args = parser.parse_args(argv)

    if not args.estimate_csv:
        parser.error("provide at least one --estimate-csv LABEL=PATH[@WEIGHT]")
    estimate_inputs = [parse_estimate_spec(spec) for spec in args.estimate_csv]
    template = load_official_track5_template_file(args.template)
    class_map = load_sequence_class_map(args.class_map) if args.class_map is not None else {}
    paths = write_consensus_estimate_ensemble_outputs(
        estimate_inputs=estimate_inputs,
        template=template,
        output_dir=args.output_dir,
        class_map=class_map,
        default_classification=args.default_classification,
        center_policy=args.center_policy,
        adaptive_weight_policy=args.adaptive_weight_policy,
        distance_floor_m=float(args.distance_floor_m),
        distance_power=float(args.distance_power),
        gaussian_scale_m=float(args.gaussian_scale_m),
        max_nearest_time_delta_s=args.max_nearest_time_delta_s,
    )
    summary = json.loads(paths["validation_json"].read_text(encoding="utf-8"))
    print("mmuad_track5_consensus_estimate_ensemble=ok")
    for name, path in paths.items():
        print(f"{name}={path}")
    print(f"leaderboard_ready={summary.get('leaderboard_ready')}")
    print(f"codabench_upload_ready={summary.get('codabench_upload_ready')}")
    if args.require_leaderboard_ready and not summary.get("leaderboard_ready", False):
        reasons = ", ".join(summary.get("leaderboard_blocking_reasons", [])) or "unknown"
        raise SystemExit(f"consensus ensemble upload is not leaderboard-ready: {reasons}")
    return 0


def _consensus_center(xyz: np.ndarray, weights: np.ndarray, *, policy: str) -> np.ndarray:
    if policy == "weighted-mean":
        return np.sum(weights[:, None] * xyz, axis=0) / float(np.sum(weights))
    if policy == "weighted-median":
        return np.asarray([_weighted_median(xyz[:, axis], weights) for axis in range(3)])
    raise ValueError(f"unsupported center policy: {policy}")


def _adaptive_weights(
    base_weights: np.ndarray,
    distances: np.ndarray,
    *,
    policy: str,
    distance_floor_m: float,
    distance_power: float,
    gaussian_scale_m: float,
) -> np.ndarray:
    if policy == "inverse-distance":
        return base_weights / np.maximum(distances, distance_floor_m) ** distance_power
    if policy == "gaussian":
        return base_weights * np.exp(-0.5 * (distances / gaussian_scale_m) ** 2)
    raise ValueError(f"unsupported adaptive weight policy: {policy}")


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values)
    sorted_values = values[order]
    sorted_weights = weights[order]
    cumulative = np.cumsum(sorted_weights)
    cutoff = 0.5 * float(np.sum(sorted_weights))
    return float(sorted_values[int(np.searchsorted(cumulative, cutoff, side="left"))])


def _normalize_template_rows(template: pd.DataFrame) -> pd.DataFrame:
    rows = pd.DataFrame(template).copy()
    sequence_column = _first_present(rows, ("sequence_id", "Sequence", "sequence", "seq"))
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


def _time_matches(values: pd.Series, target: float) -> np.ndarray:
    numeric = pd.to_numeric(values, errors="coerce").to_numpy(float)
    return np.isclose(numeric, float(target), rtol=0.0, atol=1.0e-9)


def _finite_xyz(rows: pd.DataFrame) -> pd.Series:
    if rows.empty:
        return pd.Series(dtype=bool)
    xyz = rows[["state_x_m", "state_y_m", "state_z_m"]].apply(pd.to_numeric, errors="coerce")
    return pd.Series(np.isfinite(xyz.to_numpy(float)).all(axis=1), index=rows.index)


def _weighted_spread_m(xyz: np.ndarray, weights: np.ndarray, center: np.ndarray) -> float:
    distances = np.linalg.norm(xyz - center[None, :], axis=1)
    return float(np.sum(weights * distances) / np.sum(weights))


def _validate_weight(weight: float, *, label: str) -> float:
    if not np.isfinite(weight) or weight < 0.0:
        raise ValueError(f"estimate weight must be finite and non-negative for {label}: {weight}")
    return float(weight)


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
