"""Segment-wise source gating for MMUAD Track 5 estimate ensembles.

Independent Track 5 pose pipelines can fail in different time intervals.  A
plain weighted mean keeps all sources active even when one source has a local
spike or temporarily implausible dynamics.  This module resamples each estimate
stream onto the official template and then uses a small dynamic program to choose
one source per timestamp, penalizing implausible local dynamics and unnecessary
source switches.  The result is still inference-safe: no truth values are used.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from raft_uav.mmuad.estimate_csv import read_estimate_csv
from raft_uav.mmuad.submission import (
    load_official_track5_template_file,
    load_sequence_class_map,
    validate_official_track5_submission,
    write_official_mmaud_results_csv,
    write_official_ug2_codabench_zip,
)
from raft_uav.mmuad.track5_estimate_ensemble import EstimateInput
from raft_uav.mmuad.track5_estimate_ensemble import apply_estimate_weight_config
from raft_uav.mmuad.track5_estimate_ensemble import load_estimate_weight_config
from raft_uav.mmuad.track5_estimate_ensemble import parse_estimate_spec
from raft_uav.mmuad.track5_template_resample import resample_estimates_to_track5_template

SEGMENT_GATE_ESTIMATES_CSV = "mmuad_track5_segment_source_gate_estimates.csv"
SEGMENT_GATE_DIAGNOSTICS_CSV = "mmuad_track5_segment_source_gate_diagnostics.csv"
SEGMENT_GATE_MANIFEST_JSON = "mmuad_track5_segment_source_gate_manifest.json"
VALIDATION_JSON = "mmuad_track5_segment_source_gate_validation.json"
VALIDATION_ROWS_CSV = "mmuad_track5_segment_source_gate_validation_rows.csv"
OFFICIAL_RESULTS_CSV = "mmaud_results.csv"
OFFICIAL_ZIP = "ug2_submission.zip"
TEMPLATE_TIME_MATCH_ATOL_S = 1.0e-9


@dataclass(frozen=True)
class SegmentSourceGateConfig:
    """Configuration for template-aligned segment source selection."""

    speed_limit_mps: float = 85.0
    acceleration_limit_mps2: float = 45.0
    switch_penalty: float = 5.0
    switch_jump_penalty_per_m: float = 0.02
    weight_log_scale: float = 1.0
    invalid_penalty: float = 1.0e12


def build_track5_segment_source_gate(
    estimate_inputs: Iterable[tuple[str, pd.DataFrame, float]],
    template: pd.DataFrame,
    *,
    config: SegmentSourceGateConfig | None = None,
    max_nearest_time_delta_s: float | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Choose one resampled estimate source per template row using DP gating."""

    config = config or SegmentSourceGateConfig()
    template_rows = _normalize_template_rows(template)
    loaded_inputs = list(estimate_inputs)
    if not loaded_inputs:
        raise ValueError("at least one estimate input is required")
    source_rows = _resample_sources(
        loaded_inputs,
        template_rows,
        max_nearest_time_delta_s=max_nearest_time_delta_s,
    )
    estimate_records: list[dict[str, Any]] = []
    diagnostic_records: list[dict[str, Any]] = []
    for sequence_id, sequence_template in template_rows.groupby("sequence_id", sort=True):
        sequence_result = _select_sequence_sources(
            sequence_id=str(sequence_id),
            sequence_template=sequence_template,
            source_rows=source_rows,
            config=config,
        )
        estimate_records.extend(sequence_result["estimates"])
        diagnostic_records.extend(sequence_result["diagnostics"])
    return pd.DataFrame.from_records(estimate_records), pd.DataFrame.from_records(diagnostic_records)


def write_track5_segment_source_gate_outputs(
    *,
    estimate_inputs: Iterable[EstimateInput],
    template: pd.DataFrame,
    output_dir: Path,
    class_map: dict[str, str] | None = None,
    default_classification: int | str = 0,
    config: SegmentSourceGateConfig | None = None,
    max_nearest_time_delta_s: float | None = None,
) -> dict[str, Path]:
    """Write segment-gated estimates, official CSV/ZIP, validation, and manifest."""

    estimate_input_list = list(estimate_inputs)
    loaded = [
        (item.label, read_estimate_csv(item.path), float(item.weight))
        for item in estimate_input_list
    ]
    config = config or SegmentSourceGateConfig()
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    estimates, diagnostics = build_track5_segment_source_gate(
        loaded,
        template,
        config=config,
        max_nearest_time_delta_s=max_nearest_time_delta_s,
    )
    paths = {
        "estimates_csv": output / SEGMENT_GATE_ESTIMATES_CSV,
        "diagnostics_csv": output / SEGMENT_GATE_DIAGNOSTICS_CSV,
        "official_results_csv": output / OFFICIAL_RESULTS_CSV,
        "official_zip": output / OFFICIAL_ZIP,
        "validation_json": output / VALIDATION_JSON,
        "validation_rows_csv": output / VALIDATION_ROWS_CSV,
        "manifest_json": output / SEGMENT_GATE_MANIFEST_JSON,
    }
    estimates.to_csv(paths["estimates_csv"], index=False)
    diagnostics.to_csv(paths["diagnostics_csv"], index=False)
    class_map = class_map or {}
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
        "schema": "raft-uav-mmuad-track5-segment-source-gate-v1",
        "estimate_inputs": [
            {"label": item.label, "path": str(item.path), "weight": float(item.weight)}
            for item in estimate_input_list
        ],
        "config": asdict(config),
        "row_count": int(len(estimates)),
        "sequence_count": int(estimates["sequence_id"].nunique()) if not estimates.empty else 0,
        "source_switch_count": int(diagnostics.get("source_switched", pd.Series()).sum()),
        "leaderboard_ready": bool(validation.summary.get("leaderboard_ready", False)),
        "codabench_upload_ready": bool(validation.summary.get("codabench_upload_ready", False)),
        "max_nearest_time_delta_s": max_nearest_time_delta_s,
        "paths": {name: str(path) for name, path in paths.items() if name != "manifest_json"},
    }
    paths["manifest_json"].write_text(json.dumps(_jsonable(manifest), indent=2), encoding="utf-8")
    return paths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="raft-uav-mmuad-track5-segment-source-gate",
        description="select Track 5 estimate sources by segment-wise dynamic gating",
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
    parser.add_argument("--weight-config", type=Path)
    parser.add_argument(
        "--weight-missing-policy",
        choices=("error", "keep", "zero"),
        default="error",
    )
    parser.add_argument("--max-nearest-time-delta-s", type=float)
    parser.add_argument("--speed-limit-mps", type=float, default=85.0)
    parser.add_argument("--acceleration-limit-mps2", type=float, default=45.0)
    parser.add_argument("--switch-penalty", type=float, default=5.0)
    parser.add_argument("--switch-jump-penalty-per-m", type=float, default=0.02)
    parser.add_argument("--weight-log-scale", type=float, default=1.0)
    parser.add_argument("--require-leaderboard-ready", action="store_true")
    args = parser.parse_args(argv)

    if not args.estimate_csv:
        parser.error("provide at least one --estimate-csv LABEL=PATH[@WEIGHT]")
    estimate_inputs = [parse_estimate_spec(spec) for spec in args.estimate_csv]
    if args.weight_config is not None:
        weights = load_estimate_weight_config(args.weight_config)
        estimate_inputs = apply_estimate_weight_config(
            estimate_inputs,
            weights,
            missing_policy=args.weight_missing_policy,
        )
    template = load_official_track5_template_file(args.template)
    class_map = load_sequence_class_map(args.class_map) if args.class_map is not None else {}
    config = SegmentSourceGateConfig(
        speed_limit_mps=float(args.speed_limit_mps),
        acceleration_limit_mps2=float(args.acceleration_limit_mps2),
        switch_penalty=float(args.switch_penalty),
        switch_jump_penalty_per_m=float(args.switch_jump_penalty_per_m),
        weight_log_scale=float(args.weight_log_scale),
    )
    paths = write_track5_segment_source_gate_outputs(
        estimate_inputs=estimate_inputs,
        template=template,
        output_dir=args.output_dir,
        class_map=class_map,
        default_classification=args.default_classification,
        config=config,
        max_nearest_time_delta_s=args.max_nearest_time_delta_s,
    )
    summary = json.loads(paths["validation_json"].read_text(encoding="utf-8"))
    print("mmuad_track5_segment_source_gate=ok")
    for name, path in paths.items():
        print(f"{name}={path}")
    print(f"leaderboard_ready={summary.get('leaderboard_ready')}")
    print(f"codabench_upload_ready={summary.get('codabench_upload_ready')}")
    if args.require_leaderboard_ready and not summary.get("leaderboard_ready", False):
        reasons = ", ".join(summary.get("leaderboard_blocking_reasons", [])) or "unknown"
        raise SystemExit(f"segment-gated upload is not leaderboard-ready: {reasons}")
    return 0


def _resample_sources(
    estimate_inputs: list[tuple[str, pd.DataFrame, float]],
    template_rows: pd.DataFrame,
    *,
    max_nearest_time_delta_s: float | None,
) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    seen: set[str] = set()
    for raw_label, estimates, raw_weight in estimate_inputs:
        label = _safe_label(raw_label)
        if label in seen:
            raise ValueError(f"duplicate estimate label after normalization: {label}")
        seen.add(label)
        weight = float(raw_weight)
        if not np.isfinite(weight) or weight < 0.0:
            raise ValueError(f"estimate weight must be finite and non-negative for {label}")
        resampled, _ = resample_estimates_to_track5_template(
            estimates,
            template_rows,
            max_nearest_time_delta_s=max_nearest_time_delta_s,
        )
        part = resampled[["sequence_id", "time_s", "state_x_m", "state_y_m", "state_z_m"]].copy()
        part["source_label"] = label
        part["source_weight"] = weight
        part["source_valid"] = _finite_xyz(part) & resampled.get(
            "template_resample_valid",
            pd.Series(True, index=resampled.index),
        ).astype(bool)
        parts.append(part)
    if not parts:
        raise ValueError("at least one estimate input is required")
    return pd.concat(parts, ignore_index=True, sort=False)


def _select_sequence_sources(
    *,
    sequence_id: str,
    sequence_template: pd.DataFrame,
    source_rows: pd.DataFrame,
    config: SegmentSourceGateConfig,
) -> dict[str, list[dict[str, Any]]]:
    times = sequence_template["time_s"].to_numpy(float)
    labels = sorted(source_rows["source_label"].astype(str).unique())
    if not labels:
        raise ValueError("no estimate sources available")
    positions = np.full((len(times), len(labels), 3), np.nan, dtype=float)
    valid = np.zeros((len(times), len(labels)), dtype=bool)
    weights = np.zeros(len(labels), dtype=float)
    sequence_sources = source_rows.loc[source_rows["sequence_id"].astype(str) == sequence_id]
    for label_index, label in enumerate(labels):
        label_rows = sequence_sources.loc[sequence_sources["source_label"].astype(str) == label]
        if label_rows.empty:
            continue
        weights[label_index] = float(label_rows["source_weight"].iloc[0])
        for time_index, time_s in enumerate(times):
            rows = label_rows.loc[_time_matches(label_rows["time_s"], float(time_s))]
            if rows.empty:
                continue
            row = rows.iloc[0]
            positions[time_index, label_index, :] = [
                float(row["state_x_m"]),
                float(row["state_y_m"]),
                float(row["state_z_m"]),
            ]
            valid[time_index, label_index] = bool(row.get("source_valid", True))
    emissions = _emission_costs(times, positions, valid, weights, config=config)
    path, path_cost = _viterbi_source_path(times, positions, emissions, config=config)
    estimates: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    for time_index, label_index in enumerate(path):
        xyz = positions[time_index, label_index]
        label = labels[int(label_index)]
        switched = time_index > 0 and int(path[time_index - 1]) != int(label_index)
        estimates.append(
            {
                "sequence_id": sequence_id,
                "time_s": float(times[time_index]),
                "state_x_m": float(xyz[0]) if np.isfinite(xyz[0]) else np.nan,
                "state_y_m": float(xyz[1]) if np.isfinite(xyz[1]) else np.nan,
                "state_z_m": float(xyz[2]) if np.isfinite(xyz[2]) else np.nan,
                "track5_segment_source_gate": True,
                "selected_source_label": label,
                "selected_source_weight": float(weights[int(label_index)]),
                "segment_gate_path_cost": float(path_cost[time_index, label_index]),
            }
        )
        diagnostics.append(
            {
                "sequence_id": sequence_id,
                "time_s": float(times[time_index]),
                "selected_source_label": label,
                "selected_source_weight": float(weights[int(label_index)]),
                "selected_emission_cost": float(emissions[time_index, label_index]),
                "path_cost": float(path_cost[time_index, label_index]),
                "source_switched": bool(switched),
                "valid_source_count": int(valid[time_index].sum()),
            }
        )
    return {"estimates": estimates, "diagnostics": diagnostics}


def _emission_costs(
    times: np.ndarray,
    positions: np.ndarray,
    valid: np.ndarray,
    weights: np.ndarray,
    *,
    config: SegmentSourceGateConfig,
) -> np.ndarray:
    count, source_count, _ = positions.shape
    emissions = np.full((count, source_count), float(config.invalid_penalty), dtype=float)
    safe_weights = np.maximum(weights, 0.0)
    positive_weights = safe_weights[safe_weights > 0.0]
    weight_floor = float(np.min(positive_weights)) if len(positive_weights) else 1.0
    safe_weights = np.where(safe_weights > 0.0, safe_weights, weight_floor * 1.0e-6)
    weight_prior = -float(config.weight_log_scale) * np.log(safe_weights / float(np.max(safe_weights)))
    for source_index in range(source_count):
        source_positions = positions[:, source_index, :]
        source_valid = valid[:, source_index]
        speed_penalty = _local_speed_penalty(
            times,
            source_positions,
            source_valid,
            speed_limit_mps=float(config.speed_limit_mps),
        )
        accel_penalty = _local_acceleration_penalty(
            times,
            source_positions,
            source_valid,
            acceleration_limit_mps2=float(config.acceleration_limit_mps2),
        )
        source_cost = weight_prior[source_index] + speed_penalty + accel_penalty
        emissions[:, source_index] = np.where(
            source_valid,
            source_cost,
            float(config.invalid_penalty),
        )
    return emissions


def _viterbi_source_path(
    times: np.ndarray,
    positions: np.ndarray,
    emissions: np.ndarray,
    *,
    config: SegmentSourceGateConfig,
) -> tuple[np.ndarray, np.ndarray]:
    count, source_count = emissions.shape
    path_cost = np.full((count, source_count), np.inf, dtype=float)
    back = np.zeros((count, source_count), dtype=int)
    path_cost[0] = emissions[0]
    for time_index in range(1, count):
        previous_positions = positions[time_index - 1]
        current_positions = positions[time_index]
        for current in range(source_count):
            transition = np.zeros(source_count, dtype=float)
            if source_count > 1:
                switch = np.arange(source_count) != current
                jump = np.linalg.norm(previous_positions - current_positions[current], axis=1)
                jump = np.nan_to_num(jump, nan=float(config.invalid_penalty))
                transition = np.where(
                    switch,
                    float(config.switch_penalty)
                    + float(config.switch_jump_penalty_per_m) * jump,
                    0.0,
                )
            candidates = path_cost[time_index - 1] + transition
            previous = int(np.argmin(candidates))
            path_cost[time_index, current] = candidates[previous] + emissions[time_index, current]
            back[time_index, current] = previous
    path = np.zeros(count, dtype=int)
    path[-1] = int(np.argmin(path_cost[-1]))
    for time_index in range(count - 1, 0, -1):
        path[time_index - 1] = back[time_index, path[time_index]]
    return path, path_cost


def _local_speed_penalty(
    times: np.ndarray,
    positions: np.ndarray,
    valid: np.ndarray,
    *,
    speed_limit_mps: float,
) -> np.ndarray:
    penalty = np.zeros(len(times), dtype=float)
    if len(times) < 2 or speed_limit_mps <= 0.0:
        return penalty
    for index in range(1, len(times)):
        if not valid[index] or not valid[index - 1]:
            continue
        dt = max(float(times[index] - times[index - 1]), 1.0e-9)
        speed = float(np.linalg.norm(positions[index] - positions[index - 1]) / dt)
        excess = max(0.0, speed - float(speed_limit_mps)) / float(speed_limit_mps)
        local = excess * excess
        penalty[index] += local
        penalty[index - 1] += 0.5 * local
    return penalty


def _local_acceleration_penalty(
    times: np.ndarray,
    positions: np.ndarray,
    valid: np.ndarray,
    *,
    acceleration_limit_mps2: float,
) -> np.ndarray:
    penalty = np.zeros(len(times), dtype=float)
    if len(times) < 3 or acceleration_limit_mps2 <= 0.0:
        return penalty
    for index in range(1, len(times) - 1):
        if not (valid[index - 1] and valid[index] and valid[index + 1]):
            continue
        left_dt = max(float(times[index] - times[index - 1]), 1.0e-9)
        right_dt = max(float(times[index + 1] - times[index]), 1.0e-9)
        left_velocity = (positions[index] - positions[index - 1]) / left_dt
        right_velocity = (positions[index + 1] - positions[index]) / right_dt
        dt = max(0.5 * (left_dt + right_dt), 1.0e-9)
        acceleration = float(np.linalg.norm((right_velocity - left_velocity) / dt))
        excess = max(0.0, acceleration - float(acceleration_limit_mps2))
        penalty[index] += (excess / float(acceleration_limit_mps2)) ** 2
    return penalty


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


def _time_matches(values: pd.Series, time_s: float) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce").to_numpy(float)
    return pd.Series(
        np.isclose(numeric, float(time_s), rtol=0.0, atol=TEMPLATE_TIME_MATCH_ATOL_S),
        index=values.index,
    )


def _finite_xyz(rows: pd.DataFrame) -> pd.Series:
    if rows.empty:
        return pd.Series(dtype=bool)
    xyz = rows[["state_x_m", "state_y_m", "state_z_m"]].apply(pd.to_numeric, errors="coerce")
    return pd.Series(np.isfinite(xyz.to_numpy(float)).all(axis=1), index=rows.index)


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
