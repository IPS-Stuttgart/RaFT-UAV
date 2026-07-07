"""Agreement-gated Track 5 estimate ensemble for MMUAD submissions.

Weighted averaging can improve leaderboard submissions when independent pose
pipelines agree, but it can also hurt badly when one trajectory drifts.  This
module resamples estimate trajectories onto the official Track 5 template and
uses a simple inference-safe gate: average rows whose inputs agree, otherwise
fall back to a designated primary trajectory (or the highest-weight valid input).

Weights and the gate should be selected upstream on train folds for leaderboard
use.  The apply path never uses truth values.
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
    parse_official_sequence_cell,
    validate_official_track5_submission,
    write_official_mmaud_results_csv,
    write_official_ug2_codabench_zip,
)
from raft_uav.mmuad.track5_estimate_ensemble import EstimateInput, parse_estimate_spec
from raft_uav.mmuad.track5_template_resample import resample_estimates_to_track5_template

GATED_ESTIMATES_CSV = "mmuad_track5_agreement_gated_estimates.csv"
GATED_DIAGNOSTICS_CSV = "mmuad_track5_agreement_gated_diagnostics.csv"
GATED_MANIFEST_JSON = "mmuad_track5_agreement_gated_manifest.json"
VALIDATION_JSON = "mmuad_track5_agreement_gated_validation.json"
VALIDATION_ROWS_CSV = "mmuad_track5_agreement_gated_validation_rows.csv"
OFFICIAL_RESULTS_CSV = "mmaud_results.csv"
OFFICIAL_ZIP = "ug2_submission.zip"
TEMPLATE_TIME_MATCH_ATOL_S = 1.0e-9


def build_agreement_gated_track5_ensemble(
    estimate_inputs: Iterable[tuple[str, pd.DataFrame, float]],
    template: pd.DataFrame,
    *,
    spread_gate_m: float = 10.0,
    primary_label: str | None = None,
    max_nearest_time_delta_s: float | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return agreement-gated ensemble estimates and diagnostics.

    For each official template timestamp, finite valid input trajectories are
    resampled to the template.  If the weighted spread around the weighted mean is
    no larger than ``spread_gate_m``, the weighted mean is used.  If the inputs
    disagree, the configured primary trajectory is used when valid; otherwise the
    highest-weight valid input is used.
    """

    gate = float(spread_gate_m)
    if not np.isfinite(gate) or gate < 0.0:
        raise ValueError("spread_gate_m must be finite and non-negative")
    template_rows = _normalize_template_rows(template)
    inputs = list(estimate_inputs)
    if not inputs:
        raise ValueError("at least one estimate input is required")
    primary = _safe_label(primary_label) if primary_label else _safe_label(inputs[0][0])
    parts: list[pd.DataFrame] = []
    input_summaries: list[dict[str, Any]] = []
    for label, estimates, weight in inputs:
        label = _safe_label(label)
        weight = _validate_weight(weight, label=label)
        resampled, diagnostics = resample_estimates_to_track5_template(
            estimates,
            template_rows,
            max_nearest_time_delta_s=max_nearest_time_delta_s,
        )
        rows = resampled[["sequence_id", "time_s", "state_x_m", "state_y_m", "state_z_m"]].copy()
        rows["ensemble_label"] = label
        rows["ensemble_weight"] = weight
        rows["ensemble_valid"] = _finite_xyz(rows) & resampled.get(
            "template_resample_valid",
            pd.Series(True, index=resampled.index),
        ).astype(bool)
        parts.append(rows)
        input_summaries.append(
            {
                "label": label,
                "weight": weight,
                "input_estimate_rows": int(len(estimates)),
                "valid_resampled_rows": int(rows["ensemble_valid"].sum()),
                "mean_abs_nearest_time_delta_s": _safe_mean_abs(
                    diagnostics.get("nearest_time_delta_s", pd.Series(dtype=float))
                ),
            }
        )
    stacked = pd.concat(parts, ignore_index=True, sort=False)
    estimate_records: list[dict[str, Any]] = []
    diagnostic_records: list[dict[str, Any]] = []
    for _, template_row in template_rows.iterrows():
        sequence_id = str(template_row["sequence_id"])
        time_s = float(template_row["time_s"])
        frame_rows = stacked.loc[
            (stacked["sequence_id"].astype(str) == sequence_id)
            & _template_time_matches(stacked["time_s"], time_s)
        ]
        valid = frame_rows.loc[frame_rows["ensemble_valid"].astype(bool) & (frame_rows["ensemble_weight"] > 0.0)]
        if valid.empty:
            xyz = np.asarray([np.nan, np.nan, np.nan], dtype=float)
            action = "missing"
            spread = np.nan
            labels = ""
            weight_sum = 0.0
            selected_label = ""
        else:
            weights = valid["ensemble_weight"].to_numpy(float)
            xyz_values = valid[["state_x_m", "state_y_m", "state_z_m"]].to_numpy(float)
            weight_sum = float(np.sum(weights))
            mean_xyz = np.sum(weights[:, None] * xyz_values, axis=0) / weight_sum
            spread = _weighted_spread_m(xyz_values, weights, mean_xyz)
            labels = ";".join(valid["ensemble_label"].astype(str).tolist())
            if len(valid) == 1 or spread <= gate:
                xyz = mean_xyz
                action = "weighted_mean"
                selected_label = ""
            else:
                chosen = _fallback_row(valid, primary_label=primary)
                xyz = chosen[["state_x_m", "state_y_m", "state_z_m"]].to_numpy(float)
                action = "primary_fallback" if str(chosen["ensemble_label"]) == primary else "highest_weight_fallback"
                selected_label = str(chosen["ensemble_label"])
        estimate_records.append(
            {
                "sequence_id": sequence_id,
                "time_s": time_s,
                "state_x_m": float(xyz[0]) if np.isfinite(xyz[0]) else np.nan,
                "state_y_m": float(xyz[1]) if np.isfinite(xyz[1]) else np.nan,
                "state_z_m": float(xyz[2]) if np.isfinite(xyz[2]) else np.nan,
                "agreement_gated_ensemble": True,
                "agreement_gate_m": gate,
                "agreement_action": action,
                "agreement_selected_label": selected_label,
                "ensemble_source_count": int(len(valid)),
                "ensemble_weight_sum": weight_sum,
                "ensemble_labels": labels,
                "ensemble_position_spread_m": spread,
            }
        )
        diagnostic_records.append(
            {
                "sequence_id": sequence_id,
                "time_s": time_s,
                "candidate_input_count": int(len(frame_rows)),
                "valid_input_count": int(len(valid)),
                "weight_sum": weight_sum,
                "labels": labels,
                "position_spread_m": spread,
                "spread_gate_m": gate,
                "agreement_action": action,
                "agreement_selected_label": selected_label,
            }
        )
    estimates = pd.DataFrame.from_records(estimate_records)
    diagnostics = pd.DataFrame.from_records(diagnostic_records)
    diagnostics.attrs["input_summaries"] = input_summaries
    return estimates, diagnostics


def write_agreement_gated_track5_ensemble_outputs(
    *,
    estimate_inputs: Iterable[EstimateInput],
    template: pd.DataFrame,
    output_dir: Path,
    class_map: dict[str, str] | None = None,
    default_classification: int | str = 0,
    spread_gate_m: float = 10.0,
    primary_label: str | None = None,
    max_nearest_time_delta_s: float | None = None,
) -> dict[str, Path]:
    """Write agreement-gated estimates, official artifacts, validation, and manifest."""

    input_list = list(estimate_inputs)
    loaded_inputs = [(item.label, pd.read_csv(item.path), float(item.weight)) for item in input_list]
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    estimates, diagnostics = build_agreement_gated_track5_ensemble(
        loaded_inputs,
        template,
        spread_gate_m=spread_gate_m,
        primary_label=primary_label,
        max_nearest_time_delta_s=max_nearest_time_delta_s,
    )
    paths = {
        "estimates_csv": output / GATED_ESTIMATES_CSV,
        "diagnostics_csv": output / GATED_DIAGNOSTICS_CSV,
        "official_results_csv": output / OFFICIAL_RESULTS_CSV,
        "official_zip": output / OFFICIAL_ZIP,
        "validation_json": output / VALIDATION_JSON,
        "validation_rows_csv": output / VALIDATION_ROWS_CSV,
        "manifest_json": output / GATED_MANIFEST_JSON,
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
    paths["validation_json"].write_text(json.dumps(_jsonable(validation.summary), indent=2), encoding="utf-8")
    validation.rows.to_csv(paths["validation_rows_csv"], index=False)
    action_counts = diagnostics.get("agreement_action", pd.Series(dtype=str)).value_counts().to_dict()
    manifest = {
        "schema": "raft-uav-mmuad-track5-agreement-gated-ensemble-v1",
        "estimate_inputs": [
            {"label": item.label, "path": str(item.path), "weight": float(item.weight)}
            for item in input_list
        ],
        "input_summaries": diagnostics.attrs.get("input_summaries", []),
        "spread_gate_m": float(spread_gate_m),
        "primary_label": _safe_label(primary_label) if primary_label else _safe_label(input_list[0].label),
        "row_count": int(len(estimates)),
        "valid_ensemble_rows": int(_finite_xyz(estimates).sum()),
        "agreement_action_counts": {str(k): int(v) for k, v in action_counts.items()},
        "mean_position_spread_m": _safe_mean(diagnostics.get("position_spread_m", pd.Series(dtype=float))),
        "p95_position_spread_m": _safe_percentile(diagnostics.get("position_spread_m", pd.Series(dtype=float)), 95),
        "leaderboard_ready": bool(validation.summary.get("leaderboard_ready", False)),
        "codabench_upload_ready": bool(validation.summary.get("codabench_upload_ready", False)),
        "paths": {name: str(path) for name, path in paths.items() if name != "manifest_json"},
    }
    paths["manifest_json"].write_text(json.dumps(_jsonable(manifest), indent=2), encoding="utf-8")
    return paths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="raft-uav-mmuad-track5-agreement-gated-ensemble",
        description="agreement-gated ensemble of Track 5 estimate trajectories",
    )
    parser.add_argument("--estimate-csv", action="append", default=[], metavar="LABEL=PATH[@WEIGHT]")
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--class-map", type=Path)
    parser.add_argument("--default-classification", default="0")
    parser.add_argument("--spread-gate-m", type=float, default=10.0)
    parser.add_argument("--primary-label")
    parser.add_argument("--max-nearest-time-delta-s", type=float)
    parser.add_argument("--require-leaderboard-ready", action="store_true")
    args = parser.parse_args(argv)

    if not args.estimate_csv:
        parser.error("provide at least one --estimate-csv LABEL=PATH[@WEIGHT]")
    estimate_inputs = [parse_estimate_spec(spec) for spec in args.estimate_csv]
    template = load_official_track5_template_file(args.template)
    class_map = load_sequence_class_map(args.class_map) if args.class_map is not None else {}
    paths = write_agreement_gated_track5_ensemble_outputs(
        estimate_inputs=estimate_inputs,
        template=template,
        output_dir=args.output_dir,
        class_map=class_map,
        default_classification=args.default_classification,
        spread_gate_m=float(args.spread_gate_m),
        primary_label=args.primary_label,
        max_nearest_time_delta_s=args.max_nearest_time_delta_s,
    )
    summary = json.loads(paths["validation_json"].read_text(encoding="utf-8"))
    print("mmuad_track5_agreement_gated_ensemble=ok")
    for name, path in paths.items():
        print(f"{name}={path}")
    print(f"leaderboard_ready={summary.get('leaderboard_ready')}")
    print(f"codabench_upload_ready={summary.get('codabench_upload_ready')}")
    if args.require_leaderboard_ready and not summary.get("leaderboard_ready", False):
        reasons = ", ".join(summary.get("leaderboard_blocking_reasons", [])) or "unknown"
        raise SystemExit(f"agreement-gated ensemble is not leaderboard-ready: {reasons}")
    return 0


def _fallback_row(valid: pd.DataFrame, *, primary_label: str) -> pd.Series:
    primary = valid.loc[valid["ensemble_label"].astype(str) == primary_label]
    if not primary.empty:
        return primary.sort_values("ensemble_weight", ascending=False).iloc[0]
    return valid.sort_values("ensemble_weight", ascending=False).iloc[0]


def _weighted_spread_m(xyz: np.ndarray, weights: np.ndarray, center: np.ndarray) -> float:
    distances = np.linalg.norm(xyz - center[None, :], axis=1)
    return float(np.sum(weights * distances) / np.sum(weights))


def _validate_weight(weight: float, *, label: str) -> float:
    value = float(weight)
    if not np.isfinite(value) or value < 0.0:
        raise ValueError(f"estimate weight must be finite and non-negative for {label}: {value}")
    return value


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


def _first_present(rows: pd.DataFrame, names: tuple[str, ...]) -> str | None:
    lower = {str(column).lower(): str(column) for column in rows.columns}
    for name in names:
        if name in rows.columns:
            return name
        found = lower.get(name.lower())
        if found is not None:
            return found
    return None


def _safe_label(value: object) -> str:
    return ("" if value is None else str(value)).strip().replace(" ", "_").replace("/", "_").replace("\\", "_")


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
