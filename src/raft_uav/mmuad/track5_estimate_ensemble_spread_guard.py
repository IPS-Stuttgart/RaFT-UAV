"""Spread-guarded estimate ensembling for MMUAD Track 5 submissions.

Averaging independently generated pose trajectories can help on normal frames but
can hurt badly when one branch is an outlier.  This inference-safe helper first
resamples each estimate trajectory to the official Track 5 template, computes the
weighted ensemble spread per requested row, and falls back to a trusted estimate
when the disagreement exceeds a configured threshold.
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

SPREAD_GUARD_ESTIMATES_CSV = "mmuad_track5_spread_guard_estimates.csv"
SPREAD_GUARD_DIAGNOSTICS_CSV = "mmuad_track5_spread_guard_diagnostics.csv"
SPREAD_GUARD_MANIFEST_JSON = "mmuad_track5_spread_guard_manifest.json"
SPREAD_GUARD_VALIDATION_JSON = "mmuad_track5_spread_guard_validation.json"
SPREAD_GUARD_VALIDATION_ROWS_CSV = "mmuad_track5_spread_guard_validation_rows.csv"
OFFICIAL_RESULTS_CSV = "mmaud_results.csv"
OFFICIAL_ZIP = "ug2_submission.zip"
FALLBACK_POLICIES = ("max-weight", "first", "label")
TEMPLATE_TIME_MATCH_ATOL_S = 1.0e-9


def build_spread_guarded_estimate_ensemble(
    estimate_inputs: Iterable[tuple[str, pd.DataFrame, float]],
    template: pd.DataFrame,
    *,
    spread_threshold_m: float,
    fallback_policy: str = "max-weight",
    fallback_label: str | None = None,
    max_nearest_time_delta_s: float | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return spread-guarded Track 5 estimates and diagnostics.

    The normal output is the weighted mean over valid inputs.  If the weighted
    mean spread exceeds ``spread_threshold_m``, the output falls back to one
    resampled input selected by ``fallback_policy``.
    """

    threshold = float(spread_threshold_m)
    if not np.isfinite(threshold) or threshold < 0.0:
        raise ValueError("spread_threshold_m must be finite and non-negative")
    if fallback_policy not in FALLBACK_POLICIES:
        raise ValueError(f"fallback_policy must be one of: {', '.join(FALLBACK_POLICIES)}")
    if fallback_policy == "label" and not fallback_label:
        raise ValueError("fallback_label is required when fallback_policy='label'")
    template_rows = _normalize_template_rows(template)
    loaded_inputs = tuple(estimate_inputs)
    if not loaded_inputs:
        raise ValueError("at least one estimate input is required")

    resampled_parts: list[pd.DataFrame] = []
    input_order: list[str] = []
    for order, (label_text, estimates, weight) in enumerate(loaded_inputs):
        label = _safe_label(label_text)
        input_order.append(label)
        weight = _validate_weight(weight, label=label)
        resampled, _ = resample_estimates_to_track5_template(
            estimates,
            template_rows,
            max_nearest_time_delta_s=max_nearest_time_delta_s,
        )
        part = resampled[["sequence_id", "time_s", "state_x_m", "state_y_m", "state_z_m"]].copy()
        part["input_label"] = label
        part["input_order"] = int(order)
        part["input_weight"] = weight
        part["input_valid"] = _finite_xyz(part) & resampled.get(
            "template_resample_valid",
            pd.Series(True, index=resampled.index),
        ).astype(bool)
        resampled_parts.append(part)
    stacked = pd.concat(resampled_parts, ignore_index=True, sort=False)
    fallback_label = None if fallback_label is None else _safe_label(fallback_label)

    estimates_records: list[dict[str, Any]] = []
    diagnostic_records: list[dict[str, Any]] = []
    for _, template_row in template_rows.iterrows():
        sequence_id = str(template_row["sequence_id"])
        time_s = float(template_row["time_s"])
        rows = stacked.loc[
            (stacked["sequence_id"].astype(str) == sequence_id)
            & _template_time_matches(stacked["time_s"], time_s)
        ]
        valid = rows.loc[rows["input_valid"].astype(bool) & (rows["input_weight"] > 0.0)]
        if valid.empty:
            xyz = np.asarray([np.nan, np.nan, np.nan], dtype=float)
            weighted_xyz = xyz.copy()
            spread = np.nan
            chosen_label = ""
            guard_applied = False
        else:
            weights = valid["input_weight"].to_numpy(float)
            xyz_values = valid[["state_x_m", "state_y_m", "state_z_m"]].to_numpy(float)
            weighted_xyz = np.sum(weights[:, None] * xyz_values, axis=0) / float(np.sum(weights))
            spread = _weighted_spread_m(xyz_values, weights, weighted_xyz)
            guard_applied = bool(np.isfinite(spread) and spread > threshold)
            if guard_applied:
                chosen = _fallback_row(
                    valid,
                    policy=fallback_policy,
                    fallback_label=fallback_label,
                    input_order=input_order,
                )
                xyz = chosen[["state_x_m", "state_y_m", "state_z_m"]].to_numpy(float)
                chosen_label = str(chosen["input_label"])
            else:
                xyz = weighted_xyz
                chosen_label = "weighted-mean"
        estimates_records.append(
            {
                "sequence_id": sequence_id,
                "time_s": time_s,
                "source": "track5-spread-guard-ensemble",
                "track_id": "track5-spread-guard-ensemble",
                "state_x_m": float(xyz[0]) if np.isfinite(xyz[0]) else np.nan,
                "state_y_m": float(xyz[1]) if np.isfinite(xyz[1]) else np.nan,
                "state_z_m": float(xyz[2]) if np.isfinite(xyz[2]) else np.nan,
                "spread_guard_applied": bool(guard_applied),
                "spread_guard_threshold_m": threshold,
                "ensemble_position_spread_m": spread,
                "spread_guard_chosen_label": chosen_label,
            }
        )
        diagnostic_records.append(
            {
                "sequence_id": sequence_id,
                "time_s": time_s,
                "valid_input_count": int(len(valid)),
                "input_labels": ";".join(valid["input_label"].astype(str)) if not valid.empty else "",
                "weighted_x_m": float(weighted_xyz[0]) if np.isfinite(weighted_xyz[0]) else np.nan,
                "weighted_y_m": float(weighted_xyz[1]) if np.isfinite(weighted_xyz[1]) else np.nan,
                "weighted_z_m": float(weighted_xyz[2]) if np.isfinite(weighted_xyz[2]) else np.nan,
                "position_spread_m": spread,
                "spread_guard_applied": bool(guard_applied),
                "spread_guard_policy": fallback_policy,
                "spread_guard_chosen_label": chosen_label,
            }
        )
    estimates = pd.DataFrame.from_records(estimates_records)
    diagnostics = pd.DataFrame.from_records(diagnostic_records)
    return estimates.reset_index(drop=True), diagnostics.reset_index(drop=True)


def write_spread_guard_outputs(
    *,
    estimate_inputs: Iterable[EstimateInput],
    template: pd.DataFrame,
    output_dir: Path,
    spread_threshold_m: float,
    fallback_policy: str = "max-weight",
    fallback_label: str | None = None,
    max_nearest_time_delta_s: float | None = None,
    class_map: dict[str, str] | None = None,
    default_classification: int | str = 0,
) -> dict[str, Path]:
    """Write spread-guarded estimates, diagnostics, official artifacts, and manifest."""

    input_list = list(estimate_inputs)
    loaded_inputs = [
        (item.label, pd.read_csv(item.path), float(item.weight)) for item in input_list
    ]
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    estimates, diagnostics = build_spread_guarded_estimate_ensemble(
        loaded_inputs,
        template,
        spread_threshold_m=spread_threshold_m,
        fallback_policy=fallback_policy,
        fallback_label=fallback_label,
        max_nearest_time_delta_s=max_nearest_time_delta_s,
    )
    paths = {
        "estimates_csv": output / SPREAD_GUARD_ESTIMATES_CSV,
        "diagnostics_csv": output / SPREAD_GUARD_DIAGNOSTICS_CSV,
        "official_results_csv": output / OFFICIAL_RESULTS_CSV,
        "official_zip": output / OFFICIAL_ZIP,
        "validation_json": output / SPREAD_GUARD_VALIDATION_JSON,
        "validation_rows_csv": output / SPREAD_GUARD_VALIDATION_ROWS_CSV,
        "manifest_json": output / SPREAD_GUARD_MANIFEST_JSON,
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
    validation = validate_official_track5_submission(paths["official_zip"], template=template, require_zip=True)
    paths["validation_json"].write_text(
        json.dumps(_jsonable(validation.summary), indent=2),
        encoding="utf-8",
    )
    validation.rows.to_csv(paths["validation_rows_csv"], index=False)
    manifest = {
        "schema": "raft-uav-mmuad-track5-estimate-spread-guard-v1",
        "estimate_inputs": [
            {"label": item.label, "path": str(item.path), "weight": float(item.weight)}
            for item in input_list
        ],
        "spread_threshold_m": float(spread_threshold_m),
        "fallback_policy": fallback_policy,
        "fallback_label": fallback_label,
        "max_nearest_time_delta_s": max_nearest_time_delta_s,
        "row_count": int(len(estimates)),
        "guard_applied_rows": int(estimates["spread_guard_applied"].astype(bool).sum()),
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
        prog="raft-uav-mmuad-track5-spread-guard-ensemble",
        description="guard Track 5 estimate ensembles against high-disagreement rows",
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
    parser.add_argument("--spread-threshold-m", type=float, required=True)
    parser.add_argument("--fallback-policy", choices=FALLBACK_POLICIES, default="max-weight")
    parser.add_argument("--fallback-label")
    parser.add_argument("--max-nearest-time-delta-s", type=float)
    parser.add_argument("--class-map", type=Path)
    parser.add_argument("--default-classification", default="0")
    parser.add_argument("--require-leaderboard-ready", action="store_true")
    args = parser.parse_args(argv)

    if not args.estimate_csv:
        parser.error("provide at least one --estimate-csv LABEL=PATH[@WEIGHT]")
    template = load_official_track5_template_file(args.template)
    class_map = load_sequence_class_map(args.class_map) if args.class_map is not None else {}
    paths = write_spread_guard_outputs(
        estimate_inputs=[parse_estimate_spec(value) for value in args.estimate_csv],
        template=template,
        output_dir=args.output_dir,
        spread_threshold_m=float(args.spread_threshold_m),
        fallback_policy=args.fallback_policy,
        fallback_label=args.fallback_label,
        max_nearest_time_delta_s=args.max_nearest_time_delta_s,
        class_map=class_map,
        default_classification=args.default_classification,
    )
    validation = json.loads(paths["validation_json"].read_text(encoding="utf-8"))
    print("mmuad_track5_spread_guard_ensemble=ok")
    for name, path in paths.items():
        print(f"{name}={path}")
    print(f"leaderboard_ready={validation.get('leaderboard_ready')}")
    print(f"codabench_upload_ready={validation.get('codabench_upload_ready')}")
    if args.require_leaderboard_ready and not validation.get("leaderboard_ready", False):
        reasons = ", ".join(validation.get("leaderboard_blocking_reasons", [])) or "unknown"
        raise SystemExit(f"spread-guarded ensemble is not leaderboard-ready: {reasons}")
    return 0


def _fallback_row(
    valid: pd.DataFrame,
    *,
    policy: str,
    fallback_label: str | None,
    input_order: list[str],
) -> pd.Series:
    if policy == "label" and fallback_label is not None:
        rows = valid.loc[valid["input_label"].astype(str) == fallback_label]
        if not rows.empty:
            return rows.iloc[0]
    if policy == "first":
        order_map = {label: index for index, label in enumerate(input_order)}
        work = valid.copy()
        work["_fallback_order"] = work["input_label"].map(lambda label: order_map.get(str(label), 10**9))
        return work.sort_values(["_fallback_order", "input_order"]).iloc[0]
    return valid.sort_values(["input_weight", "input_order"], ascending=[False, True]).iloc[0]


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


def _validate_weight(weight: float, *, label: str) -> float:
    value = float(weight)
    if not np.isfinite(value) or value < 0.0:
        raise ValueError(f"estimate weight for {label} must be finite and non-negative")
    return value


def _weighted_spread_m(xyz: np.ndarray, weights: np.ndarray, center: np.ndarray) -> float:
    if len(xyz) == 0:
        return np.nan
    distances = np.linalg.norm(xyz - center[None, :], axis=1)
    return float(np.sum(weights * distances) / np.sum(weights))


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
