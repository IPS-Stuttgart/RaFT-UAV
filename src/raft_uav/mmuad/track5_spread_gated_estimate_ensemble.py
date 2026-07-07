"""Spread-gated Track 5 estimate ensembling for MMUAD submissions.

Weighted means, medians, and trimmed means help combine independent pose
pipelines, but they can still be harmful when the input trajectories strongly
disagree.  This module adds an inference-safe disagreement gate: compute an
ordinary estimate ensemble on the official Track 5 template, then fall back to a
train-selected anchor trajectory on rows whose ensemble position spread exceeds a
fixed threshold.

The gate uses only submitted estimates and the official timestamp template.  The
threshold and anchor must be selected upstream on train folds for leaderboard use.
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
from raft_uav.mmuad.track5_estimate_ensemble import (
    ENSEMBLE_POLICIES,
    EstimateInput,
    build_track5_estimate_ensemble,
    parse_estimate_spec,
)
from raft_uav.mmuad.track5_template_resample import resample_estimates_to_track5_template

GATED_ESTIMATES_CSV = "mmuad_track5_spread_gated_estimates.csv"
GATED_DIAGNOSTICS_CSV = "mmuad_track5_spread_gated_diagnostics.csv"
GATED_MANIFEST_JSON = "mmuad_track5_spread_gated_manifest.json"
VALIDATION_JSON = "mmuad_track5_spread_gated_validation.json"
VALIDATION_ROWS_CSV = "mmuad_track5_spread_gated_validation_rows.csv"
OFFICIAL_RESULTS_CSV = "mmaud_results.csv"
OFFICIAL_ZIP = "ug2_submission.zip"


def build_spread_gated_estimate_ensemble(
    estimate_inputs: Iterable[tuple[str, pd.DataFrame, float]],
    template: pd.DataFrame,
    *,
    anchor_label: str,
    spread_threshold_m: float,
    max_nearest_time_delta_s: float | None = None,
    aggregation_policy: str = "weighted-mean",
    trim_fraction: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return a spread-gated estimate trajectory and diagnostics.

    The base ensemble is computed with :func:`build_track5_estimate_ensemble`.
    The anchor trajectory is independently resampled to the same template.  Rows
    with ``position_spread_m > spread_threshold_m`` use the anchor position when
    the anchor row is finite and valid; other rows keep the base ensemble.
    """

    threshold = float(spread_threshold_m)
    if not np.isfinite(threshold) or threshold < 0.0:
        raise ValueError("spread_threshold_m must be finite and non-negative")
    inputs = [(str(label), pd.DataFrame(rows).copy(), float(weight)) for label, rows, weight in estimate_inputs]
    if not inputs:
        raise ValueError("at least one estimate input is required")
    safe_anchor_label = _safe_label(anchor_label)
    anchor_matches = [rows for label, rows, _ in inputs if _safe_label(label) == safe_anchor_label]
    if not anchor_matches:
        available = sorted(_safe_label(label) for label, _, _ in inputs)
        raise ValueError(f"anchor label {safe_anchor_label!r} not present; available labels: {available}")

    ensemble, ensemble_diagnostics = build_track5_estimate_ensemble(
        inputs,
        template,
        max_nearest_time_delta_s=max_nearest_time_delta_s,
        aggregation_policy=aggregation_policy,
        trim_fraction=trim_fraction,
    )
    anchor_rows, anchor_diagnostics = resample_estimates_to_track5_template(
        anchor_matches[0],
        template,
        max_nearest_time_delta_s=max_nearest_time_delta_s,
    )
    anchor = anchor_rows.rename(
        columns={
            "state_x_m": "anchor_state_x_m",
            "state_y_m": "anchor_state_y_m",
            "state_z_m": "anchor_state_z_m",
            "template_resample_valid": "anchor_template_resample_valid",
            "template_nearest_time_delta_s": "anchor_nearest_time_delta_s",
        }
    )
    joined = ensemble.merge(
        anchor[
            [
                "sequence_id",
                "time_s",
                "anchor_state_x_m",
                "anchor_state_y_m",
                "anchor_state_z_m",
                "anchor_template_resample_valid",
                "anchor_nearest_time_delta_s",
            ]
        ],
        on=["sequence_id", "time_s"],
        how="left",
        validate="one_to_one",
    )
    diag = ensemble_diagnostics.copy()
    if "position_spread_m" not in diag.columns:
        diag["position_spread_m"] = np.nan
    joined = joined.merge(
        diag[["sequence_id", "time_s", "position_spread_m"]],
        on=["sequence_id", "time_s"],
        how="left",
        suffixes=("", "_diag"),
    )
    anchor_valid = _finite_columns(joined, ["anchor_state_x_m", "anchor_state_y_m", "anchor_state_z_m"])
    if "anchor_template_resample_valid" in joined.columns:
        anchor_valid &= joined["anchor_template_resample_valid"].fillna(False).astype(bool)
    spread = pd.to_numeric(joined["position_spread_m"], errors="coerce")
    use_anchor = anchor_valid & np.isfinite(spread.to_numpy(float)) & (spread > threshold)

    out = joined.copy()
    for axis in ("x", "y", "z"):
        state_column = f"state_{axis}_m"
        anchor_column = f"anchor_state_{axis}_m"
        out[state_column] = np.where(use_anchor, out[anchor_column], out[state_column])
    out["spread_gated"] = use_anchor.astype(bool)
    out["spread_gate_anchor_label"] = safe_anchor_label
    out["spread_gate_threshold_m"] = threshold
    out["spread_gate_reason"] = np.where(use_anchor, "spread_gt_threshold_anchor", "base_ensemble")

    diagnostics = diag.merge(
        out[
            [
                "sequence_id",
                "time_s",
                "spread_gated",
                "spread_gate_reason",
                "anchor_nearest_time_delta_s",
            ]
        ],
        on=["sequence_id", "time_s"],
        how="left",
    )
    diagnostics["spread_gate_anchor_label"] = safe_anchor_label
    diagnostics["spread_gate_threshold_m"] = threshold
    diagnostics.attrs["input_summaries"] = ensemble_diagnostics.attrs.get("input_summaries", [])
    diagnostics.attrs["anchor_resample_mean_abs_time_delta_s"] = _safe_mean_abs(
        anchor_diagnostics.get("nearest_time_delta_s", pd.Series(dtype=float))
    )
    return out.reset_index(drop=True), diagnostics.reset_index(drop=True)


def write_spread_gated_estimate_ensemble_outputs(
    *,
    estimate_inputs: Iterable[EstimateInput],
    template: pd.DataFrame,
    output_dir: Path,
    anchor_label: str,
    spread_threshold_m: float,
    class_map: dict[str, str] | None = None,
    default_classification: int | str = 0,
    max_nearest_time_delta_s: float | None = None,
    aggregation_policy: str = "weighted-mean",
    trim_fraction: float = 0.2,
) -> dict[str, Path]:
    """Write spread-gated ensemble estimates, official artifacts, and manifest."""

    input_list = list(estimate_inputs)
    loaded_inputs = [(item.label, pd.read_csv(item.path), float(item.weight)) for item in input_list]
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    estimates, diagnostics = build_spread_gated_estimate_ensemble(
        loaded_inputs,
        template,
        anchor_label=anchor_label,
        spread_threshold_m=spread_threshold_m,
        max_nearest_time_delta_s=max_nearest_time_delta_s,
        aggregation_policy=aggregation_policy,
        trim_fraction=trim_fraction,
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
    paths["validation_json"].write_text(
        json.dumps(_jsonable(validation.summary), indent=2),
        encoding="utf-8",
    )
    validation.rows.to_csv(paths["validation_rows_csv"], index=False)
    manifest = {
        "schema": "raft-uav-mmuad-track5-spread-gated-estimate-ensemble-v1",
        "estimate_inputs": [
            {"label": item.label, "path": str(item.path), "weight": float(item.weight)}
            for item in input_list
        ],
        "anchor_label": _safe_label(anchor_label),
        "spread_threshold_m": float(spread_threshold_m),
        "aggregation_policy": aggregation_policy,
        "trim_fraction": float(trim_fraction),
        "row_count": int(len(estimates)),
        "spread_gated_rows": int(estimates["spread_gated"].sum()) if "spread_gated" in estimates else 0,
        "mean_position_spread_m": _safe_mean(diagnostics.get("position_spread_m", pd.Series(dtype=float))),
        "p95_position_spread_m": _safe_percentile(
            diagnostics.get("position_spread_m", pd.Series(dtype=float)),
            95,
        ),
        "anchor_resample_mean_abs_time_delta_s": diagnostics.attrs.get(
            "anchor_resample_mean_abs_time_delta_s"
        ),
        "input_summaries": diagnostics.attrs.get("input_summaries", []),
        "max_nearest_time_delta_s": max_nearest_time_delta_s,
        "leaderboard_ready": bool(validation.summary.get("leaderboard_ready", False)),
        "codabench_upload_ready": bool(validation.summary.get("codabench_upload_ready", False)),
        "paths": {name: str(path) for name, path in paths.items() if name != "manifest_json"},
    }
    paths["manifest_json"].write_text(json.dumps(_jsonable(manifest), indent=2), encoding="utf-8")
    return paths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="raft-uav-mmuad-spread-gated-estimate-ensemble",
        description="fallback to an anchor Track 5 estimate when ensemble disagreement is high",
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
    parser.add_argument("--anchor-label", required=True)
    parser.add_argument("--spread-threshold-m", type=float, required=True)
    parser.add_argument("--class-map", type=Path)
    parser.add_argument("--default-classification", default="0")
    parser.add_argument("--max-nearest-time-delta-s", type=float)
    parser.add_argument("--aggregation-policy", choices=ENSEMBLE_POLICIES, default="weighted-mean")
    parser.add_argument("--trim-fraction", type=float, default=0.2)
    parser.add_argument("--require-leaderboard-ready", action="store_true")
    args = parser.parse_args(argv)

    if not args.estimate_csv:
        parser.error("provide at least one --estimate-csv LABEL=PATH[@WEIGHT]")
    estimate_inputs = [parse_estimate_spec(spec) for spec in args.estimate_csv]
    template = load_official_track5_template_file(args.template)
    class_map = load_sequence_class_map(args.class_map) if args.class_map is not None else {}
    paths = write_spread_gated_estimate_ensemble_outputs(
        estimate_inputs=estimate_inputs,
        template=template,
        output_dir=args.output_dir,
        anchor_label=args.anchor_label,
        spread_threshold_m=float(args.spread_threshold_m),
        class_map=class_map,
        default_classification=args.default_classification,
        max_nearest_time_delta_s=args.max_nearest_time_delta_s,
        aggregation_policy=args.aggregation_policy,
        trim_fraction=float(args.trim_fraction),
    )
    summary = json.loads(paths["validation_json"].read_text(encoding="utf-8"))
    print("mmuad_spread_gated_estimate_ensemble=ok")
    for name, path in paths.items():
        print(f"{name}={path}")
    print(f"leaderboard_ready={summary.get('leaderboard_ready')}")
    print(f"codabench_upload_ready={summary.get('codabench_upload_ready')}")
    if args.require_leaderboard_ready and not summary.get("leaderboard_ready", False):
        reasons = ", ".join(summary.get("leaderboard_blocking_reasons", [])) or "unknown"
        raise SystemExit(f"spread-gated ensemble is not leaderboard-ready: {reasons}")
    return 0


def _finite_columns(rows: pd.DataFrame, columns: list[str]) -> pd.Series:
    if rows.empty:
        return pd.Series(dtype=bool)
    values = rows[columns].apply(pd.to_numeric, errors="coerce")
    return pd.Series(np.isfinite(values.to_numpy(float)).all(axis=1), index=rows.index)


def _safe_label(value: object) -> str:
    text = "" if value is None else str(value)
    return text.strip().replace(" ", "_").replace("/", "_").replace("\\", "_") or "estimate"


def _safe_mean(values: pd.Series) -> float | None:
    numeric = pd.to_numeric(values, errors="coerce")
    numeric = numeric[np.isfinite(numeric.to_numpy(float))]
    if numeric.empty:
        return None
    return float(numeric.mean())


def _safe_mean_abs(values: pd.Series) -> float | None:
    numeric = pd.to_numeric(values, errors="coerce")
    numeric = numeric[np.isfinite(numeric.to_numpy(float))]
    if numeric.empty:
        return None
    return float(np.mean(np.abs(numeric.to_numpy(float))))


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
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
