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
        return EstimateInput(label=_safe_label(path.stem), path=path, weight=1.0)
    label, path_weight = value.split("=", 1)
    weight = 1.0
    path_text = path_weight
    if "@" in path_weight:
        path_text, weight_text = path_weight.rsplit("@", 1)
        weight = float(weight_text)
    return EstimateInput(label=_safe_label(label), path=Path(path_text), weight=float(weight))


def build_track5_estimate_ensemble(
    estimate_inputs: Iterable[tuple[str, pd.DataFrame, float]],
    template: pd.DataFrame,
    *,
    max_nearest_time_delta_s: float | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return weighted-mean estimates and per-template-row diagnostics.

    Each estimate input is first interpolated to the template with
    :func:`resample_estimates_to_track5_template`.  The final position is a
    row-wise weighted average over finite, valid resampled trajectories.  Missing
    sequences or invalid rows are ignored for that row; diagnostics report which
    labels contributed.
    """

    template_rows = _normalize_template_rows(template)
    if template_rows.empty:
        empty = pd.DataFrame(columns=["sequence_id", "time_s", "state_x_m", "state_y_m", "state_z_m"])
        return empty, pd.DataFrame()

    resampled_parts: list[pd.DataFrame] = []
    input_summaries: list[dict[str, Any]] = []
    for label, estimates, weight in estimate_inputs:
        label = _safe_label(label)
        weight = float(weight)
        if weight < 0.0:
            raise ValueError(f"estimate weight must be non-negative for {label}: {weight}")
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
            & np.isclose(pd.to_numeric(stacked["time_s"], errors="coerce"), time_s)
        ]
        valid = rows.loc[rows["ensemble_valid"].astype(bool) & (rows["ensemble_weight"] > 0.0)]
        if valid.empty:
            xyz = np.asarray([np.nan, np.nan, np.nan], dtype=float)
            total_weight = 0.0
            labels = ""
        else:
            weights = valid["ensemble_weight"].to_numpy(float)
            xyz_values = valid[["state_x_m", "state_y_m", "state_z_m"]].to_numpy(float)
            total_weight = float(np.sum(weights))
            xyz = np.sum(weights[:, None] * xyz_values, axis=0) / total_weight
            labels = ";".join(valid["ensemble_label"].astype(str).tolist())
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
) -> dict[str, Path]:
    """Write ensemble estimates, official CSV/ZIP, validation, and manifest."""

    loaded_inputs = [
        (item.label, pd.read_csv(item.path), float(item.weight)) for item in estimate_inputs
    ]
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    ensemble, diagnostics = build_track5_estimate_ensemble(
        loaded_inputs,
        template,
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
            for item in estimate_inputs
        ],
        "input_summaries": diagnostics.attrs.get("input_summaries", []),
        "row_count": int(len(ensemble)),
        "valid_ensemble_rows": int(_finite_xyz(ensemble).sum()),
        "max_nearest_time_delta_s": max_nearest_time_delta_s,
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
    parser.add_argument("--require-leaderboard-ready", action="store_true")
    args = parser.parse_args(argv)

    if not args.estimate_csv:
        parser.error("provide at least one --estimate-csv LABEL=PATH[@WEIGHT]")
    estimate_inputs = [parse_estimate_spec(spec) for spec in args.estimate_csv]
    template = load_official_track5_template_file(args.template)
    class_map = load_sequence_class_map(args.class_map) if args.class_map is not None else {}
    paths = write_track5_estimate_ensemble_outputs(
        estimate_inputs=estimate_inputs,
        template=template,
        output_dir=args.output_dir,
        class_map=class_map,
        default_classification=args.default_classification,
        max_nearest_time_delta_s=args.max_nearest_time_delta_s,
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
