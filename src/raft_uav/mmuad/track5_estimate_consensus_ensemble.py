"""Consensus-cluster estimate ensembling for MMUAD Track 5 submissions.

Weighted averaging helps when independent trajectories make small errors, but it can
be harmed by a single divergent branch.  Spread guarding falls back to a trusted
branch when disagreement is high; this module instead chooses the largest/highest
weight spatial consensus cluster at every official Track 5 timestamp and averages
only that cluster.  It is inference-safe: only submitted estimate trajectories and
an official Sequence/Timestamp template are used.
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

CONSENSUS_ESTIMATES_CSV = "mmuad_track5_consensus_ensemble_estimates.csv"
CONSENSUS_DIAGNOSTICS_CSV = "mmuad_track5_consensus_ensemble_diagnostics.csv"
CONSENSUS_MANIFEST_JSON = "mmuad_track5_consensus_ensemble_manifest.json"
VALIDATION_JSON = "mmuad_track5_consensus_ensemble_validation.json"
VALIDATION_ROWS_CSV = "mmuad_track5_consensus_ensemble_validation_rows.csv"
OFFICIAL_RESULTS_CSV = "mmaud_results.csv"
OFFICIAL_ZIP = "ug2_submission.zip"
TEMPLATE_TIME_MATCH_ATOL_S = 1.0e-9
FALLBACK_POLICIES = ("max-weight", "weighted-mean")


def build_track5_consensus_estimate_ensemble(
    estimate_inputs: Iterable[tuple[str, pd.DataFrame, float]],
    template: pd.DataFrame,
    *,
    consensus_radius_m: float = 5.0,
    fallback_policy: str = "max-weight",
    min_consensus_weight_fraction: float = 0.0,
    max_nearest_time_delta_s: float | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return Track 5 estimates from row-wise spatial consensus clusters.

    For every requested template row, each estimate input is first resampled onto
    that timestamp.  A consensus set is then formed around each valid input by
    collecting all inputs within ``consensus_radius_m``.  The selected set is the
    one with the largest total input weight, then largest count, then smallest
    weighted spread.  If the selected cluster is too weak according to
    ``min_consensus_weight_fraction``, the configured fallback is used.
    """

    radius = float(consensus_radius_m)
    if not np.isfinite(radius) or radius < 0.0:
        raise ValueError("consensus_radius_m must be finite and non-negative")
    if fallback_policy not in FALLBACK_POLICIES:
        raise ValueError(f"fallback_policy must be one of: {', '.join(FALLBACK_POLICIES)}")
    min_fraction = float(min_consensus_weight_fraction)
    if not 0.0 <= min_fraction <= 1.0:
        raise ValueError("min_consensus_weight_fraction must be in [0, 1]")
    template_rows = _normalize_template_rows(template)
    loaded_inputs = tuple(estimate_inputs)
    if not loaded_inputs:
        raise ValueError("at least one estimate input is required")

    parts: list[pd.DataFrame] = []
    for order, (label_text, estimates, weight) in enumerate(loaded_inputs):
        label = _safe_label(label_text)
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
        parts.append(part)
    stacked = pd.concat(parts, ignore_index=True, sort=False)

    estimate_records: list[dict[str, Any]] = []
    diagnostic_records: list[dict[str, Any]] = []
    for _, template_row in template_rows.iterrows():
        sequence_id = str(template_row["sequence_id"])
        time_s = float(template_row["time_s"])
        rows = stacked.loc[
            (stacked["sequence_id"].astype(str) == sequence_id)
            & _template_time_matches(stacked["time_s"], time_s)
        ]
        valid = rows.loc[rows["input_valid"].astype(bool) & (rows["input_weight"] > 0.0)].copy()
        if valid.empty:
            xyz = np.asarray([np.nan, np.nan, np.nan], dtype=float)
            chosen = valid
            fallback_applied = False
            reason = "no_valid_inputs"
            input_spread = np.nan
            consensus_spread = np.nan
        else:
            xyz_values = valid[["state_x_m", "state_y_m", "state_z_m"]].to_numpy(float)
            weights = valid["input_weight"].to_numpy(float)
            weighted_xyz = np.sum(weights[:, None] * xyz_values, axis=0) / float(np.sum(weights))
            input_spread = _weighted_spread_m(xyz_values, weights, weighted_xyz)
            chosen_indices, consensus_spread = _best_consensus_indices(xyz_values, weights, radius)
            chosen = valid.iloc[chosen_indices].copy()
            total_weight = float(np.sum(weights))
            chosen_weight = float(chosen["input_weight"].sum())
            fallback_applied = bool(total_weight > 0.0 and chosen_weight < min_fraction * total_weight)
            reason = "consensus"
            if fallback_applied:
                reason = "fallback_low_consensus_weight"
                if fallback_policy == "max-weight":
                    chosen = valid.sort_values(["input_weight", "input_order"], ascending=[False, True]).head(1)
                else:
                    chosen = valid
            chosen_xyz = chosen[["state_x_m", "state_y_m", "state_z_m"]].to_numpy(float)
            chosen_weights = chosen["input_weight"].to_numpy(float)
            xyz = np.sum(chosen_weights[:, None] * chosen_xyz, axis=0) / float(np.sum(chosen_weights))
            if len(chosen) > 1:
                consensus_spread = _weighted_spread_m(chosen_xyz, chosen_weights, xyz)
            else:
                consensus_spread = 0.0
        chosen_labels = ";".join(chosen["input_label"].astype(str)) if not chosen.empty else ""
        all_labels = ";".join(valid["input_label"].astype(str)) if not valid.empty else ""
        rejected = sorted(set(all_labels.split(";")).difference(chosen_labels.split(";"))) if all_labels else []
        estimate_records.append(
            {
                "sequence_id": sequence_id,
                "time_s": time_s,
                "source": "track5-consensus-ensemble",
                "track_id": "track5-consensus-ensemble",
                "state_x_m": float(xyz[0]) if np.isfinite(xyz[0]) else np.nan,
                "state_y_m": float(xyz[1]) if np.isfinite(xyz[1]) else np.nan,
                "state_z_m": float(xyz[2]) if np.isfinite(xyz[2]) else np.nan,
                "consensus_radius_m": radius,
                "consensus_input_count": int(len(chosen)),
                "consensus_labels": chosen_labels,
                "consensus_fallback_applied": bool(fallback_applied),
            }
        )
        diagnostic_records.append(
            {
                "sequence_id": sequence_id,
                "time_s": time_s,
                "valid_input_count": int(len(valid)),
                "selected_input_count": int(len(chosen)),
                "input_labels": all_labels,
                "selected_labels": chosen_labels,
                "rejected_labels": ";".join(label for label in rejected if label),
                "input_spread_m": input_spread,
                "consensus_spread_m": consensus_spread,
                "consensus_radius_m": radius,
                "fallback_policy": fallback_policy,
                "fallback_applied": bool(fallback_applied),
                "selection_reason": reason,
            }
        )
    return pd.DataFrame.from_records(estimate_records), pd.DataFrame.from_records(diagnostic_records)


def write_track5_consensus_ensemble_outputs(
    *,
    estimate_inputs: Iterable[EstimateInput],
    template: pd.DataFrame,
    output_dir: Path,
    class_map: dict[str, str] | None = None,
    default_classification: int | str = 0,
    consensus_radius_m: float = 5.0,
    fallback_policy: str = "max-weight",
    min_consensus_weight_fraction: float = 0.0,
    max_nearest_time_delta_s: float | None = None,
) -> dict[str, Path]:
    """Write consensus estimates, official CSV/ZIP, validation, and manifest."""

    input_list = list(estimate_inputs)
    loaded = [(item.label, pd.read_csv(item.path), float(item.weight)) for item in input_list]
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    estimates, diagnostics = build_track5_consensus_estimate_ensemble(
        loaded,
        template,
        consensus_radius_m=consensus_radius_m,
        fallback_policy=fallback_policy,
        min_consensus_weight_fraction=min_consensus_weight_fraction,
        max_nearest_time_delta_s=max_nearest_time_delta_s,
    )
    paths = {
        "estimates_csv": output / CONSENSUS_ESTIMATES_CSV,
        "diagnostics_csv": output / CONSENSUS_DIAGNOSTICS_CSV,
        "official_results_csv": output / OFFICIAL_RESULTS_CSV,
        "official_zip": output / OFFICIAL_ZIP,
        "validation_json": output / VALIDATION_JSON,
        "validation_rows_csv": output / VALIDATION_ROWS_CSV,
        "manifest_json": output / CONSENSUS_MANIFEST_JSON,
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
        paths["official_zip"], template=template, require_zip=True
    )
    paths["validation_json"].write_text(
        json.dumps(_jsonable(validation.summary), indent=2), encoding="utf-8"
    )
    validation.rows.to_csv(paths["validation_rows_csv"], index=False)
    manifest = {
        "schema": "raft-uav-mmuad-track5-consensus-ensemble-v1",
        "estimate_inputs": [
            {"label": item.label, "path": str(item.path), "weight": float(item.weight)}
            for item in input_list
        ],
        "row_count": int(len(estimates)),
        "valid_rows": int(_finite_xyz(estimates).sum()),
        "consensus_radius_m": float(consensus_radius_m),
        "fallback_policy": fallback_policy,
        "min_consensus_weight_fraction": float(min_consensus_weight_fraction),
        "max_nearest_time_delta_s": max_nearest_time_delta_s,
        "mean_input_spread_m": _safe_mean(diagnostics.get("input_spread_m", pd.Series(dtype=float))),
        "mean_consensus_spread_m": _safe_mean(diagnostics.get("consensus_spread_m", pd.Series(dtype=float))),
        "fallback_rows": int(diagnostics.get("fallback_applied", pd.Series(dtype=bool)).sum()),
        "leaderboard_ready": bool(validation.summary.get("leaderboard_ready", False)),
        "codabench_upload_ready": bool(validation.summary.get("codabench_upload_ready", False)),
        "paths": {name: str(path) for name, path in paths.items() if name != "manifest_json"},
    }
    paths["manifest_json"].write_text(json.dumps(_jsonable(manifest), indent=2), encoding="utf-8")
    return paths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="raft-uav-mmuad-track5-consensus-ensemble",
        description="build a consensus-cluster ensemble for MMUAD Track 5 estimates",
    )
    parser.add_argument("--estimate-csv", action="append", default=[], metavar="LABEL=PATH[@WEIGHT]")
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--class-map", type=Path)
    parser.add_argument("--default-classification", default="0")
    parser.add_argument("--consensus-radius-m", type=float, default=5.0)
    parser.add_argument("--fallback-policy", choices=FALLBACK_POLICIES, default="max-weight")
    parser.add_argument("--min-consensus-weight-fraction", type=float, default=0.0)
    parser.add_argument("--max-nearest-time-delta-s", type=float)
    parser.add_argument("--require-leaderboard-ready", action="store_true")
    args = parser.parse_args(argv)

    if not args.estimate_csv:
        parser.error("provide at least one --estimate-csv LABEL=PATH[@WEIGHT]")
    inputs = [parse_estimate_spec(value) for value in args.estimate_csv]
    template = load_official_track5_template_file(args.template)
    class_map = load_sequence_class_map(args.class_map) if args.class_map is not None else {}
    paths = write_track5_consensus_ensemble_outputs(
        estimate_inputs=inputs,
        template=template,
        output_dir=args.output_dir,
        class_map=class_map,
        default_classification=args.default_classification,
        consensus_radius_m=float(args.consensus_radius_m),
        fallback_policy=args.fallback_policy,
        min_consensus_weight_fraction=float(args.min_consensus_weight_fraction),
        max_nearest_time_delta_s=args.max_nearest_time_delta_s,
    )
    summary = json.loads(paths["validation_json"].read_text(encoding="utf-8"))
    print("mmuad_track5_consensus_ensemble=ok")
    for name, path in paths.items():
        print(f"{name}={path}")
    print(f"leaderboard_ready={summary.get('leaderboard_ready')}")
    print(f"codabench_upload_ready={summary.get('codabench_upload_ready')}")
    if args.require_leaderboard_ready and not summary.get("leaderboard_ready", False):
        reasons = ", ".join(summary.get("leaderboard_blocking_reasons", [])) or "unknown"
        raise SystemExit(f"consensus ensemble upload is not leaderboard-ready: {reasons}")
    return 0


def _best_consensus_indices(xyz: np.ndarray, weights: np.ndarray, radius: float) -> tuple[np.ndarray, float]:
    distances = np.linalg.norm(xyz[:, None, :] - xyz[None, :, :], axis=2)
    best_indices = np.asarray([0], dtype=int)
    best_key: tuple[float, int, float] | None = None
    best_spread = 0.0
    for index in range(len(xyz)):
        indices = np.flatnonzero(distances[index] <= radius)
        cluster_xyz = xyz[indices]
        cluster_weights = weights[indices]
        center = np.sum(cluster_weights[:, None] * cluster_xyz, axis=0) / float(np.sum(cluster_weights))
        spread = _weighted_spread_m(cluster_xyz, cluster_weights, center)
        key = (float(np.sum(cluster_weights)), int(len(indices)), -float(spread))
        if best_key is None or key > best_key:
            best_key = key
            best_indices = indices
            best_spread = float(spread)
    return best_indices, best_spread


def _weighted_spread_m(xyz: np.ndarray, weights: np.ndarray, center: np.ndarray) -> float:
    distances = np.linalg.norm(xyz - center[None, :], axis=1)
    return float(np.sum(weights * distances) / np.sum(weights))


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
    weight = float(weight)
    if not np.isfinite(weight) or weight < 0.0:
        raise ValueError(f"estimate weight must be finite and non-negative for {label}: {weight}")
    return weight


def _safe_mean(values: pd.Series) -> float | None:
    numeric = pd.to_numeric(values, errors="coerce")
    numeric = numeric[np.isfinite(numeric.to_numpy(float))]
    if numeric.empty:
        return None
    return float(np.mean(numeric.to_numpy(float)))


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
