"""Train-only calibration for MMUAD per-candidate uncertainty.

Learned candidate uncertainty is useful for robust mixture-MAP tracking, but a
single regressor can still be systematically over- or under-confident for raw,
dynamic, source-translated, or sensor-specific candidate branches.  This module
fits a hierarchical multiplicative calibration on training truth only and
applies it without validation/test truth.

The hierarchy backs off from source+branch to branch, source, and finally a
global scale.  Group scales are shrunk toward their parent scale so sparse
branches cannot dominate mixture weighting.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from raft_uav.mmuad.cluster_ranker import build_cluster_feature_table
from raft_uav.mmuad.evaluator import load_evaluation_truth_file
from raft_uav.mmuad.io import load_candidate_file
from raft_uav.mmuad.schema import CandidateFrame, normalize_candidate_columns

CALIBRATION_SCHEMA = "raft-uav-mmuad-candidate-sigma-calibration-v1"
_DEFAULT_BRANCH = "unbranched"
_DEFAULT_SOURCE = "unknown"


@dataclass(frozen=True)
class CandidateSigmaCalibration:
    """Hierarchical multiplicative calibration for predicted candidate sigma."""

    schema: str
    input_sigma_column: str
    branch_column: str
    target_quantile: float
    min_group_rows: int
    shrinkage_rows: float
    scale_min: float
    scale_max: float
    calibration_row_count: int
    global_scale: float
    source_scales: dict[str, float]
    branch_scales: dict[str, float]
    source_branch_scales: dict[str, float]
    source_counts: dict[str, int]
    branch_counts: dict[str, int]
    source_branch_counts: dict[str, int]


def fit_candidate_sigma_calibration(
    features: pd.DataFrame,
    *,
    input_sigma_column: str = "predicted_sigma_m",
    branch_column: str = "candidate_branch",
    target_quantile: float = 0.5,
    min_group_rows: int = 20,
    shrinkage_rows: float = 50.0,
    scale_min: float = 0.25,
    scale_max: float = 4.0,
) -> CandidateSigmaCalibration:
    """Fit hierarchical sigma scales from train-labeled candidate features.

    ``truth_distance_3d_m / predicted_sigma_m`` is calibrated at the requested
    quantile.  Source and branch scales are shrunk toward the global scale;
    source+branch scales are shrunk toward the geometric mean of their source
    and branch parents.
    """

    if not 0.0 < float(target_quantile) <= 1.0:
        raise ValueError("target_quantile must be in (0, 1]")
    if int(min_group_rows) < 1:
        raise ValueError("min_group_rows must be at least 1")
    if float(shrinkage_rows) < 0.0:
        raise ValueError("shrinkage_rows must be non-negative")
    if not 0.0 < float(scale_min) <= float(scale_max):
        raise ValueError("scale bounds must satisfy 0 < scale_min <= scale_max")

    rows = pd.DataFrame(features).copy()
    if "truth_distance_3d_m" not in rows.columns:
        raise ValueError("sigma calibration requires truth_distance_3d_m labels")
    if input_sigma_column not in rows.columns:
        raise ValueError(f"sigma calibration missing input column {input_sigma_column!r}")

    truth = pd.to_numeric(rows["truth_distance_3d_m"], errors="coerce")
    sigma = pd.to_numeric(rows[input_sigma_column], errors="coerce")
    finite = np.isfinite(truth.to_numpy(float)) & np.isfinite(sigma.to_numpy(float))
    finite &= truth.to_numpy(float) >= 0.0
    finite &= sigma.to_numpy(float) > 0.0
    rows = rows.loc[finite].reset_index(drop=True)
    truth = truth.loc[finite].reset_index(drop=True)
    sigma = sigma.loc[finite].reset_index(drop=True)
    if rows.empty:
        raise ValueError("no finite positive-sigma rows for calibration")

    rows["_sigma_ratio"] = truth.to_numpy(float) / sigma.to_numpy(float)
    rows["_source_group"] = _normalized_group_values(
        rows.get("source"),
        index=rows.index,
        fallback=_DEFAULT_SOURCE,
    )
    rows["_branch_group"] = _normalized_group_values(
        rows.get(branch_column),
        index=rows.index,
        fallback=_DEFAULT_BRANCH,
    )

    global_scale = _raw_scale(
        rows["_sigma_ratio"],
        quantile=target_quantile,
        scale_min=scale_min,
        scale_max=scale_max,
    )
    source_scales, source_counts = _fit_group_scales(
        rows,
        group_column="_source_group",
        parent_scale=lambda _key: global_scale,
        quantile=target_quantile,
        min_group_rows=min_group_rows,
        shrinkage_rows=shrinkage_rows,
        scale_min=scale_min,
        scale_max=scale_max,
    )
    branch_scales, branch_counts = _fit_group_scales(
        rows,
        group_column="_branch_group",
        parent_scale=lambda _key: global_scale,
        quantile=target_quantile,
        min_group_rows=min_group_rows,
        shrinkage_rows=shrinkage_rows,
        scale_min=scale_min,
        scale_max=scale_max,
    )

    rows["_source_branch_group"] = [
        _source_branch_key(source, branch)
        for source, branch in zip(
            rows["_source_group"],
            rows["_branch_group"],
            strict=True,
        )
    ]

    def source_branch_parent(key: str) -> float:
        source, branch = _decode_source_branch_key(key)
        source_scale = source_scales.get(source, global_scale)
        branch_scale = branch_scales.get(branch, global_scale)
        return float(np.sqrt(source_scale * branch_scale))

    source_branch_scales, source_branch_counts = _fit_group_scales(
        rows,
        group_column="_source_branch_group",
        parent_scale=source_branch_parent,
        quantile=target_quantile,
        min_group_rows=min_group_rows,
        shrinkage_rows=shrinkage_rows,
        scale_min=scale_min,
        scale_max=scale_max,
    )

    return CandidateSigmaCalibration(
        schema=CALIBRATION_SCHEMA,
        input_sigma_column=str(input_sigma_column),
        branch_column=str(branch_column),
        target_quantile=float(target_quantile),
        min_group_rows=int(min_group_rows),
        shrinkage_rows=float(shrinkage_rows),
        scale_min=float(scale_min),
        scale_max=float(scale_max),
        calibration_row_count=int(len(rows)),
        global_scale=float(global_scale),
        source_scales=source_scales,
        branch_scales=branch_scales,
        source_branch_scales=source_branch_scales,
        source_counts=source_counts,
        branch_counts=branch_counts,
        source_branch_counts=source_branch_counts,
    )


def apply_candidate_sigma_calibration(
    candidates: CandidateFrame | pd.DataFrame,
    calibration: CandidateSigmaCalibration,
    *,
    input_sigma_column: str | None = None,
    output_sigma_column: str = "calibrated_sigma_m",
    replace_covariance: bool = False,
    z_scale: float = 1.0,
) -> CandidateFrame:
    """Apply a train-fitted hierarchical calibration without target truth."""

    rows = _candidate_rows(candidates)
    if rows.empty:
        return CandidateFrame(rows)
    sigma_column = str(input_sigma_column or calibration.input_sigma_column)
    if sigma_column not in rows.columns:
        raise ValueError(f"candidate rows missing sigma column {sigma_column!r}")
    raw_sigma = pd.to_numeric(rows[sigma_column], errors="coerce")
    if not np.isfinite(raw_sigma.to_numpy(float)).all() or (raw_sigma <= 0.0).any():
        raise ValueError("candidate sigma values must be finite and positive")

    sources = _normalized_group_values(
        rows.get("source"),
        index=rows.index,
        fallback=_DEFAULT_SOURCE,
    )
    branches = _normalized_group_values(
        rows.get(calibration.branch_column),
        index=rows.index,
        fallback=_DEFAULT_BRANCH,
    )
    scales: list[float] = []
    levels: list[str] = []
    for source, branch in zip(sources, branches, strict=True):
        key = _source_branch_key(source, branch)
        if key in calibration.source_branch_scales:
            scales.append(float(calibration.source_branch_scales[key]))
            levels.append("source_branch")
        elif branch in calibration.branch_scales:
            scales.append(float(calibration.branch_scales[branch]))
            levels.append("branch")
        elif source in calibration.source_scales:
            scales.append(float(calibration.source_scales[source]))
            levels.append("source")
        else:
            scales.append(float(calibration.global_scale))
            levels.append("global")

    out = rows.copy()
    scale_array = np.asarray(scales, dtype=float)
    out["candidate_sigma_uncalibrated_m"] = raw_sigma.to_numpy(float)
    out["candidate_sigma_calibration_scale"] = scale_array
    out["candidate_sigma_calibration_level"] = levels
    out[output_sigma_column] = raw_sigma.to_numpy(float) * scale_array
    if replace_covariance:
        out["raw_std_xy_m"] = pd.to_numeric(out.get("std_xy_m"), errors="coerce")
        out["raw_std_z_m"] = pd.to_numeric(out.get("std_z_m"), errors="coerce")
        out["std_xy_m"] = out[output_sigma_column]
        out["std_z_m"] = out[output_sigma_column] * float(z_scale)
    return CandidateFrame(normalize_candidate_columns(out))


def candidate_sigma_calibration_summary(
    features: pd.DataFrame,
    calibration: CandidateSigmaCalibration,
) -> dict[str, Any]:
    """Return compact in-sample diagnostics for calibration provenance."""

    rows = pd.DataFrame(features).copy()
    if rows.empty or "truth_distance_3d_m" not in rows.columns:
        return {"row_count": 0}
    calibrated = apply_candidate_sigma_calibration(rows, calibration).rows
    truth = pd.to_numeric(calibrated["truth_distance_3d_m"], errors="coerce")
    raw = pd.to_numeric(calibrated["candidate_sigma_uncalibrated_m"], errors="coerce")
    adjusted = pd.to_numeric(calibrated["calibrated_sigma_m"], errors="coerce")
    finite = truth.notna() & raw.notna() & adjusted.notna() & (raw > 0.0) & (adjusted > 0.0)
    if not finite.any():
        return {"row_count": 0}
    truth_values = truth.loc[finite].to_numpy(float)
    raw_values = raw.loc[finite].to_numpy(float)
    adjusted_values = adjusted.loc[finite].to_numpy(float)
    return {
        "row_count": int(finite.sum()),
        "target_quantile": float(calibration.target_quantile),
        "global_scale": float(calibration.global_scale),
        "source_group_count": int(len(calibration.source_scales)),
        "branch_group_count": int(len(calibration.branch_scales)),
        "source_branch_group_count": int(len(calibration.source_branch_scales)),
        "raw_ratio_target_quantile": float(
            np.quantile(truth_values / raw_values, calibration.target_quantile)
        ),
        "calibrated_ratio_target_quantile": float(
            np.quantile(truth_values / adjusted_values, calibration.target_quantile)
        ),
        "raw_coverage_at_1sigma": float(np.mean(truth_values <= raw_values)),
        "calibrated_coverage_at_1sigma": float(np.mean(truth_values <= adjusted_values)),
    }


def save_candidate_sigma_calibration(
    calibration: CandidateSigmaCalibration,
    path: Path,
) -> Path:
    """Write calibration JSON."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(calibration), indent=2, sort_keys=True), encoding="utf-8")
    return path


def load_candidate_sigma_calibration(path: Path) -> CandidateSigmaCalibration:
    """Load calibration JSON."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    schema = str(payload.get("schema", CALIBRATION_SCHEMA))
    if schema != CALIBRATION_SCHEMA:
        raise ValueError(f"unsupported candidate sigma calibration schema: {schema!r}")
    return CandidateSigmaCalibration(
        schema=schema,
        input_sigma_column=str(payload.get("input_sigma_column", "predicted_sigma_m")),
        branch_column=str(payload.get("branch_column", "candidate_branch")),
        target_quantile=float(payload.get("target_quantile", 0.5)),
        min_group_rows=int(payload.get("min_group_rows", 20)),
        shrinkage_rows=float(payload.get("shrinkage_rows", 50.0)),
        scale_min=float(payload.get("scale_min", 0.25)),
        scale_max=float(payload.get("scale_max", 4.0)),
        calibration_row_count=int(payload.get("calibration_row_count", 0)),
        global_scale=float(payload["global_scale"]),
        source_scales={str(key): float(value) for key, value in payload.get("source_scales", {}).items()},
        branch_scales={str(key): float(value) for key, value in payload.get("branch_scales", {}).items()},
        source_branch_scales={
            str(key): float(value)
            for key, value in payload.get("source_branch_scales", {}).items()
        },
        source_counts={str(key): int(value) for key, value in payload.get("source_counts", {}).items()},
        branch_counts={str(key): int(value) for key, value in payload.get("branch_counts", {}).items()},
        source_branch_counts={
            str(key): int(value)
            for key, value in payload.get("source_branch_counts", {}).items()
        },
    )


def fit_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="raft-uav-mmuad-fit-candidate-sigma-calibration",
        description="fit train-only branch/source calibration for candidate uncertainty",
    )
    parser.add_argument("--candidates-csv", type=Path, required=True)
    parser.add_argument("--truth-csv", type=Path, required=True)
    parser.add_argument("--calibration-json", type=Path, required=True)
    parser.add_argument("--features-csv", type=Path)
    parser.add_argument("--summary-json", type=Path)
    parser.add_argument("--input-sigma-column", default="predicted_sigma_m")
    parser.add_argument("--branch-column", default="candidate_branch")
    parser.add_argument("--target-quantile", type=float, default=0.5)
    parser.add_argument("--min-group-rows", type=int, default=20)
    parser.add_argument("--shrinkage-rows", type=float, default=50.0)
    parser.add_argument("--scale-min", type=float, default=0.25)
    parser.add_argument("--scale-max", type=float, default=4.0)
    parser.add_argument("--max-truth-time-delta-s", type=float, default=0.5)
    args = parser.parse_args(argv)

    candidates = load_candidate_file(args.candidates_csv)
    truth = load_evaluation_truth_file(args.truth_csv).rows
    features = build_cluster_feature_table(
        candidates,
        truth=truth,
        max_truth_time_delta_s=float(args.max_truth_time_delta_s),
    )
    calibration = fit_candidate_sigma_calibration(
        features,
        input_sigma_column=args.input_sigma_column,
        branch_column=args.branch_column,
        target_quantile=args.target_quantile,
        min_group_rows=args.min_group_rows,
        shrinkage_rows=args.shrinkage_rows,
        scale_min=args.scale_min,
        scale_max=args.scale_max,
    )
    save_candidate_sigma_calibration(calibration, args.calibration_json)
    if args.features_csv is not None:
        args.features_csv.parent.mkdir(parents=True, exist_ok=True)
        features.to_csv(args.features_csv, index=False)
    summary = candidate_sigma_calibration_summary(features, calibration)
    if args.summary_json is not None:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("mmuad_candidate_sigma_calibration_fit=ok")
    print(f"calibration_json={args.calibration_json}")
    print(f"calibration_rows={calibration.calibration_row_count}")
    return 0


def apply_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="raft-uav-mmuad-apply-candidate-sigma-calibration",
        description="apply train-fitted branch/source candidate uncertainty calibration",
    )
    parser.add_argument("--candidates-csv", type=Path, required=True)
    parser.add_argument("--calibration-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--input-sigma-column")
    parser.add_argument("--output-sigma-column", default="calibrated_sigma_m")
    parser.add_argument("--replace-covariance", action="store_true")
    parser.add_argument("--z-scale", type=float, default=1.0)
    args = parser.parse_args(argv)

    calibration = load_candidate_sigma_calibration(args.calibration_json)
    calibrated = apply_candidate_sigma_calibration(
        load_candidate_file(args.candidates_csv),
        calibration,
        input_sigma_column=args.input_sigma_column,
        output_sigma_column=args.output_sigma_column,
        replace_covariance=args.replace_covariance,
        z_scale=args.z_scale,
    )
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    calibrated.rows.to_csv(args.output_csv, index=False)
    print("mmuad_candidate_sigma_calibration_apply=ok")
    print(f"output_csv={args.output_csv}")
    print(f"output_rows={len(calibrated.rows)}")
    return 0


def _candidate_rows(candidates: CandidateFrame | pd.DataFrame) -> pd.DataFrame:
    rows = candidates.rows.copy() if isinstance(candidates, CandidateFrame) else pd.DataFrame(candidates).copy()
    return normalize_candidate_columns(rows)


def _normalized_group_values(
    values: pd.Series | None,
    *,
    index: pd.Index,
    fallback: str,
) -> pd.Series:
    if values is None:
        values = pd.Series(fallback, index=index)
    text = values.where(values.notna(), fallback).astype(str).str.strip()
    missing = text.eq("") | text.str.lower().isin({"nan", "none", "<na>"})
    return text.where(~missing, fallback)


def _raw_scale(
    ratios: pd.Series,
    *,
    quantile: float,
    scale_min: float,
    scale_max: float,
) -> float:
    values = pd.to_numeric(ratios, errors="coerce").dropna().to_numpy(float)
    values = values[np.isfinite(values) & (values >= 0.0)]
    if values.size == 0:
        return 1.0
    raw = float(np.quantile(values, float(quantile)))
    return float(np.clip(raw, float(scale_min), float(scale_max)))


def _fit_group_scales(
    rows: pd.DataFrame,
    *,
    group_column: str,
    parent_scale: Any,
    quantile: float,
    min_group_rows: int,
    shrinkage_rows: float,
    scale_min: float,
    scale_max: float,
) -> tuple[dict[str, float], dict[str, int]]:
    scales: dict[str, float] = {}
    counts: dict[str, int] = {}
    for key, group in rows.groupby(group_column, sort=True):
        key_text = str(key)
        count = int(len(group))
        counts[key_text] = count
        if count < int(min_group_rows):
            continue
        raw = _raw_scale(
            group["_sigma_ratio"],
            quantile=quantile,
            scale_min=scale_min,
            scale_max=scale_max,
        )
        parent = float(parent_scale(key_text))
        weight = 1.0 if float(shrinkage_rows) <= 0.0 else count / (count + float(shrinkage_rows))
        shrunk = float(np.exp(weight * np.log(raw) + (1.0 - weight) * np.log(parent)))
        scales[key_text] = float(np.clip(shrunk, float(scale_min), float(scale_max)))
    return scales, counts


def _source_branch_key(source: str, branch: str) -> str:
    return json.dumps([str(source), str(branch)], separators=(",", ":"))


def _decode_source_branch_key(key: str) -> tuple[str, str]:
    value = json.loads(str(key))
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError(f"invalid source-branch calibration key: {key!r}")
    return str(value[0]), str(value[1])


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(fit_main())
