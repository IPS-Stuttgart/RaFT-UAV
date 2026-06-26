"""Post-ranker candidate-pull refinement for MMUAD Track 5 results."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import ast
import json
from pathlib import Path
from typing import Any, Literal
import zipfile

import numpy as np
import pandas as pd


CandidatePullPolicy = Literal["none", "constant", "feature-rule-v2"]
CandidatePullSmoother = Literal["none", "rts"]


@dataclass(frozen=True)
class CandidatePullConfig:
    """Configuration for pulling official-style rows toward top candidates."""

    policy: CandidatePullPolicy = "none"
    smoother: CandidatePullSmoother = "none"
    constant_alpha_xy: float = 1.0
    constant_alpha_z: float = 1.0
    top_k: int = 5
    time_tolerance_s: float = 0.5
    measurement_base_std_m: float = 0.1
    measurement_dispersion_weight: float = 0.0
    measurement_margin_weight: float = 0.25
    measurement_cross_sensor_weight: float = 0.0
    rts_accel_std_mps2: float = 0.02


@dataclass(frozen=True)
class CandidatePullResult:
    """Post-processed official rows plus diagnostic artifacts."""

    rows: pd.DataFrame
    centers: pd.DataFrame
    sequence_features: pd.DataFrame
    alpha_assignments: pd.DataFrame
    provenance: dict[str, Any]


def refine_official_results_with_candidate_pull(
    results: pd.DataFrame,
    candidates: pd.DataFrame,
    *,
    config: CandidatePullConfig | None = None,
) -> CandidatePullResult:
    """Pull official result positions toward top candidate detections.

    This is a post-processing method for train-selected ablations.  It uses only
    result rows and candidate metadata at inference time; labels are not read.
    """

    config = config or CandidatePullConfig()
    result_rows, current_xyz = _normalize_official_results(results)
    centers = candidate_centers_for_results(
        candidates,
        result_rows,
        current_xyz,
        top_k=config.top_k,
        time_tolerance_s=config.time_tolerance_s,
    )
    aligned = align_rowwise_candidate_centers(result_rows, centers)
    uncertainty = topk_candidate_centers(candidates, top_k=config.top_k)
    uncertainty_aligned = align_candidate_centers(
        result_rows,
        uncertainty,
        time_tolerance_s=config.time_tolerance_s,
    )
    for column in (
        "topk_dispersion_m",
        "top_score_margin",
        "nearest_cross_sensor_distance_m",
        "cross_sensor_neighbor_count",
        "frame_source_count",
        "topk_candidate_count",
    ):
        if column in uncertainty_aligned.columns:
            aligned[column] = uncertainty_aligned[column]
    sequence_features = candidate_pull_sequence_features(aligned)
    alpha_assignments = assign_candidate_pull_alphas(
        sequence_features,
        policy=config.policy,
        constant_alpha_xy=config.constant_alpha_xy,
        constant_alpha_z=config.constant_alpha_z,
    )
    pulled_xyz = apply_candidate_pull(aligned, current_xyz, alpha_assignments)
    if config.smoother == "rts":
        aligned = aligned.copy()
        aligned["meas_x"] = pulled_xyz[:, 0]
        aligned["meas_y"] = pulled_xyz[:, 1]
        aligned["meas_z"] = pulled_xyz[:, 2]
        aligned["measurement_std_m"] = [
            measurement_std(
                row,
                base=config.measurement_base_std_m,
                disp_w=config.measurement_dispersion_weight,
                margin_w=config.measurement_margin_weight,
                cross_w=config.measurement_cross_sensor_weight,
            )
            for _, row in aligned.iterrows()
        ]
        final_xyz = smooth_all_rts(
            aligned,
            accel_std=config.rts_accel_std_mps2,
        )
    elif config.smoother == "none":
        final_xyz = pulled_xyz
    else:
        raise ValueError("candidate-pull smoother must be 'none' or 'rts'")
    out = result_rows.copy()
    out["Position"] = [format_position(row) for row in final_xyz]
    provenance = {
        "schema": "raft-uav-mmuad-candidate-pull-provenance-v1",
        "policy": config.policy,
        "smoother": config.smoother,
        "constant_alpha_xy": float(config.constant_alpha_xy),
        "constant_alpha_z": float(config.constant_alpha_z),
        "top_k": int(config.top_k),
        "time_tolerance_s": float(config.time_tolerance_s),
        "measurement_base_std_m": float(config.measurement_base_std_m),
        "measurement_dispersion_weight": float(config.measurement_dispersion_weight),
        "measurement_margin_weight": float(config.measurement_margin_weight),
        "measurement_cross_sensor_weight": float(config.measurement_cross_sensor_weight),
        "rts_accel_std_mps2": float(config.rts_accel_std_mps2),
        "row_count": int(len(out)),
        "candidate_center_count": int(len(centers)),
        "matched_candidate_center_count": int(aligned["top1_x"].notna().sum()),
    }
    return CandidatePullResult(
        rows=out[["Sequence", "Timestamp", "Position", "Classification"]],
        centers=centers,
        sequence_features=sequence_features,
        alpha_assignments=alpha_assignments,
        provenance=provenance,
    )


def topk_candidate_centers(candidates: pd.DataFrame, *, top_k: int = 5) -> pd.DataFrame:
    """Return one top-candidate/uncertainty row per sequence timestamp."""

    rows = pd.DataFrame(candidates).copy()
    if rows.empty:
        return _empty_centers()
    _rename_candidate_columns(rows)
    score_column = _first_existing_column(
        rows,
        ("ranker_score", "cluster_ranker_score", "candidate_ranker_score", "confidence", "score"),
    )
    rows["_candidate_pull_score"] = (
        pd.to_numeric(rows[score_column], errors="coerce").fillna(0.0)
        if score_column
        else 0.0
    )
    if "confidence" in rows.columns:
        rows["_candidate_pull_confidence"] = pd.to_numeric(
            rows["confidence"],
            errors="coerce",
        ).fillna(0.0)
    else:
        rows["_candidate_pull_confidence"] = 0.0
    for column in ("Sequence", "Timestamp", "x_m", "y_m", "z_m"):
        if column not in rows.columns:
            raise ValueError(f"candidate rows missing required column {column!r}")
    for column in ("Timestamp", "x_m", "y_m", "z_m"):
        rows[column] = pd.to_numeric(rows[column], errors="coerce")
    finite = np.isfinite(rows[["Timestamp", "x_m", "y_m", "z_m"]].to_numpy(float)).all(axis=1)
    rows = rows.loc[finite].copy()
    records: list[dict[str, Any]] = []
    for (sequence, timestamp), group in rows.groupby(["Sequence", "Timestamp"], sort=True):
        top = group.sort_values(
            ["_candidate_pull_score", "_candidate_pull_confidence"],
            ascending=[False, False],
        ).head(max(int(top_k), 1))
        xyz = top[["x_m", "y_m", "z_m"]].to_numpy(float)
        if len(top) == 0:
            continue
        scores = top["_candidate_pull_score"].to_numpy(float)
        weights = np.clip(scores, 0.0, None)
        if float(weights.sum()) <= 1.0e-12:
            weights = np.ones(len(top), dtype=float) / float(len(top))
        else:
            weights = weights / float(weights.sum())
        center = (xyz * weights[:, None]).sum(axis=0)
        dists = np.linalg.norm(xyz - center.reshape(1, 3), axis=1)
        score_margin = float(scores[0] - scores[1]) if len(scores) > 1 else float(scores[0])
        records.append(
            {
                "Sequence": str(sequence),
                "candidate_time_s": float(timestamp),
                "top1_x": float(xyz[0, 0]),
                "top1_y": float(xyz[0, 1]),
                "top1_z": float(xyz[0, 2]),
                "weighted5_x": float(center[0]),
                "weighted5_y": float(center[1]),
                "weighted5_z": float(center[2]),
                "dispersion5": float(np.mean(dists)),
                "topk_dispersion_m": float(np.mean(dists)),
                "topk_dispersion_p95_m": float(np.percentile(dists, 95.0)),
                "top_score": float(scores[0]),
                "top_score_margin": score_margin,
                "nearest_cross_sensor_distance_m": _top_numeric(
                    top,
                    "nearest_cross_sensor_distance_m",
                    10.0,
                ),
                "cross_sensor_neighbor_count": _top_numeric(
                    top,
                    "cross_sensor_neighbor_count",
                    0.0,
                ),
                "frame_source_count": _top_numeric(top, "frame_source_count", 1.0),
                "topk_candidate_count": int(len(top)),
            }
        )
    return pd.DataFrame.from_records(records) if records else _empty_centers()


def candidate_centers_for_results(
    candidates: pd.DataFrame,
    results: pd.DataFrame,
    current_xyz: np.ndarray,
    *,
    top_k: int = 5,
    time_tolerance_s: float = 0.5,
) -> pd.DataFrame:
    """Return row-wise candidate centers in a time window around each result row."""

    rows = pd.DataFrame(candidates).copy()
    if rows.empty:
        return _empty_centers()
    _rename_candidate_columns(rows)
    for column in ("Timestamp", "x_m", "y_m", "z_m"):
        if column not in rows.columns:
            raise ValueError(f"candidate rows missing required column {column!r}")
        rows[column] = pd.to_numeric(rows[column], errors="coerce")
    for column in ("ranker_score", "confidence", "cluster_point_count"):
        if column in rows.columns:
            rows[column] = pd.to_numeric(rows[column], errors="coerce")
        else:
            rows[column] = np.nan
    finite = np.isfinite(rows[["Timestamp", "x_m", "y_m", "z_m"]].to_numpy(float)).all(axis=1)
    rows = rows.loc[finite].copy()
    records: list[dict[str, Any]] = []
    top_k = max(int(top_k), 1)
    for sequence, seq_results in results.groupby("Sequence", sort=True):
        seq_candidates = rows.loc[rows["Sequence"].astype(str) == str(sequence)].sort_values(
            "Timestamp"
        )
        if seq_candidates.empty:
            continue
        times = seq_candidates["Timestamp"].to_numpy(float)
        xyz = seq_candidates[["x_m", "y_m", "z_m"]].to_numpy(float)
        score = seq_candidates["ranker_score"].fillna(0.0).clip(lower=0.0).to_numpy(float)
        confidence = seq_candidates["confidence"].fillna(0.0).to_numpy(float)
        point_count = seq_candidates["cluster_point_count"].fillna(0.0).to_numpy(float)
        for row_index in seq_results.index.to_numpy():
            timestamp = float(results.at[row_index, "Timestamp"])
            mask = np.abs(times - timestamp) <= float(time_tolerance_s)
            if not mask.any():
                continue
            candidate_indices = np.where(mask)[0]
            order = np.lexsort(
                (
                    -point_count[candidate_indices],
                    -confidence[candidate_indices],
                    -score[candidate_indices],
                )
            )
            candidate_indices = candidate_indices[order][: max(top_k * 4, top_k)]
            cxyz = xyz[candidate_indices]
            if len(cxyz) == 0 or not np.isfinite(cxyz).all():
                continue
            cscore = np.clip(score[candidate_indices], 0.0, None)
            weights = (
                cscore / cscore.sum()
                if float(cscore.sum()) > 1.0e-12
                else np.ones(len(candidate_indices), dtype=float) / float(len(candidate_indices))
            )
            top1 = cxyz[0]
            weighted5 = _weighted_or_mean(cxyz, weights, min(top_k, 5))
            weighted10 = _weighted_or_mean(cxyz, weights, min(max(top_k, 10), len(cxyz)))
            median10 = np.median(cxyz[: min(10, len(cxyz))], axis=0)
            current = current_xyz[int(row_index)]
            dist_current = np.linalg.norm(cxyz - current.reshape(1, 3), axis=1)
            nearest_current = cxyz[int(np.argmin(dist_current))]
            dispersion5 = float(
                np.mean(np.linalg.norm(cxyz[: min(5, len(cxyz))] - weighted5.reshape(1, 3), axis=1))
            )
            records.append(
                {
                    "row_index": int(row_index),
                    "Sequence": str(sequence),
                    "candidate_time_s": timestamp,
                    "candidate_count": int(mask.sum()),
                    "top1_x": float(top1[0]),
                    "top1_y": float(top1[1]),
                    "top1_z": float(top1[2]),
                    "weighted5_x": float(weighted5[0]),
                    "weighted5_y": float(weighted5[1]),
                    "weighted5_z": float(weighted5[2]),
                    "weighted10_x": float(weighted10[0]),
                    "weighted10_y": float(weighted10[1]),
                    "weighted10_z": float(weighted10[2]),
                    "median10_x": float(median10[0]),
                    "median10_y": float(median10[1]),
                    "median10_z": float(median10[2]),
                    "nearest_current_x": float(nearest_current[0]),
                    "nearest_current_y": float(nearest_current[1]),
                    "nearest_current_z": float(nearest_current[2]),
                    "dispersion5": dispersion5,
                    "topk_dispersion_m": dispersion5,
                    "topk_dispersion_p95_m": float(
                        np.percentile(
                            np.linalg.norm(
                                cxyz[: min(5, len(cxyz))] - weighted5.reshape(1, 3),
                                axis=1,
                            ),
                            95.0,
                        )
                    ),
                    "top_score": float(cscore[0]) if len(cscore) else 0.0,
                    "top_score_margin": float(cscore[0] - cscore[1])
                    if len(cscore) > 1
                    else float(cscore[0])
                    if len(cscore)
                    else 0.0,
                    "weighted5_distance_to_current": float(np.linalg.norm(weighted5 - current)),
                    "nearest_candidate_distance_to_current": float(np.min(dist_current)),
                    "topk_candidate_count": int(min(top_k, len(cxyz))),
                }
            )
    return pd.DataFrame.from_records(records) if records else _empty_centers()


def align_rowwise_candidate_centers(results: pd.DataFrame, centers: pd.DataFrame) -> pd.DataFrame:
    """Attach row-wise candidate centers by ``row_index``."""

    aligned = results.copy()
    for column in centers.columns:
        if column in {"Sequence", "candidate_time_s"}:
            continue
        if column == "row_index":
            continue
        aligned[column] = np.nan
    if centers.empty or "row_index" not in centers.columns:
        return aligned
    indexed = centers.set_index("row_index")
    for column in indexed.columns:
        if column in {"Sequence"}:
            continue
        aligned.loc[indexed.index, column] = indexed[column].to_numpy()
    return aligned


def align_candidate_centers(
    results: pd.DataFrame,
    centers: pd.DataFrame,
    *,
    time_tolerance_s: float,
) -> pd.DataFrame:
    """Align per-frame candidate centers to official result rows."""

    parts: list[pd.DataFrame] = []
    for sequence, group in results.groupby("Sequence", sort=True):
        left = group.sort_values("Timestamp").copy()
        right = centers.loc[centers["Sequence"].astype(str) == str(sequence)].sort_values(
            "candidate_time_s"
        )
        if right.empty:
            for column in centers.columns:
                if column not in {"Sequence", "candidate_time_s"}:
                    left[column] = np.nan
            left["candidate_time_s"] = np.nan
            parts.append(left)
            continue
        right = right.drop(columns=["Sequence"], errors="ignore")
        parts.append(
            pd.merge_asof(
                left,
                right,
                left_on="Timestamp",
                right_on="candidate_time_s",
                direction="nearest",
                tolerance=float(time_tolerance_s),
            )
        )
    return pd.concat(parts, ignore_index=True) if parts else results.iloc[0:0].copy()


def candidate_pull_sequence_features(aligned: pd.DataFrame) -> pd.DataFrame:
    """Aggregate non-oracle sequence features used by alpha rules."""

    rows = aligned.copy()
    for column in (
        "top_score",
        "dispersion5",
        "weighted5_x",
        "weighted5_y",
        "weighted5_z",
        "current_x",
        "current_y",
        "current_z",
    ):
        if column in rows.columns:
            rows[column] = pd.to_numeric(rows[column], errors="coerce")
    position_columns = {
        "weighted5_x",
        "weighted5_y",
        "weighted5_z",
        "current_x",
        "current_y",
        "current_z",
    }
    if position_columns.issubset(rows.columns):
        rows["weighted5_distance_to_current"] = np.linalg.norm(
            rows[["weighted5_x", "weighted5_y", "weighted5_z"]].to_numpy(float)
            - rows[["current_x", "current_y", "current_z"]].to_numpy(float),
            axis=1,
        )
    else:
        rows["weighted5_distance_to_current"] = np.nan
    records = []
    for sequence, group in rows.groupby("Sequence", sort=True):
        records.append(
            {
                "Sequence": str(sequence),
                "row_count": int(len(group)),
                "matched_candidate_rows": int(group["top1_x"].notna().sum())
                if "top1_x" in group
                else 0,
                "dispersion5_mean": _mean(group.get("dispersion5")),
                "dispersion5_median": _median(group.get("dispersion5")),
                "top_score_mean": _mean(group.get("top_score")),
                "top_score_median": _median(group.get("top_score")),
                "current_distance_mean": _mean(group.get("weighted5_distance_to_current")),
                "current_distance_median": _median(group.get("weighted5_distance_to_current")),
            }
        )
    return pd.DataFrame.from_records(records)


def assign_candidate_pull_alphas(
    sequence_features: pd.DataFrame,
    *,
    policy: CandidatePullPolicy,
    constant_alpha_xy: float = 1.0,
    constant_alpha_z: float = 1.0,
) -> pd.DataFrame:
    """Assign per-sequence alpha values from observable sequence features."""

    records: list[dict[str, Any]] = []
    for row in sequence_features.itertuples(index=False):
        if policy == "none":
            alpha_xy, alpha_z, reason = 0.0, 0.0, "disabled"
        elif policy == "constant":
            alpha_xy = float(constant_alpha_xy)
            alpha_z = float(constant_alpha_z)
            reason = "constant"
        elif policy == "feature-rule-v2":
            alpha_xy, alpha_z, reason = _feature_rule_v2_alpha(row)
        else:
            raise ValueError(
                "candidate-pull policy must be 'none', 'constant', or 'feature-rule-v2'"
            )
        records.append(
            {
                "Sequence": str(row.Sequence),
                "candidate_pull_alpha_xy": float(alpha_xy),
                "candidate_pull_alpha_z": float(alpha_z),
                "candidate_pull_reason": reason,
            }
        )
    return pd.DataFrame.from_records(records)


def apply_candidate_pull(
    aligned: pd.DataFrame,
    current_xyz: np.ndarray,
    alpha_assignments: pd.DataFrame,
) -> np.ndarray:
    """Return candidate-pulled measurements before optional smoothing."""

    out = np.asarray(current_xyz, dtype=float).copy()
    if out.size == 0 or alpha_assignments.empty:
        return out
    alpha = alpha_assignments.set_index("Sequence")
    for index, row in aligned.iterrows():
        sequence = str(row["Sequence"])
        if sequence not in alpha.index:
            continue
        if not np.isfinite([row.get("top1_x"), row.get("top1_y"), row.get("top1_z")]).all():
            continue
        alpha_row = alpha.loc[sequence]
        dx = float(row["top1_x"]) - out[index, 0]
        dy = float(row["top1_y"]) - out[index, 1]
        dz = float(row["top1_z"]) - out[index, 2]
        out[index, 0] += float(alpha_row["candidate_pull_alpha_xy"]) * dx
        out[index, 1] += float(alpha_row["candidate_pull_alpha_xy"]) * dy
        out[index, 2] += float(alpha_row["candidate_pull_alpha_z"]) * dz
    return out


def measurement_std(
    row: pd.Series,
    *,
    base: float,
    disp_w: float,
    margin_w: float,
    cross_w: float,
) -> float:
    """Return per-row measurement standard deviation for RTS smoothing."""

    dispersion = _finite_float(row.get("topk_dispersion_m"), 5.0)
    margin = _finite_float(row.get("top_score_margin"), 0.0)
    cross = _finite_float(row.get("nearest_cross_sensor_distance_m"), 10.0)
    neighbors = _finite_float(row.get("cross_sensor_neighbor_count"), 0.0)
    margin_penalty = 1.0 / (0.05 + max(0.0, margin))
    neighbor_bonus = 1.0 / (1.0 + max(0.0, neighbors))
    std = float(base) + float(disp_w) * dispersion
    std += float(margin_w) * margin_penalty
    std += float(cross_w) * min(cross, 20.0) * neighbor_bonus
    return float(np.clip(std, 0.5, 80.0))


def smooth_all_rts(aligned: pd.DataFrame, *, accel_std: float) -> np.ndarray:
    """Apply constant-velocity RTS smoothing per sequence."""

    xyz_parts = []
    for _sequence, group in aligned.groupby("Sequence", sort=True):
        group = group.sort_values("Timestamp")
        z = group[["meas_x", "meas_y", "meas_z"]].to_numpy(float)
        times = group["Timestamp"].to_numpy(float)
        r_std = group["measurement_std_m"].to_numpy(float)
        smoothed = cv_rts_smooth(times, z, r_std, accel_std=accel_std)
        xyz_parts.append(pd.DataFrame(smoothed, index=group.index, columns=["x", "y", "z"]))
    if not xyz_parts:
        return np.empty((0, 3), dtype=float)
    return pd.concat(xyz_parts).sort_index()[["x", "y", "z"]].to_numpy(float)


def cv_rts_smooth(
    times: np.ndarray,
    z: np.ndarray,
    r_std: np.ndarray,
    accel_std: float,
) -> np.ndarray:
    """Constant-velocity Kalman filter plus Rauch-Tung-Striebel smoother."""

    n = len(times)
    if n <= 1:
        return z.copy()
    state_dim = 6
    h = np.zeros((3, state_dim), dtype=float)
    h[:, :3] = np.eye(3)
    eye = np.eye(state_dim)
    x_f = np.zeros((n, state_dim), dtype=float)
    p_f = np.zeros((n, state_dim, state_dim), dtype=float)
    x_p = np.zeros_like(x_f)
    p_p = np.zeros_like(p_f)
    x = np.zeros(state_dim, dtype=float)
    x[:3] = z[0]
    dt0 = max(float(times[1] - times[0]), 1.0e-3)
    x[3:] = (z[1] - z[0]) / dt0
    p = np.diag([float(r_std[0]) ** 2] * 3 + [100.0] * 3)
    for idx in range(n):
        if idx == 0:
            f = eye.copy()
            q = np.zeros((state_dim, state_dim), dtype=float)
        else:
            dt = max(float(times[idx] - times[idx - 1]), 1.0e-3)
            f = eye.copy()
            f[:3, 3:] = np.eye(3) * dt
            accel_var = float(accel_std) ** 2
            q = np.zeros((state_dim, state_dim), dtype=float)
            q[:3, :3] = np.eye(3) * accel_var * dt**4 / 4.0
            q[:3, 3:] = np.eye(3) * accel_var * dt**3 / 2.0
            q[3:, :3] = np.eye(3) * accel_var * dt**3 / 2.0
            q[3:, 3:] = np.eye(3) * accel_var * dt**2
            x = f @ x
            p = f @ p @ f.T + q
        x_p[idx] = x
        p_p[idx] = p
        r = np.eye(3) * float(r_std[idx]) ** 2
        innovation = z[idx] - h @ x
        s = h @ p @ h.T + r
        k = p @ h.T @ np.linalg.pinv(s)
        x = x + k @ innovation
        p = (eye - k @ h) @ p
        x_f[idx] = x
        p_f[idx] = p
    x_s = x_f.copy()
    p_s = p_f.copy()
    for idx in range(n - 2, -1, -1):
        dt = max(float(times[idx + 1] - times[idx]), 1.0e-3)
        f = eye.copy()
        f[:3, 3:] = np.eye(3) * dt
        gain = p_f[idx] @ f.T @ np.linalg.pinv(p_p[idx + 1])
        x_s[idx] = x_f[idx] + gain @ (x_s[idx + 1] - x_p[idx + 1])
        p_s[idx] = p_f[idx] + gain @ (p_s[idx + 1] - p_p[idx + 1]) @ gain.T
    return x_s[:, :3]


def write_candidate_pull_artifacts(
    result: CandidatePullResult,
    *,
    results_csv: Path,
    submission_zip: Path | None = None,
    provenance_json: Path | None = None,
    centers_csv: Path | None = None,
    sequence_features_csv: Path | None = None,
    alpha_assignments_csv: Path | None = None,
) -> dict[str, str]:
    """Write result CSV/ZIP plus diagnostics."""

    paths: dict[str, str] = {}
    results_csv.parent.mkdir(parents=True, exist_ok=True)
    result.rows.to_csv(results_csv, index=False)
    paths["results_csv"] = str(results_csv)
    if submission_zip is not None:
        submission_zip.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(submission_zip, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.write(results_csv, arcname="mmaud_results.csv")
        paths["submission_zip"] = str(submission_zip)
    if provenance_json is not None:
        provenance_json.parent.mkdir(parents=True, exist_ok=True)
        provenance_json.write_text(
            json.dumps(_jsonable(result.provenance), indent=2),
            encoding="utf-8",
        )
        paths["provenance_json"] = str(provenance_json)
    if centers_csv is not None:
        centers_csv.parent.mkdir(parents=True, exist_ok=True)
        result.centers.to_csv(centers_csv, index=False)
        paths["centers_csv"] = str(centers_csv)
    if sequence_features_csv is not None:
        sequence_features_csv.parent.mkdir(parents=True, exist_ok=True)
        result.sequence_features.to_csv(sequence_features_csv, index=False)
        paths["sequence_features_csv"] = str(sequence_features_csv)
    if alpha_assignments_csv is not None:
        alpha_assignments_csv.parent.mkdir(parents=True, exist_ok=True)
        result.alpha_assignments.to_csv(alpha_assignments_csv, index=False)
        paths["alpha_assignments_csv"] = str(alpha_assignments_csv)
    return paths


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint for candidate-pull Track 5 postprocessing.

    The command intentionally accepts only existing result rows and candidate
    metadata.  Ground-truth/reference files belong in separate scoring commands.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-in", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--results-out", type=Path, required=True)
    parser.add_argument("--submission-zip", type=Path)
    parser.add_argument("--provenance-json", type=Path)
    parser.add_argument("--centers-csv", type=Path)
    parser.add_argument("--sequence-features-csv", type=Path)
    parser.add_argument("--alpha-assignments-csv", type=Path)
    parser.add_argument(
        "--candidate-pull-policy",
        choices=("none", "constant", "feature-rule-v2"),
        default="feature-rule-v2",
    )
    parser.add_argument("--candidate-pull-alpha-xy", type=float, default=1.0)
    parser.add_argument("--candidate-pull-alpha-z", type=float, default=1.0)
    parser.add_argument("--candidate-pull-top-k", type=int, default=5)
    parser.add_argument("--candidate-pull-time-tolerance-s", type=float, default=0.5)
    parser.add_argument(
        "--candidate-pull-smoother",
        choices=("none", "rts"),
        default="rts",
    )
    parser.add_argument("--candidate-pull-measurement-base-std-m", type=float, default=0.1)
    parser.add_argument("--candidate-pull-measurement-dispersion-weight", type=float, default=0.0)
    parser.add_argument("--candidate-pull-measurement-margin-weight", type=float, default=0.25)
    parser.add_argument("--candidate-pull-measurement-cross-sensor-weight", type=float, default=0.0)
    parser.add_argument("--candidate-pull-rts-accel-std-mps2", type=float, default=0.02)
    args = parser.parse_args(argv)

    config = CandidatePullConfig(
        policy=args.candidate_pull_policy,
        smoother=args.candidate_pull_smoother,
        constant_alpha_xy=args.candidate_pull_alpha_xy,
        constant_alpha_z=args.candidate_pull_alpha_z,
        top_k=args.candidate_pull_top_k,
        time_tolerance_s=args.candidate_pull_time_tolerance_s,
        measurement_base_std_m=args.candidate_pull_measurement_base_std_m,
        measurement_dispersion_weight=args.candidate_pull_measurement_dispersion_weight,
        measurement_margin_weight=args.candidate_pull_measurement_margin_weight,
        measurement_cross_sensor_weight=args.candidate_pull_measurement_cross_sensor_weight,
        rts_accel_std_mps2=args.candidate_pull_rts_accel_std_mps2,
    )
    result = refine_official_results_with_candidate_pull(
        pd.read_csv(args.results_in),
        pd.read_csv(args.candidates),
        config=config,
    )
    paths = write_candidate_pull_artifacts(
        result,
        results_csv=args.results_out,
        submission_zip=args.submission_zip,
        provenance_json=args.provenance_json,
        centers_csv=args.centers_csv,
        sequence_features_csv=args.sequence_features_csv,
        alpha_assignments_csv=args.alpha_assignments_csv,
    )
    print("mmuad_candidate_pull_postprocess=ok")
    for key, value in paths.items():
        print(f"{key}={value}")
    return 0


def _feature_rule_v2_alpha(row: object) -> tuple[float, float, str]:
    top_score = _finite_float(getattr(row, "top_score_mean", np.nan), np.nan)
    dispersion = _finite_float(getattr(row, "dispersion5_mean", np.nan), np.nan)
    current_distance = _finite_float(getattr(row, "current_distance_mean", np.nan), np.nan)
    if top_score < 0.65 and dispersion < 0.10:
        return -0.5, 0.5, "ultra_compact_low_score"
    if top_score < 0.67 and dispersion < 1.50:
        return 0.5, 0.5, "compact_low_score"
    if top_score < 0.70 and dispersion >= 1.50:
        return 0.75, 1.0, "dispersed_low_score"
    if current_distance > 20.0:
        return 1.2, 1.1, "large_current_to_candidate_gap"
    if 0.80 <= top_score < 0.90 and dispersion < 1.0 and current_distance < 4.0:
        return 0.25, 0.5, "low_gap_mid_confidence"
    if 0.70 <= top_score < 0.80 and dispersion >= 1.50 and current_distance < 8.0:
        return 1.35, 1.0, "moderate_dispersed_candidate_cloud"
    return 1.1, 1.0, "default_pull"


def _normalize_official_results(results: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    rows = pd.DataFrame(results).copy()
    missing = {"Sequence", "Timestamp", "Position", "Classification"}.difference(rows.columns)
    if missing:
        raise ValueError(f"official results missing required columns: {sorted(missing)}")
    xyz = (
        np.vstack(rows["Position"].map(parse_position).to_numpy())
        if len(rows)
        else np.empty((0, 3))
    )
    rows = rows[["Sequence", "Timestamp", "Position", "Classification"]].copy()
    rows["Sequence"] = rows["Sequence"].astype(str)
    rows["Timestamp"] = pd.to_numeric(rows["Timestamp"], errors="coerce")
    rows["current_x"] = xyz[:, 0] if len(rows) else []
    rows["current_y"] = xyz[:, 1] if len(rows) else []
    rows["current_z"] = xyz[:, 2] if len(rows) else []
    return rows, xyz


def parse_position(value: object) -> np.ndarray:
    """Parse official Track 5 ``Position`` text into ``x, y, z``."""

    text = str(value).strip().replace("(", "[").replace(")", "]")
    return np.asarray(ast.literal_eval(text), dtype=float)


def format_position(xyz: np.ndarray) -> str:
    """Format an official Track 5 ``Position`` cell."""

    arr = np.asarray(xyz, dtype=float).reshape(3)
    return f"({arr[0]:.12g},{arr[1]:.12g},{arr[2]:.12g})"


def _rename_candidate_columns(rows: pd.DataFrame) -> None:
    rename = {}
    if "Sequence" not in rows.columns and "sequence_id" in rows.columns:
        rename["sequence_id"] = "Sequence"
    if "Timestamp" not in rows.columns and "time_s" in rows.columns:
        rename["time_s"] = "Timestamp"
    rows.rename(columns=rename, inplace=True)
    if "Sequence" in rows.columns:
        rows["Sequence"] = rows["Sequence"].astype(str)


def _first_existing_column(rows: pd.DataFrame, names: tuple[str, ...]) -> str | None:
    for name in names:
        if name in rows.columns:
            return name
    return None


def _top_numeric(rows: pd.DataFrame, column: str, default: float) -> float:
    if column not in rows.columns or rows.empty:
        return float(default)
    value = pd.to_numeric(rows[column], errors="coerce").iloc[0]
    return float(value) if np.isfinite(value) else float(default)


def _weighted_or_mean(xyz: np.ndarray, weights: np.ndarray, count: int) -> np.ndarray:
    count = max(min(int(count), len(xyz)), 1)
    local_xyz = xyz[:count]
    local_weights = weights[:count]
    if float(local_weights.sum()) > 1.0e-12:
        local_weights = local_weights / float(local_weights.sum())
        return (local_xyz * local_weights[:, None]).sum(axis=0)
    return local_xyz.mean(axis=0)


def _finite_float(value: Any, default: float) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    return out if np.isfinite(out) else float(default)


def _mean(values: Any) -> float:
    series = (
        pd.to_numeric(values, errors="coerce")
        if values is not None
        else pd.Series(dtype=float)
    )
    return float(series.mean()) if len(series.dropna()) else float("nan")


def _median(values: Any) -> float:
    series = (
        pd.to_numeric(values, errors="coerce")
        if values is not None
        else pd.Series(dtype=float)
    )
    return float(series.median()) if len(series.dropna()) else float("nan")


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _empty_centers() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "Sequence",
            "candidate_time_s",
            "top1_x",
            "top1_y",
            "top1_z",
            "weighted5_x",
            "weighted5_y",
            "weighted5_z",
            "dispersion5",
            "topk_dispersion_m",
            "topk_dispersion_p95_m",
            "top_score",
            "top_score_margin",
            "nearest_cross_sensor_distance_m",
            "cross_sensor_neighbor_count",
            "frame_source_count",
            "topk_candidate_count",
        ]
    )


if __name__ == "__main__":
    raise SystemExit(main())
