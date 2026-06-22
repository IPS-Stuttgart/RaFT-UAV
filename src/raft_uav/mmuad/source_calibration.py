"""Train-learned source-coordinate calibration for MMUAD candidates."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from raft_uav.mmuad.evaluator import load_evaluation_truth_file
from raft_uav.mmuad.schema import CandidateFrame, normalize_candidate_columns


SOURCE_CALIBRATION_SCHEMA = "raft-uav-mmuad-source-calibration-v1"
SOURCE_CALIBRATION_MODES = ("identity", "source-translation", "source-rigid", "source-affine")


@dataclass(frozen=True)
class SourceTransform:
    """Affine source transform applied as ``xyz @ linear.T + translation``."""

    linear: np.ndarray
    translation_m: np.ndarray
    metadata: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        linear = np.asarray(self.linear, dtype=float)
        translation = np.asarray(self.translation_m, dtype=float).reshape(3)
        if linear.shape != (3, 3):
            raise ValueError(f"linear transform must be 3x3, got {linear.shape}")
        object.__setattr__(self, "linear", linear)
        object.__setattr__(self, "translation_m", translation)
        object.__setattr__(self, "metadata", dict(self.metadata or {}))

    @classmethod
    def identity(cls) -> "SourceTransform":
        return cls(np.eye(3), np.zeros(3))

    def apply(self, xyz: np.ndarray) -> np.ndarray:
        points = np.asarray(xyz, dtype=float)
        if points.ndim == 1:
            return self.linear @ points.reshape(3) + self.translation_m
        return points @ self.linear.T + self.translation_m

    def to_jsonable(self) -> dict[str, Any]:
        payload = {
            "linear": self.linear.tolist(),
            "translation_m": self.translation_m.tolist(),
            "translation_norm_m": float(np.linalg.norm(self.translation_m)),
        }
        payload.update(_jsonable(self.metadata or {}))
        return payload

    @classmethod
    def from_jsonable(cls, payload: dict[str, Any]) -> "SourceTransform":
        metadata = {
            str(key): value
            for key, value in payload.items()
            if key not in {"linear", "translation_m", "translation_norm_m"}
        }
        return cls(
            np.asarray(payload.get("linear", np.eye(3)), dtype=float),
            np.asarray(payload.get("translation_m", [0.0, 0.0, 0.0]), dtype=float),
            metadata=metadata,
        )


def fit_source_calibration(
    candidates: CandidateFrame | pd.DataFrame,
    truth: pd.DataFrame,
    *,
    mode: str,
    max_truth_time_delta_s: float = 0.5,
    max_pair_distance_m: float = 120.0,
    min_pairs_per_source: int = 20,
    source_translation_alpha_grid: tuple[float, ...] | list[float] | None = None,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    """Fit source transforms from train candidates and train truth only."""

    mode = _normalize_source_calibration_mode(mode)
    alpha_grid = _normalize_alpha_grid(source_translation_alpha_grid)
    candidate_rows = candidates.rows if isinstance(candidates, CandidateFrame) else candidates
    pairs = build_source_calibration_pairs(
        CandidateFrame(normalize_candidate_columns(candidate_rows)),
        truth,
        max_truth_time_delta_s=max_truth_time_delta_s,
        max_pair_distance_m=max_pair_distance_m,
    )
    transforms, fit_summary = fit_source_transforms(
        pairs,
        mode=mode,
        min_pairs_per_source=min_pairs_per_source,
        source_translation_alpha_grid=alpha_grid,
    )
    payload = source_calibration_payload(
        transforms,
        mode=mode,
        max_truth_time_delta_s=max_truth_time_delta_s,
        max_pair_distance_m=max_pair_distance_m,
        min_pairs_per_source=min_pairs_per_source,
        source_translation_alpha_grid=alpha_grid,
        fit_pair_count=len(pairs),
        fit_summary=fit_summary,
    )
    return payload, pairs, fit_summary


def build_source_calibration_pairs(
    candidates: CandidateFrame,
    truth: pd.DataFrame,
    *,
    max_truth_time_delta_s: float,
    max_pair_distance_m: float,
) -> pd.DataFrame:
    """Pair candidates with nearest train truth rows for transform fitting."""

    rows = candidates.rows.copy()
    if rows.empty:
        return _empty_pairs()
    for column in ("time_s", "x_m", "y_m", "z_m"):
        rows[column] = pd.to_numeric(rows[column], errors="coerce")
    rows = rows.loc[np.isfinite(rows[["time_s", "x_m", "y_m", "z_m"]]).all(axis=1)].copy()
    truth_rows = _normalize_truth_rows(truth)
    parts: list[pd.DataFrame] = []
    for sequence_id, seq_rows in rows.groupby(rows["sequence_id"].astype(str), sort=False):
        seq_truth = truth_rows.loc[truth_rows["sequence_id"].astype(str) == str(sequence_id)].copy()
        if seq_truth.empty:
            continue
        seq_rows = seq_rows.sort_values("time_s")
        seq_truth = seq_truth.sort_values("time_s")
        paired = pd.merge_asof(
            seq_rows,
            seq_truth[["time_s", "x_m", "y_m", "z_m"]].rename(
                columns={"x_m": "truth_x_m", "y_m": "truth_y_m", "z_m": "truth_z_m"}
            ),
            on="time_s",
            direction="nearest",
            tolerance=float(max_truth_time_delta_s),
        )
        parts.append(paired)
    if not parts:
        return _empty_pairs()
    pairs = pd.concat(parts, ignore_index=True)
    finite_truth = np.isfinite(pairs[["truth_x_m", "truth_y_m", "truth_z_m"]]).all(axis=1)
    pairs = pairs.loc[finite_truth].copy()
    if pairs.empty:
        return _empty_pairs()
    before = np.linalg.norm(
        pairs[["x_m", "y_m", "z_m"]].to_numpy(float)
        - pairs[["truth_x_m", "truth_y_m", "truth_z_m"]].to_numpy(float),
        axis=1,
    )
    pairs["pair_error_before_m"] = before
    pairs = pairs.loc[pairs["pair_error_before_m"] <= float(max_pair_distance_m)].copy()
    if pairs.empty:
        return _empty_pairs()
    return (
        pairs.sort_values("pair_error_before_m")
        .drop_duplicates(["sequence_id", "source", "time_s"], keep="first")
        .sort_values(["sequence_id", "source", "time_s"])
        .reset_index(drop=True)
    )


def fit_source_transforms(
    pairs: pd.DataFrame,
    *,
    mode: str,
    min_pairs_per_source: int,
    source_translation_alpha_grid: tuple[float, ...] | list[float] | None = None,
) -> tuple[dict[str, SourceTransform], pd.DataFrame]:
    """Fit one transform per source from paired train rows."""

    mode = _normalize_source_calibration_mode(mode)
    alpha_grid = _normalize_alpha_grid(source_translation_alpha_grid)
    transforms: dict[str, SourceTransform] = {}
    records: list[dict[str, Any]] = []
    if pairs.empty:
        return transforms, pd.DataFrame()
    for source, group in pairs.groupby("source", sort=True):
        source_text = str(source)
        x = group[["x_m", "y_m", "z_m"]].to_numpy(float)
        y = group[["truth_x_m", "truth_y_m", "truth_z_m"]].to_numpy(float)
        if mode == "identity" or len(group) < int(min_pairs_per_source):
            transform = SourceTransform.identity()
            fit_status = "identity" if mode == "identity" else "insufficient_pairs"
        elif mode == "source-translation":
            alpha_result = _fit_source_translation_alpha_cv(group, alpha_grid)
            transform = _fit_translation(
                x,
                y,
                alpha=alpha_result["alpha"],
                alpha_metadata=alpha_result,
            )
            fit_status = "fit"
        elif mode == "source-rigid":
            transform = _fit_rigid(x, y)
            fit_status = "fit"
        elif mode == "source-affine":
            transform = _fit_affine(x, y)
            fit_status = "fit"
        else:  # pragma: no cover - normalized above
            raise ValueError(f"unsupported mode: {mode}")
        transforms[source_text] = transform
        after = np.linalg.norm(transform.apply(x) - y, axis=1)
        before = np.linalg.norm(x - y, axis=1)
        records.append(
            {
                "mode": mode,
                "source": source_text,
                "fit_status": fit_status,
                "pair_count": int(len(group)),
                "before_mean_m": _mean(before),
                "before_p95_m": _percentile(before, 95.0),
                "after_mean_m": _mean(after),
                "after_p95_m": _percentile(after, 95.0),
                "translation_x_m": float(transform.translation_m[0]),
                "translation_y_m": float(transform.translation_m[1]),
                "translation_z_m": float(transform.translation_m[2]),
                "translation_norm_m": float(np.linalg.norm(transform.translation_m)),
                "source_translation_alpha": _metadata_float(
                    transform,
                    "source_translation_alpha",
                ),
                "source_translation_alpha_cv_mse": _metadata_float(
                    transform,
                    "source_translation_alpha_cv_mse",
                ),
                "source_translation_alpha_cv_fold_count": int(
                    transform.metadata.get("source_translation_alpha_cv_fold_count", 0)
                    if transform.metadata
                    else 0
                ),
                "source_translation_base_x_m": _metadata_float(
                    transform,
                    "source_translation_base_x_m",
                ),
                "source_translation_base_y_m": _metadata_float(
                    transform,
                    "source_translation_base_y_m",
                ),
                "source_translation_base_z_m": _metadata_float(
                    transform,
                    "source_translation_base_z_m",
                ),
                "source_translation_base_norm_m": _metadata_float(
                    transform,
                    "source_translation_base_norm_m",
                ),
                "linear_det": float(np.linalg.det(transform.linear)),
            }
        )
    return transforms, pd.DataFrame.from_records(records)


def source_calibration_payload(
    transforms: dict[str, SourceTransform],
    *,
    mode: str,
    max_truth_time_delta_s: float,
    max_pair_distance_m: float,
    min_pairs_per_source: int,
    source_translation_alpha_grid: tuple[float, ...] | list[float] | None = None,
    fit_pair_count: int,
    fit_summary: pd.DataFrame | None = None,
    provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a portable source-calibration JSON payload."""

    return {
        "schema": SOURCE_CALIBRATION_SCHEMA,
        "mode": _normalize_source_calibration_mode(mode),
        "protocol": "fit_on_train_only_apply_same_transform_to_val_or_test",
        "max_truth_time_delta_s": float(max_truth_time_delta_s),
        "max_pair_distance_m": float(max_pair_distance_m),
        "min_pairs_per_source": int(min_pairs_per_source),
        "source_translation_alpha_grid": list(_normalize_alpha_grid(source_translation_alpha_grid)),
        "source_translation_alpha_protocol": (
            "per_source_leave_one_sequence_out_train_only"
        ),
        "fit_pair_count": int(fit_pair_count),
        "transforms": {
            str(source): transform.to_jsonable()
            for source, transform in sorted(transforms.items())
        },
        "fit_summary": [] if fit_summary is None else fit_summary.to_dict(orient="records"),
        "provenance": provenance or {},
    }


def load_source_calibration_json(path: Path) -> dict[str, Any]:
    """Load and validate source-calibration JSON."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema") != SOURCE_CALIBRATION_SCHEMA:
        raise ValueError(f"{path} is not an MMUAD source-calibration JSON")
    _normalize_source_calibration_mode(str(payload.get("mode", "identity")))
    if not isinstance(payload.get("transforms", {}), dict):
        raise ValueError("source-calibration JSON must contain a transforms object")
    return payload


def write_source_calibration_json(payload: dict[str, Any], path: Path) -> Path:
    """Write source-calibration JSON."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2), encoding="utf-8")
    return path


def apply_source_calibration_json(
    frame: CandidateFrame | pd.DataFrame,
    path: Path,
    *,
    mode: str | None = None,
) -> CandidateFrame:
    """Apply source-calibration JSON to candidate rows."""

    payload = load_source_calibration_json(path)
    return apply_source_calibration_payload(frame, payload, mode=mode)


def apply_source_calibration_payload(
    frame: CandidateFrame | pd.DataFrame,
    payload: dict[str, Any],
    *,
    mode: str | None = None,
) -> CandidateFrame:
    """Apply a loaded source-calibration payload to candidate rows."""

    selected_mode = _normalize_source_calibration_mode(mode or str(payload.get("mode", "identity")))
    rows = frame.rows.copy() if isinstance(frame, CandidateFrame) else pd.DataFrame(frame).copy()
    rows = normalize_candidate_columns(rows)
    if rows.empty or selected_mode == "identity":
        out = rows.copy()
        if not out.empty:
            out["mmuad_source_calibration_mode"] = selected_mode
            out["mmuad_source_calibration_applied"] = False
        return CandidateFrame(normalize_candidate_columns(out))
    payload_mode = _normalize_source_calibration_mode(str(payload.get("mode", selected_mode)))
    if payload_mode != selected_mode:
        raise ValueError(
            f"source-calibration JSON mode {payload_mode!r} does not match requested mode "
            f"{selected_mode!r}"
        )
    transforms = {
        str(source): SourceTransform.from_jsonable(transform)
        for source, transform in payload.get("transforms", {}).items()
        if isinstance(transform, dict)
    }
    parts: list[pd.DataFrame] = []
    for source, group in rows.groupby("source", sort=False):
        group = group.copy()
        transform = _match_source_transform(str(source), transforms)
        if transform is None:
            group["mmuad_source_calibration_applied"] = False
            group["mmuad_source_calibration_source"] = ""
        else:
            xyz = group[["x_m", "y_m", "z_m"]].to_numpy(float)
            out_xyz = transform.apply(xyz)
            group["x_m"] = out_xyz[:, 0]
            group["y_m"] = out_xyz[:, 1]
            group["z_m"] = out_xyz[:, 2]
            group["mmuad_source_calibration_applied"] = True
            group["mmuad_source_calibration_source"] = str(source)
            group["mmuad_source_calibration_alpha"] = transform.metadata.get(
                "source_translation_alpha",
                np.nan,
            )
        group["mmuad_source_calibration_mode"] = selected_mode
        parts.append(group)
    return CandidateFrame(normalize_candidate_columns(pd.concat(parts, ignore_index=True)))


def fit_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="raft-uav-mmuad-fit-source-calibration",
        description="fit MMUAD source-coordinate calibration from train candidates and train truth",
    )
    parser.add_argument("--train-candidates", type=Path, required=True)
    parser.add_argument("--train-truth", type=Path, required=True)
    parser.add_argument("--mode", choices=SOURCE_CALIBRATION_MODES, default="source-translation")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--fit-pairs-csv", type=Path)
    parser.add_argument("--fit-summary-csv", type=Path)
    parser.add_argument("--max-truth-time-delta-s", type=float, default=0.5)
    parser.add_argument("--max-pair-distance-m", type=float, default=120.0)
    parser.add_argument("--min-pairs-per-source", type=int, default=20)
    parser.add_argument(
        "--source-translation-alpha-grid",
        default="1.0",
        help=(
            "comma-separated train-CV alpha grid for source-translation shrinkage, "
            "for example 0,0.25,0.5,0.75,1"
        ),
    )
    args = parser.parse_args(argv)

    candidates = CandidateFrame(normalize_candidate_columns(pd.read_csv(args.train_candidates)))
    truth = load_evaluation_truth_file(args.train_truth).rows
    payload, pairs, fit_summary = fit_source_calibration(
        candidates,
        truth,
        mode=args.mode,
        max_truth_time_delta_s=args.max_truth_time_delta_s,
        max_pair_distance_m=args.max_pair_distance_m,
        min_pairs_per_source=args.min_pairs_per_source,
        source_translation_alpha_grid=_parse_alpha_grid(args.source_translation_alpha_grid),
    )
    payload["provenance"] = {
        "train_candidates": str(args.train_candidates),
        "train_truth": str(args.train_truth),
    }
    write_source_calibration_json(payload, args.output_json)
    if args.fit_pairs_csv is not None:
        args.fit_pairs_csv.parent.mkdir(parents=True, exist_ok=True)
        pairs.to_csv(args.fit_pairs_csv, index=False)
    if args.fit_summary_csv is not None:
        args.fit_summary_csv.parent.mkdir(parents=True, exist_ok=True)
        fit_summary.to_csv(args.fit_summary_csv, index=False)
    print("mmuad_fit_source_calibration=ok")
    print(f"output_json={args.output_json}")
    print(f"mode={args.mode}")
    print(f"fit_pair_count={len(pairs)}")
    print(f"transformed_source_count={len(payload.get('transforms', {}))}")
    print(f"source_translation_alpha_grid={payload.get('source_translation_alpha_grid')}")
    return 0


def apply_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="raft-uav-mmuad-apply-source-calibration",
        description="apply train-learned MMUAD source calibration to candidate rows",
    )
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--output-candidates", type=Path, required=True)
    parser.add_argument("--mmuad-source-calibration-json", type=Path, required=True)
    parser.add_argument("--mmuad-source-calibration-mode", choices=SOURCE_CALIBRATION_MODES)
    args = parser.parse_args(argv)

    frame = CandidateFrame(normalize_candidate_columns(pd.read_csv(args.candidates)))
    calibrated = apply_source_calibration_json(
        frame,
        args.mmuad_source_calibration_json,
        mode=args.mmuad_source_calibration_mode,
    )
    args.output_candidates.parent.mkdir(parents=True, exist_ok=True)
    calibrated.rows.to_csv(args.output_candidates, index=False)
    print("mmuad_apply_source_calibration=ok")
    print(f"output_candidates={args.output_candidates}")
    print(f"rows={len(calibrated.rows)}")
    return 0


def _fit_translation(
    x: np.ndarray,
    y: np.ndarray,
    *,
    alpha: float = 1.0,
    alpha_metadata: dict[str, Any] | None = None,
) -> SourceTransform:
    base_translation = np.nanmedian(y - x, axis=0)
    alpha = float(alpha)
    metadata = dict(alpha_metadata or {})
    metadata.update(
        {
            "source_translation_alpha": alpha,
            "source_translation_base_translation_m": base_translation.tolist(),
            "source_translation_base_x_m": float(base_translation[0]),
            "source_translation_base_y_m": float(base_translation[1]),
            "source_translation_base_z_m": float(base_translation[2]),
            "source_translation_base_norm_m": float(np.linalg.norm(base_translation)),
        }
    )
    return SourceTransform(np.eye(3), alpha * base_translation, metadata=metadata)


def _fit_source_translation_alpha_cv(
    group: pd.DataFrame,
    alpha_grid: tuple[float, ...],
) -> dict[str, Any]:
    """Pick a source-translation shrinkage alpha using train-only sequence CV."""

    alpha_grid = _normalize_alpha_grid(alpha_grid)
    x_all = group[["x_m", "y_m", "z_m"]].to_numpy(float)
    y_all = group[["truth_x_m", "truth_y_m", "truth_z_m"]].to_numpy(float)
    if len(alpha_grid) == 1:
        return {
            "source_translation_alpha": float(alpha_grid[0]),
            "alpha": float(alpha_grid[0]),
            "source_translation_alpha_cv_mse": float("nan"),
            "source_translation_alpha_cv_fold_count": 0,
            "source_translation_alpha_cv_protocol": "fixed_alpha_no_cv",
            "source_translation_alpha_grid": list(alpha_grid),
        }

    sequence_ids = sorted({str(value) for value in group["sequence_id"].astype(str)})
    cv_records: list[dict[str, float | str | int]] = []
    if len(sequence_ids) >= 2:
        for heldout_sequence in sequence_ids:
            train = group.loc[group["sequence_id"].astype(str) != heldout_sequence]
            valid = group.loc[group["sequence_id"].astype(str) == heldout_sequence]
            if train.empty or valid.empty:
                continue
            train_x = train[["x_m", "y_m", "z_m"]].to_numpy(float)
            train_y = train[["truth_x_m", "truth_y_m", "truth_z_m"]].to_numpy(float)
            valid_x = valid[["x_m", "y_m", "z_m"]].to_numpy(float)
            valid_y = valid[["truth_x_m", "truth_y_m", "truth_z_m"]].to_numpy(float)
            base_translation = np.nanmedian(train_y - train_x, axis=0)
            for alpha in alpha_grid:
                residual = valid_x + float(alpha) * base_translation - valid_y
                cv_records.append(
                    {
                        "heldout_sequence": heldout_sequence,
                        "alpha": float(alpha),
                        "mse": float(np.nanmean(np.sum(residual**2, axis=1))),
                        "count": int(len(valid)),
                    }
                )
        protocol = "leave_one_sequence_out"
    else:
        base_translation = np.nanmedian(y_all - x_all, axis=0)
        for alpha in alpha_grid:
            residual = x_all + float(alpha) * base_translation - y_all
            cv_records.append(
                {
                    "heldout_sequence": "__insample__",
                    "alpha": float(alpha),
                    "mse": float(np.nanmean(np.sum(residual**2, axis=1))),
                    "count": int(len(group)),
                }
            )
        protocol = "insample_single_sequence_fallback"

    if not cv_records:
        selected_alpha = 1.0 if 1.0 in alpha_grid else float(alpha_grid[-1])
        selected_mse = float("nan")
        fold_count = 0
    else:
        cv = pd.DataFrame.from_records(cv_records)
        summary_records = []
        for alpha, rows in cv.groupby("alpha", sort=True):
            summary_records.append(
                {
                    "alpha": float(alpha),
                    "weighted_mse": float(np.average(rows["mse"], weights=rows["count"])),
                }
            )
        summary = pd.DataFrame.from_records(summary_records)
        summary = summary.sort_values(["weighted_mse", "alpha"]).reset_index(drop=True)
        selected_alpha = float(summary.loc[0, "alpha"])
        selected_mse = float(summary.loc[0, "weighted_mse"])
        fold_count = int(cv["heldout_sequence"].nunique())

    return {
        "source_translation_alpha": selected_alpha,
        "alpha": selected_alpha,
        "source_translation_alpha_cv_mse": selected_mse,
        "source_translation_alpha_cv_fold_count": fold_count,
        "source_translation_alpha_cv_protocol": protocol,
        "source_translation_alpha_grid": list(alpha_grid),
    }


def _fit_rigid(x: np.ndarray, y: np.ndarray) -> SourceTransform:
    x_centroid = np.nanmean(x, axis=0)
    y_centroid = np.nanmean(y, axis=0)
    x_centered = x - x_centroid
    y_centered = y - y_centroid
    covariance = x_centered.T @ y_centered
    u, _s, vt = np.linalg.svd(covariance)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[-1, :] *= -1.0
        rotation = vt.T @ u.T
    translation = y_centroid - rotation @ x_centroid
    return SourceTransform(rotation, translation)


def _fit_affine(x: np.ndarray, y: np.ndarray) -> SourceTransform:
    design = np.column_stack([x, np.ones(len(x), dtype=float)])
    params, *_ = np.linalg.lstsq(design, y, rcond=None)
    return SourceTransform(params[:3, :].T, params[3, :])


def _match_source_transform(
    source: str,
    transforms: dict[str, SourceTransform],
) -> SourceTransform | None:
    source_l = str(source).lower()
    for key, transform in transforms.items():
        if source_l == str(key).lower():
            return transform
    matches = [
        (len(str(key)), transform)
        for key, transform in transforms.items()
        if source_l.startswith(str(key).lower()) or str(key).lower().startswith(source_l)
    ]
    if not matches:
        return None
    return max(matches, key=lambda item: item[0])[1]


def _normalize_source_calibration_mode(mode: str) -> str:
    normalized = str(mode).strip().lower().replace("_", "-")
    aliases = {
        "none": "identity",
        "translation": "source-translation",
        "rigid": "source-rigid",
        "affine": "source-affine",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in SOURCE_CALIBRATION_MODES:
        allowed = ", ".join(SOURCE_CALIBRATION_MODES)
        raise ValueError(f"unsupported MMUAD source-calibration mode {mode!r}; allowed={allowed}")
    return normalized


def _parse_alpha_grid(value: str | float | int | None) -> tuple[float, ...]:
    if value is None:
        return (1.0,)
    if isinstance(value, (int, float)):
        return _normalize_alpha_grid([float(value)])
    parts = [
        part.strip()
        for part in str(value).replace(";", ",").split(",")
        if part.strip()
    ]
    if not parts:
        return (1.0,)
    return _normalize_alpha_grid([float(part) for part in parts])


def parse_source_translation_alpha_grid(value: str | float | int | None) -> tuple[float, ...]:
    """Parse a user-facing source-translation alpha grid specification."""

    return _parse_alpha_grid(value)


def _normalize_alpha_grid(values: tuple[float, ...] | list[float] | None) -> tuple[float, ...]:
    if values is None:
        return (1.0,)
    finite_values: list[float] = []
    for value in values:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(numeric):
            finite_values.append(float(np.clip(numeric, 0.0, 1.0)))
    out = sorted(set(finite_values))
    if not out:
        return (1.0,)
    return tuple(out)


def _metadata_float(transform: SourceTransform, key: str) -> float:
    if not transform.metadata or key not in transform.metadata:
        return float("nan")
    try:
        return float(transform.metadata[key])
    except (TypeError, ValueError):
        return float("nan")


def _normalize_truth_rows(truth: pd.DataFrame) -> pd.DataFrame:
    rows = truth.copy()
    if "sequence_id" not in rows.columns and "Sequence" in rows.columns:
        rows["sequence_id"] = rows["Sequence"]
    if "time_s" not in rows.columns and "Timestamp" in rows.columns:
        rows["time_s"] = rows["Timestamp"]
    if not {"x_m", "y_m", "z_m"}.issubset(rows.columns) and "Position" in rows.columns:
        positions = rows["Position"].map(_parse_position)
        rows["x_m"] = [value[0] for value in positions]
        rows["y_m"] = [value[1] for value in positions]
        rows["z_m"] = [value[2] for value in positions]
    required = {"sequence_id", "time_s", "x_m", "y_m", "z_m"}
    missing = required.difference(rows.columns)
    if missing:
        raise ValueError(f"truth rows missing required columns: {sorted(missing)}")
    rows = rows[list(required)].copy()
    rows["sequence_id"] = rows["sequence_id"].astype(str)
    for column in ("time_s", "x_m", "y_m", "z_m"):
        rows[column] = pd.to_numeric(rows[column], errors="coerce")
    return rows.loc[np.isfinite(rows[["time_s", "x_m", "y_m", "z_m"]]).all(axis=1)].copy()


def _parse_position(value: Any) -> tuple[float, float, float]:
    if isinstance(value, str):
        stripped = value.strip().strip("[]()")
        parts = [part.strip() for part in stripped.replace(";", ",").split(",") if part.strip()]
        if len(parts) >= 3:
            return float(parts[0]), float(parts[1]), float(parts[2])
    if isinstance(value, (list, tuple, np.ndarray)) and len(value) >= 3:
        return float(value[0]), float(value[1]), float(value[2])
    return float("nan"), float("nan"), float("nan")


def _empty_pairs() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "sequence_id",
            "time_s",
            "source",
            "track_id",
            "x_m",
            "y_m",
            "z_m",
            "truth_x_m",
            "truth_y_m",
            "truth_z_m",
            "pair_error_before_m",
        ]
    )


def _mean(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return float(np.mean(values)) if values.size else float("nan")


def _percentile(values: np.ndarray, q: float) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return float(np.percentile(values, q)) if values.size else float("nan")


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


def main(argv: list[str] | None = None) -> int:
    return fit_main(argv)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(fit_main())
