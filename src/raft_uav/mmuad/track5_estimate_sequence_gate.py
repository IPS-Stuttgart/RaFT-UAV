"""Per-sequence blending for MMUAD Track 5 estimate trajectories.

The existing Track 5 sequence gate works on completed official submissions.
This helper works one step earlier on raw estimate CSVs: it resamples both
trajectories to the official template, applies a sequence -> alternate-weight
map, then writes upload-ready Track 5 artifacts.  This is useful for Codabench
experiments where one pose pipeline is better on some sequence families and a
second pipeline is better on others, while classification labels are supplied by
an independent class map.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

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
from raft_uav.mmuad.track5_template_resample import resample_estimates_to_track5_template

ESTIMATE_SEQUENCE_GATE_ESTIMATES_CSV = "mmuad_track5_estimate_sequence_gate_estimates.csv"
ESTIMATE_SEQUENCE_GATE_DIAGNOSTICS_CSV = "mmuad_track5_estimate_sequence_gate_diagnostics.csv"
ESTIMATE_SEQUENCE_GATE_WEIGHTS_CSV = "mmuad_track5_estimate_sequence_gate_weights.csv"
ESTIMATE_SEQUENCE_GATE_MANIFEST_JSON = "mmuad_track5_estimate_sequence_gate_manifest.json"
ESTIMATE_SEQUENCE_GATE_VALIDATION_JSON = "mmuad_track5_estimate_sequence_gate_validation.json"
ESTIMATE_SEQUENCE_GATE_VALIDATION_ROWS_CSV = "mmuad_track5_estimate_sequence_gate_validation_rows.csv"
OFFICIAL_RESULTS_CSV = "mmaud_results.csv"
OFFICIAL_ZIP = "ug2_submission.zip"
SEQUENCE_ALIASES = ("sequence_id", "Sequence", "sequence", "seq")
WEIGHT_ALIASES = ("weight", "blend_weight", "sequence_weight", "alternate_weight", "gate_weight")


def blend_track5_estimate_sequence_gate(
    *,
    base_estimates: pd.DataFrame,
    alternate_estimates: pd.DataFrame,
    template: pd.DataFrame,
    sequence_weights: pd.DataFrame,
    default_weight: float = 0.0,
    max_nearest_time_delta_s: float | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Blend two estimate trajectories with per-sequence alternate weights.

    Both inputs are interpolated to the same official template before blending.
    The returned tuple is ``(estimates, diagnostics, weight_table)``.  Weight 0
    keeps the base trajectory; weight 1 uses the alternate trajectory.
    """

    default = _validate_weight(default_weight, name="default_weight")
    weight_map = _sequence_weight_map(sequence_weights)
    template_rows = _normalize_template_rows(template)
    base, base_diag = resample_estimates_to_track5_template(
        base_estimates,
        template_rows,
        max_nearest_time_delta_s=max_nearest_time_delta_s,
    )
    alternate, alt_diag = resample_estimates_to_track5_template(
        alternate_estimates,
        template_rows,
        max_nearest_time_delta_s=max_nearest_time_delta_s,
    )
    base = base.sort_values(["sequence_id", "time_s"]).reset_index(drop=True)
    alternate = alternate.sort_values(["sequence_id", "time_s"]).reset_index(drop=True)
    if _template_keys(base) != _template_keys(alternate):
        raise ValueError("base and alternate estimates do not align after template resampling")

    weights = np.asarray(
        [weight_map.get(str(sequence_id), default) for sequence_id in base["sequence_id"]],
        dtype=float,
    )
    base_xyz = base[["state_x_m", "state_y_m", "state_z_m"]].to_numpy(float)
    alt_xyz = alternate[["state_x_m", "state_y_m", "state_z_m"]].to_numpy(float)
    valid_base = np.isfinite(base_xyz).all(axis=1)
    valid_alt = np.isfinite(alt_xyz).all(axis=1)
    blended = np.full_like(base_xyz, np.nan, dtype=float)
    both = valid_base & valid_alt
    blended[both] = (1.0 - weights[both, None]) * base_xyz[both] + weights[both, None] * alt_xyz[both]
    base_only = valid_base & ~valid_alt
    alt_only = ~valid_base & valid_alt
    blended[base_only] = base_xyz[base_only]
    blended[alt_only] = alt_xyz[alt_only]

    estimates = pd.DataFrame(
        {
            "sequence_id": base["sequence_id"].astype(str),
            "time_s": base["time_s"].astype(float),
            "source": "track5-estimate-sequence-gate",
            "track_id": "track5-estimate-sequence-gate",
            "state_x_m": blended[:, 0],
            "state_y_m": blended[:, 1],
            "state_z_m": blended[:, 2],
            "sequence_gate_weight": weights,
            "base_estimate_valid": valid_base.astype(bool),
            "alternate_estimate_valid": valid_alt.astype(bool),
        }
    )
    displacement = np.linalg.norm(alt_xyz - base_xyz, axis=1)
    diagnostics = pd.DataFrame(
        {
            "sequence_id": base["sequence_id"].astype(str),
            "time_s": base["time_s"].astype(float),
            "sequence_gate_weight": weights,
            "weight_source": [
                "sequence_weights" if str(sequence_id) in weight_map else "default"
                for sequence_id in base["sequence_id"]
            ],
            "base_x_m": base_xyz[:, 0],
            "base_y_m": base_xyz[:, 1],
            "base_z_m": base_xyz[:, 2],
            "alternate_x_m": alt_xyz[:, 0],
            "alternate_y_m": alt_xyz[:, 1],
            "alternate_z_m": alt_xyz[:, 2],
            "blended_x_m": blended[:, 0],
            "blended_y_m": blended[:, 1],
            "blended_z_m": blended[:, 2],
            "base_estimate_valid": valid_base.astype(bool),
            "alternate_estimate_valid": valid_alt.astype(bool),
            "base_to_alternate_displacement_m": displacement,
            "applied_displacement_m": weights * displacement,
        }
    )
    if "nearest_time_delta_s" in base_diag.columns:
        diagnostics["base_nearest_time_delta_s"] = pd.to_numeric(
            base_diag["nearest_time_delta_s"],
            errors="coerce",
        )
    if "nearest_time_delta_s" in alt_diag.columns:
        diagnostics["alternate_nearest_time_delta_s"] = pd.to_numeric(
            alt_diag["nearest_time_delta_s"],
            errors="coerce",
        )
    weights_df = pd.DataFrame(
        {
            "sequence_id": sorted({*base["sequence_id"].astype(str), *weight_map.keys()}),
        }
    )
    weights_df["sequence_gate_weight"] = weights_df["sequence_id"].map(
        lambda sequence: float(weight_map.get(sequence, default))
    )
    weights_df["weight_source"] = weights_df["sequence_id"].map(
        lambda sequence: "sequence_weights" if sequence in weight_map else "default"
    )
    return estimates, diagnostics, weights_df


def write_track5_estimate_sequence_gate_outputs(
    *,
    estimates: pd.DataFrame,
    diagnostics: pd.DataFrame,
    sequence_weights: pd.DataFrame,
    output_dir: Path,
    base_estimates_path: Path,
    alternate_estimates_path: Path,
    sequence_weights_path: Path,
    template: pd.DataFrame,
    class_map: dict[str, str] | None = None,
    default_classification: int | str = 0,
    require_leaderboard_ready: bool = False,
) -> dict[str, Path]:
    """Write blended estimates, official CSV/ZIP, validation, and manifest."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    paths = {
        "estimates_csv": output / ESTIMATE_SEQUENCE_GATE_ESTIMATES_CSV,
        "diagnostics_csv": output / ESTIMATE_SEQUENCE_GATE_DIAGNOSTICS_CSV,
        "weights_csv": output / ESTIMATE_SEQUENCE_GATE_WEIGHTS_CSV,
        "results_csv": output / OFFICIAL_RESULTS_CSV,
        "zip": output / OFFICIAL_ZIP,
        "validation_json": output / ESTIMATE_SEQUENCE_GATE_VALIDATION_JSON,
        "validation_rows_csv": output / ESTIMATE_SEQUENCE_GATE_VALIDATION_ROWS_CSV,
        "manifest_json": output / ESTIMATE_SEQUENCE_GATE_MANIFEST_JSON,
    }
    estimates.to_csv(paths["estimates_csv"], index=False)
    diagnostics.to_csv(paths["diagnostics_csv"], index=False)
    sequence_weights.to_csv(paths["weights_csv"], index=False)
    write_official_mmaud_results_csv(
        estimates,
        paths["results_csv"],
        classification=default_classification,
        class_map=class_map or {},
        invalid_row_policy="raise",
    )
    write_official_ug2_codabench_zip(
        estimates,
        paths["zip"],
        classification=default_classification,
        class_map=class_map or {},
        invalid_row_policy="raise",
    )
    validation = validate_official_track5_submission(paths["zip"], template=template, require_zip=True)
    paths["validation_json"].write_text(
        json.dumps(_jsonable(validation.summary), indent=2),
        encoding="utf-8",
    )
    validation.rows.to_csv(paths["validation_rows_csv"], index=False)
    if require_leaderboard_ready and not validation.summary.get("leaderboard_ready", False):
        reasons = ", ".join(validation.summary.get("leaderboard_blocking_reasons", [])) or "unknown"
        raise SystemExit(f"estimate sequence gate is not leaderboard-ready: {reasons}")
    manifest = {
        "schema": "raft-uav-mmuad-track5-estimate-sequence-gate-v1",
        "base_estimates": str(base_estimates_path),
        "alternate_estimates": str(alternate_estimates_path),
        "sequence_weights": str(sequence_weights_path),
        "row_count": int(len(estimates)),
        "sequence_count": int(estimates["sequence_id"].nunique()) if not estimates.empty else 0,
        "mean_sequence_gate_weight": _safe_mean(sequence_weights["sequence_gate_weight"]),
        "defaulted_sequence_count": int(sequence_weights["weight_source"].eq("default").sum()),
        "mean_applied_displacement_m": _safe_mean(diagnostics.get("applied_displacement_m", pd.Series(dtype=float))),
        "p95_applied_displacement_m": _safe_percentile(
            diagnostics.get("applied_displacement_m", pd.Series(dtype=float)),
            95,
        ),
        "validation": _jsonable(validation.summary),
        "paths": {name: str(path) for name, path in paths.items() if name != "manifest_json"},
    }
    paths["manifest_json"].write_text(json.dumps(_jsonable(manifest), indent=2), encoding="utf-8")
    return paths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="raft-uav-mmuad-track5-estimate-sequence-gate",
        description="blend two Track 5 estimate trajectories with per-sequence weights",
    )
    parser.add_argument("--base-estimates", type=Path, required=True)
    parser.add_argument("--alternate-estimates", type=Path, required=True)
    parser.add_argument("--sequence-weights", type=Path, required=True)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--default-weight", type=float, default=0.0)
    parser.add_argument("--class-map", type=Path)
    parser.add_argument("--default-classification", default="0")
    parser.add_argument("--max-nearest-time-delta-s", type=float)
    parser.add_argument("--require-leaderboard-ready", action="store_true")
    args = parser.parse_args(argv)

    template = load_official_track5_template_file(args.template)
    estimates, diagnostics, weights = blend_track5_estimate_sequence_gate(
        base_estimates=pd.read_csv(args.base_estimates),
        alternate_estimates=pd.read_csv(args.alternate_estimates),
        template=template,
        sequence_weights=pd.read_csv(args.sequence_weights),
        default_weight=float(args.default_weight),
        max_nearest_time_delta_s=args.max_nearest_time_delta_s,
    )
    class_map = load_sequence_class_map(args.class_map) if args.class_map is not None else {}
    paths = write_track5_estimate_sequence_gate_outputs(
        estimates=estimates,
        diagnostics=diagnostics,
        sequence_weights=weights,
        output_dir=args.output_dir,
        base_estimates_path=args.base_estimates,
        alternate_estimates_path=args.alternate_estimates,
        sequence_weights_path=args.sequence_weights,
        template=template,
        class_map=class_map,
        default_classification=args.default_classification,
        require_leaderboard_ready=args.require_leaderboard_ready,
    )
    validation = json.loads(paths["validation_json"].read_text(encoding="utf-8"))
    print("mmuad_track5_estimate_sequence_gate=ok")
    for name, path in paths.items():
        print(f"{name}={path}")
    print(f"leaderboard_ready={validation.get('leaderboard_ready')}")
    print(f"codabench_upload_ready={validation.get('codabench_upload_ready')}")
    return 0


def _sequence_weight_map(rows: pd.DataFrame) -> dict[str, float]:
    frame = pd.DataFrame(rows).copy()
    sequence_column = _first_present(frame, SEQUENCE_ALIASES)
    weight_column = _first_present(frame, WEIGHT_ALIASES)
    if sequence_column is None or weight_column is None:
        raise ValueError("sequence weight table must contain sequence_id and weight columns")
    result: dict[str, float] = {}
    for _, row in frame.iterrows():
        sequence_id = _official_sequence_id_or_none(row[sequence_column])
        if sequence_id is None:
            continue
        result[sequence_id] = _validate_weight(row[weight_column], name="sequence_weight")
    return result


def _official_sequence_id_or_none(value: Any) -> str | None:
    try:
        return parse_official_sequence_cell(value)
    except ValueError:
        return None


def _validate_weight(value: object, *, name: str) -> float:
    weight = float(value)
    if not np.isfinite(weight) or weight < 0.0 or weight > 1.0:
        raise ValueError(f"{name} must be finite and in [0, 1]: {value}")
    return weight


def _normalize_template_rows(template: pd.DataFrame) -> pd.DataFrame:
    frame = pd.DataFrame(template).copy()
    sequence_column = _first_present(frame, ("sequence_id", "Sequence", "sequence", "seq"))
    time_column = _first_present(frame, ("time_s", "Timestamp", "timestamp", "timestamp_s", "time"))
    if sequence_column is None or time_column is None:
        raise ValueError("template must contain sequence and timestamp columns")
    rows = pd.DataFrame(
        {
            "sequence_id": frame[sequence_column].map(_official_sequence_id_or_none),
            "time_s": pd.to_numeric(frame[time_column], errors="coerce"),
        }
    )
    finite = rows["sequence_id"].notna() & np.isfinite(rows["time_s"].to_numpy(float))
    return rows.loc[finite].sort_values(["sequence_id", "time_s"]).reset_index(drop=True)


def _template_keys(rows: pd.DataFrame) -> list[tuple[str, float]]:
    return [
        (str(sequence_id), round(float(time_s), 9))
        for sequence_id, time_s in zip(rows["sequence_id"], rows["time_s"], strict=True)
    ]


def _first_present(rows: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    normalized: dict[str, str] = {}
    for column in rows.columns:
        normalized.setdefault(str(column).strip().lower(), str(column))
    for candidate in candidates:
        if candidate in rows.columns:
            return candidate
        found = normalized.get(str(candidate).strip().lower())
        if found is not None:
            return found
    return None


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
