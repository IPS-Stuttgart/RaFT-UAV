"""Blend two official MMUAD/UG2+ Track 5 submissions with sequence weights.

This helper is meant for inference-safe sequence-level gating experiments.  It
uses only two official-style submissions plus a sequence -> blend weight table:

``output = (1 - weight) * base + weight * alternate``

Classification labels are copied from the base submission by default, so the
tool can improve or probe pose without accidentally changing a separately
selected UAV-type classifier.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from raft_uav.mmuad.submission import (
    parse_official_sequence_cell,
    validate_official_track5_submission,
    write_official_mmaud_results_csv,
    write_official_ug2_codabench_zip,
)
from raft_uav.mmuad.track5_submission_ensemble import _jsonable
from raft_uav.mmuad.track5_submission_ensemble import _submission_keys
from raft_uav.mmuad.track5_submission_ensemble import load_track5_submission

SEQUENCE_GATE_ESTIMATES_CSV = "mmuad_track5_sequence_gate_estimates.csv"
SEQUENCE_GATE_RESULTS_CSV = "mmaud_results_sequence_gate.csv"
SEQUENCE_GATE_ZIP = "ug2_submission_sequence_gate.zip"
SEQUENCE_GATE_DIAGNOSTICS_CSV = "mmuad_track5_sequence_gate_diagnostics.csv"
SEQUENCE_GATE_MANIFEST_JSON = "mmuad_track5_sequence_gate_manifest.json"
SEQUENCE_GATE_VALIDATION_JSON = "mmuad_track5_sequence_gate_validation.json"
SEQUENCE_GATE_VALIDATION_ROWS_CSV = "mmuad_track5_sequence_gate_validation_rows.csv"

SEQUENCE_ALIASES = ("Sequence", "sequence_id", "sequence", "heldout_sequence", "seq")
WEIGHT_ALIASES = ("weight", "blend_weight", "sequence_weight", "alternate_weight", "gate_weight")
ClassPolicy = Literal["base", "alternate"]


@dataclass(frozen=True)
class SequenceGateResult:
    """Sequence-gated estimates and row diagnostics."""

    estimates: pd.DataFrame
    diagnostics: pd.DataFrame
    sequence_weights: pd.DataFrame


def blend_track5_sequence_gate(
    *,
    base_submission: pd.DataFrame,
    alternate_submission: pd.DataFrame,
    sequence_weights: pd.DataFrame,
    default_weight: float = 0.0,
    class_policy: ClassPolicy = "base",
) -> SequenceGateResult:
    """Return per-sequence weighted Track 5 estimates.

    ``base_submission`` and ``alternate_submission`` must already be normalized
    frames from :func:`load_track5_submission`.  The two submissions must share
    the same ``sequence_id,time_s`` template.
    """

    if class_policy not in {"base", "alternate"}:
        raise ValueError("class_policy must be 'base' or 'alternate'")
    base = pd.DataFrame(base_submission).copy().sort_values(["sequence_id", "time_s"])
    alternate = pd.DataFrame(alternate_submission).copy().sort_values(["sequence_id", "time_s"])
    if _submission_keys(base) != _submission_keys(alternate):
        raise ValueError("base and alternate submissions do not match sequence/timestamp keys")

    weight_map = _sequence_weight_map(sequence_weights)
    default = _validate_weight(default_weight, name="default_weight")

    base = base.reset_index(drop=True)
    alternate = alternate.reset_index(drop=True)
    sequence_keys = [_sequence_weight_key(sequence_id) for sequence_id in base["sequence_id"]]
    weights = np.asarray(
        [weight_map.get(sequence_id, default) for sequence_id in sequence_keys],
        dtype=float,
    )
    base_xyz = base[["state_x_m", "state_y_m", "state_z_m"]].to_numpy(float)
    alt_xyz = alternate[["state_x_m", "state_y_m", "state_z_m"]].to_numpy(float)
    blended_xyz = (1.0 - weights[:, None]) * base_xyz + weights[:, None] * alt_xyz
    classes = (
        alternate["Classification"].to_numpy(int)
        if class_policy == "alternate"
        else base["Classification"].to_numpy(int)
    )
    estimates = pd.DataFrame(
        {
            "sequence_id": base["sequence_id"].astype(str),
            "time_s": base["time_s"].astype(float),
            "source": "track5-sequence-gate",
            "track_id": "track5-sequence-gate",
            "state_x_m": blended_xyz[:, 0].astype(float),
            "state_y_m": blended_xyz[:, 1].astype(float),
            "state_z_m": blended_xyz[:, 2].astype(float),
            "Classification": classes.astype(int),
            "sequence_gate_weight": weights.astype(float),
        }
    )
    displacement = np.linalg.norm(alt_xyz - base_xyz, axis=1)
    diagnostics = pd.DataFrame(
        {
            "sequence_id": base["sequence_id"].astype(str),
            "time_s": base["time_s"].astype(float),
            "sequence_gate_weight": weights.astype(float),
            "weight_source": [
                "sequence_weights" if sequence_id in weight_map else "default"
                for sequence_id in sequence_keys
            ],
            "base_x_m": base_xyz[:, 0],
            "base_y_m": base_xyz[:, 1],
            "base_z_m": base_xyz[:, 2],
            "alternate_x_m": alt_xyz[:, 0],
            "alternate_y_m": alt_xyz[:, 1],
            "alternate_z_m": alt_xyz[:, 2],
            "blended_x_m": blended_xyz[:, 0],
            "blended_y_m": blended_xyz[:, 1],
            "blended_z_m": blended_xyz[:, 2],
            "base_classification": base["Classification"].to_numpy(int),
            "alternate_classification": alternate["Classification"].to_numpy(int),
            "classification": classes.astype(int),
            "base_to_alternate_displacement_m": displacement.astype(float),
            "applied_displacement_m": (weights * displacement).astype(float),
        }
    )
    weights_df = pd.DataFrame(
        {
            "sequence_id": sorted({*base["sequence_id"].astype(str), *weight_map.keys()}),
        }
    )
    weights_df["sequence_gate_weight"] = weights_df["sequence_id"].map(
        lambda sequence: float(weight_map.get(_sequence_weight_key(sequence), default))
    )
    weights_df["weight_source"] = weights_df["sequence_id"].map(
        lambda sequence: "sequence_weights" if _sequence_weight_key(sequence) in weight_map else "default"
    )
    return SequenceGateResult(
        estimates=estimates.reset_index(drop=True),
        diagnostics=diagnostics.reset_index(drop=True),
        sequence_weights=weights_df.reset_index(drop=True),
    )


def write_track5_sequence_gate_outputs(
    *,
    result: SequenceGateResult,
    output_dir: Path,
    base_submission_path: Path,
    alternate_submission_path: Path,
    sequence_weights_path: Path,
    template: pd.DataFrame | None = None,
    manifest: dict[str, Any] | None = None,
    require_leaderboard_ready: bool = False,
) -> dict[str, Path]:
    """Write sequence-gated estimates, official CSV/ZIP, diagnostics, and manifest."""

    if require_leaderboard_ready and template is None:
        raise SystemExit(
            "sequence-gated leaderboard readiness requires an official template; "
            "pass template=... or --template so timestamp coverage can be checked"
        )

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    paths = {
        "estimates_csv": output / SEQUENCE_GATE_ESTIMATES_CSV,
        "results_csv": output / SEQUENCE_GATE_RESULTS_CSV,
        "zip": output / SEQUENCE_GATE_ZIP,
        "diagnostics_csv": output / SEQUENCE_GATE_DIAGNOSTICS_CSV,
        "manifest_json": output / SEQUENCE_GATE_MANIFEST_JSON,
    }
    result.estimates.to_csv(paths["estimates_csv"], index=False)
    result.diagnostics.to_csv(paths["diagnostics_csv"], index=False)
    official_rows = result.estimates.copy()
    # Preserve row-level official labels; class_map is a sequence-level override.
    official_rows["classification"] = official_rows["Classification"]
    write_official_mmaud_results_csv(
        official_rows,
        paths["results_csv"],
        classification=0,
        invalid_row_policy="raise",
    )
    write_official_ug2_codabench_zip(
        official_rows,
        paths["zip"],
        classification=0,
        invalid_row_policy="raise",
    )
    validation_summary: dict[str, Any] | None = None
    if template is not None:
        validation = validate_official_track5_submission(
            paths["zip"],
            template=template,
            require_zip=True,
        )
        paths["validation_json"] = output / SEQUENCE_GATE_VALIDATION_JSON
        paths["validation_rows_csv"] = output / SEQUENCE_GATE_VALIDATION_ROWS_CSV
        paths["validation_json"].write_text(
            json.dumps(_jsonable(validation.summary), indent=2),
            encoding="utf-8",
        )
        validation.rows.to_csv(paths["validation_rows_csv"], index=False)
        validation_summary = _jsonable(validation.summary)
        if require_leaderboard_ready and not validation.summary.get("leaderboard_ready", False):
            reasons = (
                ", ".join(validation.summary.get("leaderboard_blocking_reasons", [])) or "unknown"
            )
            raise SystemExit(f"sequence-gated submission is not leaderboard-ready: {reasons}")
    payload = dict(manifest or {})
    diagnostics = result.diagnostics
    payload.update(
        {
            "schema": "raft-uav-mmuad-track5-sequence-gate-v1",
            "base_submission": str(base_submission_path),
            "alternate_submission": str(alternate_submission_path),
            "sequence_weights": str(sequence_weights_path),
            "row_count": int(len(result.estimates)),
            "sequence_count": int(result.estimates["sequence_id"].nunique())
            if not result.estimates.empty
            else 0,
            "defaulted_sequence_count": int(
                result.sequence_weights["weight_source"].eq("default").sum()
            ),
            "mean_sequence_gate_weight": float(
                result.sequence_weights["sequence_gate_weight"].mean()
            )
            if not result.sequence_weights.empty
            else None,
            "mean_applied_displacement_m": float(diagnostics["applied_displacement_m"].mean())
            if not diagnostics.empty
            else None,
            "p95_applied_displacement_m": float(
                np.percentile(diagnostics["applied_displacement_m"], 95)
            )
            if not diagnostics.empty
            else None,
            "validation": validation_summary,
            "paths": {name: str(path) for name, path in paths.items() if name != "manifest_json"},
        }
    )
    paths["manifest_json"].write_text(json.dumps(_jsonable(payload), indent=2), encoding="utf-8")
    return paths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="raft-uav-mmuad-track5-sequence-gate",
        description="blend two official MMUAD/UG2+ Track 5 submissions with per-sequence weights",
    )
    parser.add_argument("--base-submission", type=Path, required=True)
    parser.add_argument("--alternate-submission", type=Path, required=True)
    parser.add_argument("--sequence-weights", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--template",
        type=Path,
        help="optional official template for preflight validation",
    )
    parser.add_argument("--default-weight", type=float, default=0.0)
    parser.add_argument("--class-policy", choices=("base", "alternate"), default="base")
    parser.add_argument("--require-leaderboard-ready", action="store_true")
    args = parser.parse_args(argv)

    if args.require_leaderboard_ready and args.template is None:
        parser.error("--require-leaderboard-ready requires --template")

    result = blend_track5_sequence_gate(
        base_submission=load_track5_submission(args.base_submission),
        alternate_submission=load_track5_submission(args.alternate_submission),
        sequence_weights=pd.read_csv(args.sequence_weights),
        default_weight=float(args.default_weight),
        class_policy=args.class_policy,
    )
    template = None if args.template is None else pd.read_csv(args.template)
    paths = write_track5_sequence_gate_outputs(
        result=result,
        output_dir=args.output_dir,
        base_submission_path=args.base_submission,
        alternate_submission_path=args.alternate_submission,
        sequence_weights_path=args.sequence_weights,
        template=template,
        manifest={
            "class_policy": args.class_policy,
            "default_weight": float(args.default_weight),
            "template": None if args.template is None else str(args.template),
        },
        require_leaderboard_ready=bool(args.require_leaderboard_ready),
    )
    manifest = json.loads(paths["manifest_json"].read_text(encoding="utf-8"))
    validation = manifest.get("validation") or {}
    print("mmuad_track5_sequence_gate=ok")
    for name, path in paths.items():
        print(f"{name}={path}")
    if validation:
        print(f"leaderboard_ready={validation.get('leaderboard_ready')}")
        print(f"codabench_upload_ready={validation.get('codabench_upload_ready')}")
    return 0


def _sequence_weight_map(weights: pd.DataFrame) -> dict[str, float]:
    rows = pd.DataFrame(weights).copy()
    if rows.empty:
        return {}
    sequence_column = _first_present(rows, SEQUENCE_ALIASES)
    weight_column = _first_present(rows, WEIGHT_ALIASES)
    if sequence_column is None:
        raise ValueError(f"sequence weights missing one of columns: {SEQUENCE_ALIASES}")
    if weight_column is None:
        raise ValueError(f"sequence weights missing one of columns: {WEIGHT_ALIASES}")

    rows["__sequence_id"] = rows[sequence_column].map(_sequence_weight_key)
    rows = rows.loc[rows["__sequence_id"].notna()].copy()
    if rows.empty:
        return {}

    out: dict[str, float] = {}
    for sequence, group in rows.groupby("__sequence_id", sort=True):
        numeric = pd.to_numeric(group[weight_column], errors="coerce")
        if not np.isfinite(numeric.to_numpy(float)).all():
            raise ValueError(f"non-finite sequence weight for {sequence!r}")
        value = float(numeric.mean())
        out[str(sequence)] = _validate_weight(value, name=f"sequence weight for {sequence}")
    return out


def _sequence_weight_key(value: object) -> str | None:
    try:
        return parse_official_sequence_cell(value)
    except ValueError:
        return None


def _validate_weight(value: float, *, name: str) -> float:
    weight = float(value)
    if not np.isfinite(weight) or weight < 0.0 or weight > 1.0:
        raise ValueError(f"{name} must be finite and in [0, 1], got {value!r}")
    return weight


def _first_present(frame: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    columns = {str(column).lower(): str(column) for column in frame.columns}
    for candidate in candidates:
        match = columns.get(candidate.lower())
        if match is not None:
            return match
    return None


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
