"""Apply a train-selected MMUAD candidate-reservoir config without truth labels.

``candidate_reservoir_train_cv`` writes branch/source score offsets and reservoir
sizes selected on training sequences. This module is the inference-side
companion: it applies that frozen JSON to validation/test candidate streams,
builds the branch-preserving reservoir, and writes provenance suitable for a
paper or hidden-test submission workflow.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from raft_uav.mmuad.candidate_reservoir import (
    ReservoirConfig,
    build_candidate_reservoir,
    build_reservoir_summary,
    load_candidate_inputs,
)
from raft_uav.mmuad.candidate_reservoir_diversity import (
    diversity_cap_reservoir,
    diversity_cap_summary,
)
from raft_uav.mmuad.candidate_reservoir_spatial import (
    spatial_diversity_cap_reservoir,
    spatial_diversity_summary,
)
from raft_uav.mmuad.schema import normalize_candidate_columns

_OUTPUT_CSV = "mmuad_candidate_reservoir_applied.csv"
_SUMMARY_JSON = "mmuad_candidate_reservoir_apply_summary.json"
_PROVENANCE_JSON = "mmuad_candidate_reservoir_apply_provenance.json"
_CAP_MODES = ("score", "diversity", "spatial")
_REQUIRED_CONFIG_KEYS = (
    "score_column",
    "fallback_score_column",
    "global_top_n",
    "per_source_top_n",
    "per_branch_top_n",
    "max_candidates_per_frame",
)


@dataclass(frozen=True)
class ReservoirApplyResult:
    """Outputs from applying a frozen train-selected reservoir config."""

    adjusted_candidates: pd.DataFrame
    pre_cap_reservoir: pd.DataFrame
    reservoir: pd.DataFrame
    summary: dict[str, Any]


def load_train_selected_reservoir_config(path: Path) -> dict[str, Any]:
    """Load and validate a train-selected candidate-reservoir JSON file."""

    config_path = Path(path)
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("candidate reservoir config JSON must contain an object")
    schema_version = int(payload.get("schema_version", 1))
    if schema_version != 1:
        raise ValueError(f"unsupported candidate reservoir config schema: {schema_version}")
    missing = [key for key in _REQUIRED_CONFIG_KEYS if key not in payload]
    if missing:
        raise ValueError(f"candidate reservoir config missing required keys: {missing}")
    payload = dict(payload)
    payload["schema_version"] = schema_version
    payload["branch_score_offsets"] = _float_mapping(payload.get("branch_score_offsets", {}))
    payload["source_score_offsets"] = _float_mapping(payload.get("source_score_offsets", {}))
    return payload


def add_train_selected_reservoir_scores(
    candidates: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    """Attach the frozen base score, branch/source offsets, and adjusted score."""

    rows = normalize_candidate_columns(pd.DataFrame(candidates).copy())
    if rows.empty:
        return rows.assign(
            candidate_reservoir_train_base_score=pd.Series(dtype=float),
            candidate_reservoir_train_branch_offset=pd.Series(dtype=float),
            candidate_reservoir_train_source_offset=pd.Series(dtype=float),
            candidate_reservoir_train_adjusted_score=pd.Series(dtype=float),
        )
    rows = rows.copy()
    if "candidate_branch" not in rows.columns:
        rows["candidate_branch"] = rows["source"].fillna("candidate").astype(str)
    rows["candidate_branch"] = rows["candidate_branch"].fillna("candidate").astype(str)
    rows["source"] = rows["source"].fillna("candidate").astype(str)

    primary = _numeric_column(rows, str(config["score_column"]), default=np.nan)
    fallback = _numeric_column(
        rows,
        str(config["fallback_score_column"]),
        default=1.0,
    )
    branch_offsets = _float_mapping(config.get("branch_score_offsets", {}))
    source_offsets = _float_mapping(config.get("source_score_offsets", {}))
    rows["candidate_reservoir_train_base_score"] = primary.fillna(fallback).fillna(0.0)
    rows["candidate_reservoir_train_branch_offset"] = (
        rows["candidate_branch"].map(branch_offsets).fillna(0.0).astype(float)
    )
    rows["candidate_reservoir_train_source_offset"] = (
        rows["source"].map(source_offsets).fillna(0.0).astype(float)
    )
    rows["candidate_reservoir_train_adjusted_score"] = (
        rows["candidate_reservoir_train_base_score"]
        + rows["candidate_reservoir_train_branch_offset"]
        + rows["candidate_reservoir_train_source_offset"]
    )
    return rows


def apply_train_selected_reservoir_config(
    candidates: pd.DataFrame,
    config: dict[str, Any],
    *,
    cap_mode: str = "score",
    diversity_min_per_source: int = 1,
    diversity_min_per_branch: int = 1,
    spatial_diversity_weight: float = 1.0,
    spatial_diversity_scale_m: float = 10.0,
    spatial_distance_cap_m: float = 50.0,
) -> ReservoirApplyResult:
    """Apply a frozen train-selected reservoir config to target candidates.

    ``score`` exactly reproduces the cap used by the current train-CV selector.
    ``diversity`` and ``spatial`` are explicit inference ablations that preserve
    provenance quotas, with ``spatial`` also filling the remaining budget using
    geometric diversity.
    """

    if cap_mode not in _CAP_MODES:
        raise ValueError(f"cap_mode must be one of {_CAP_MODES}, got {cap_mode!r}")
    adjusted = add_train_selected_reservoir_scores(candidates, config)
    max_candidates = int(config["max_candidates_per_frame"])
    build_cap = max_candidates if cap_mode == "score" else 0
    pre_cap = build_candidate_reservoir(
        adjusted,
        config=ReservoirConfig(
            global_top_n=int(config["global_top_n"]),
            per_source_top_n=int(config["per_source_top_n"]),
            per_branch_top_n=int(config["per_branch_top_n"]),
            max_candidates_per_frame=build_cap,
            score_column="candidate_reservoir_train_adjusted_score",
            fallback_score_column=str(config["fallback_score_column"]),
            score_floor_quantile=_optional_float(config.get("score_floor_quantile")),
        ),
    )
    if cap_mode == "diversity":
        reservoir = diversity_cap_reservoir(
            pre_cap,
            max_candidates_per_frame=max_candidates,
            min_per_source=int(diversity_min_per_source),
            min_per_branch=int(diversity_min_per_branch),
            score_column="candidate_reservoir_score",
            fallback_score_column=str(config["fallback_score_column"]),
        )
    elif cap_mode == "spatial":
        reservoir = spatial_diversity_cap_reservoir(
            pre_cap,
            max_candidates_per_frame=max_candidates,
            min_per_source=int(diversity_min_per_source),
            min_per_branch=int(diversity_min_per_branch),
            score_column="candidate_reservoir_score",
            fallback_score_column=str(config["fallback_score_column"]),
            spatial_diversity_weight=float(spatial_diversity_weight),
            spatial_diversity_scale_m=float(spatial_diversity_scale_m),
            spatial_distance_cap_m=float(spatial_distance_cap_m),
        )
    else:
        reservoir = pre_cap

    summary = build_reservoir_summary(adjusted, reservoir)
    summary.update(
        {
            "truth_free": True,
            "config_schema_version": int(config.get("schema_version", 1)),
            "selection_protocol": config.get("selection_protocol"),
            "selection_metric": config.get("selection_metric"),
            "selected_grid_label": config.get("selected_grid_label"),
            "selected_metric_value": config.get("selected_metric_value"),
            "branch_score_offsets": _float_mapping(config.get("branch_score_offsets", {})),
            "source_score_offsets": _float_mapping(config.get("source_score_offsets", {})),
            "cap_mode": cap_mode,
            "diversity_min_per_source": int(diversity_min_per_source),
            "diversity_min_per_branch": int(diversity_min_per_branch),
            "spatial_diversity_weight": float(spatial_diversity_weight),
            "spatial_diversity_scale_m": float(spatial_diversity_scale_m),
            "spatial_distance_cap_m": float(spatial_distance_cap_m),
            "pre_cap_candidate_rows": int(len(pre_cap)),
        }
    )
    if cap_mode == "diversity":
        summary["diversity_cap"] = diversity_cap_summary(pre_cap, reservoir)
    elif cap_mode == "spatial":
        summary["spatial_cap"] = spatial_diversity_summary(pre_cap, reservoir)
    return ReservoirApplyResult(
        adjusted_candidates=adjusted,
        pre_cap_reservoir=pre_cap,
        reservoir=reservoir,
        summary=summary,
    )


def write_train_selected_reservoir_outputs(
    result: ReservoirApplyResult,
    *,
    output_dir: Path,
    config_path: Path,
    candidate_specs: Sequence[str],
    output_csv: Path | None = None,
    adjusted_candidates_csv: Path | None = None,
    summary_json: Path | None = None,
    provenance_json: Path | None = None,
) -> dict[str, Path]:
    """Write the applied reservoir and reproducibility metadata."""

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    paths = {
        "reservoir_csv": output_csv or destination / _OUTPUT_CSV,
        "summary_json": summary_json or destination / _SUMMARY_JSON,
        "provenance_json": provenance_json or destination / _PROVENANCE_JSON,
    }
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    result.reservoir.to_csv(paths["reservoir_csv"], index=False)
    paths["summary_json"].write_text(
        json.dumps(_jsonable(result.summary), indent=2),
        encoding="utf-8",
    )

    if adjusted_candidates_csv is not None:
        adjusted_candidates_csv.parent.mkdir(parents=True, exist_ok=True)
        result.adjusted_candidates.to_csv(adjusted_candidates_csv, index=False)
        paths["adjusted_candidates_csv"] = adjusted_candidates_csv

    provenance = {
        "truth_free": True,
        "config_json": str(config_path),
        "config_sha256": _sha256(config_path),
        "candidate_specs": [str(spec) for spec in candidate_specs],
        "reservoir_csv": str(paths["reservoir_csv"]),
        "summary_json": str(paths["summary_json"]),
        "adjusted_candidates_csv": (
            str(paths["adjusted_candidates_csv"])
            if "adjusted_candidates_csv" in paths
            else None
        ),
        "input_candidate_rows": int(len(result.adjusted_candidates)),
        "pre_cap_candidate_rows": int(len(result.pre_cap_reservoir)),
        "reservoir_candidate_rows": int(len(result.reservoir)),
        "cap_mode": result.summary["cap_mode"],
        "selected_grid_label": result.summary.get("selected_grid_label"),
        "selection_protocol": result.summary.get("selection_protocol"),
    }
    paths["provenance_json"].write_text(
        json.dumps(_jsonable(provenance), indent=2),
        encoding="utf-8",
    )
    return paths


def _numeric_column(rows: pd.DataFrame, column: str, *, default: float) -> pd.Series:
    if column in rows.columns:
        return pd.to_numeric(rows[column], errors="coerce")
    return pd.Series(default, index=rows.index, dtype=float)


def _float_mapping(value: Any) -> dict[str, float]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError("branch/source score offsets must be JSON objects")
    return {str(key): float(item) for key, item in value.items()}


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="raft-uav-mmuad-apply-candidate-reservoir-config",
        description="apply a train-selected MMUAD candidate-reservoir config",
    )
    parser.add_argument("--config-json", type=Path, required=True)
    parser.add_argument(
        "--candidate",
        action="append",
        default=[],
        help="candidate CSV as BRANCH=path; may be repeated",
    )
    parser.add_argument(
        "--candidate-csv",
        action="append",
        default=[],
        help="alias for --candidate",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path)
    parser.add_argument("--adjusted-candidates-csv", type=Path)
    parser.add_argument("--summary-json", type=Path)
    parser.add_argument("--provenance-json", type=Path)
    parser.add_argument("--cap-mode", choices=_CAP_MODES, default="score")
    parser.add_argument("--diversity-min-per-source", type=int, default=1)
    parser.add_argument("--diversity-min-per-branch", type=int, default=1)
    parser.add_argument("--spatial-diversity-weight", type=float, default=1.0)
    parser.add_argument("--spatial-diversity-scale-m", type=float, default=10.0)
    parser.add_argument("--spatial-distance-cap-m", type=float, default=50.0)
    args = parser.parse_args(argv)

    candidate_specs = [*args.candidate, *args.candidate_csv]
    candidates = load_candidate_inputs(candidate_specs)
    if candidates.empty:
        parser.error("provide at least one non-empty --candidate BRANCH=PATH CSV")
    config = load_train_selected_reservoir_config(args.config_json)
    result = apply_train_selected_reservoir_config(
        candidates,
        config,
        cap_mode=args.cap_mode,
        diversity_min_per_source=args.diversity_min_per_source,
        diversity_min_per_branch=args.diversity_min_per_branch,
        spatial_diversity_weight=args.spatial_diversity_weight,
        spatial_diversity_scale_m=args.spatial_diversity_scale_m,
        spatial_distance_cap_m=args.spatial_distance_cap_m,
    )
    paths = write_train_selected_reservoir_outputs(
        result,
        output_dir=args.output_dir,
        config_path=args.config_json,
        candidate_specs=candidate_specs,
        output_csv=args.output_csv,
        adjusted_candidates_csv=args.adjusted_candidates_csv,
        summary_json=args.summary_json,
        provenance_json=args.provenance_json,
    )
    print("mmuad_candidate_reservoir_apply=ok")
    print(f"input_rows={len(result.adjusted_candidates)}")
    print(f"reservoir_rows={len(result.reservoir)}")
    print(f"selected_grid_label={result.summary.get('selected_grid_label')}")
    for name, path in paths.items():
        print(f"{name}={path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
