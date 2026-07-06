"""Normalize per-pipeline uncertainty columns for Track 5 estimate ensembling.

Different MMUAD pose pipelines may emit useful per-row uncertainty under
different names, such as ``predicted_sigma_m``, ``state_sigma_m`` or a model-
specific column.  The existing inverse-variance ensemble expects one common
uncertainty column across all inputs.  This helper copies each input's selected
uncertainty column into a common name, writes normalized estimate CSVs, and can
optionally run the upload-ready uncertainty ensemble immediately.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from raft_uav.mmuad.submission import load_official_track5_template_file, load_sequence_class_map
from raft_uav.mmuad.track5_estimate_ensemble import EstimateInput, parse_estimate_spec
from raft_uav.mmuad.track5_uncertainty_ensemble import write_track5_uncertainty_ensemble_outputs

NORMALIZED_DIR = "normalized_estimates"
MANIFEST_JSON = "mmuad_track5_uncertainty_column_adapter_manifest.json"
DEFAULT_UNCERTAINTY_COLUMNS = (
    "predicted_sigma_m",
    "state_sigma_m",
    "position_sigma_m",
    "ensemble_effective_sigma_m",
    "sigma_m",
    "rmse_m",
)


def normalize_uncertainty_estimate_inputs(
    estimate_inputs: Iterable[EstimateInput],
    *,
    output_dir: Path,
    uncertainty_columns: dict[str, str] | None = None,
    output_uncertainty_column: str = "predicted_sigma_m",
    fallback_sigma_m: float = 30.0,
    require_uncertainty: bool = False,
) -> tuple[list[EstimateInput], pd.DataFrame]:
    """Copy each input's uncertainty into one common column and write CSVs."""

    inputs = list(estimate_inputs)
    if not inputs:
        raise ValueError("at least one estimate input is required")
    fallback_sigma_m = _positive_finite(fallback_sigma_m, name="fallback_sigma_m")
    column_map = dict(uncertainty_columns or {})
    normalized_dir = Path(output_dir) / NORMALIZED_DIR
    normalized_dir.mkdir(parents=True, exist_ok=True)
    normalized_inputs: list[EstimateInput] = []
    records: list[dict[str, Any]] = []
    for item in inputs:
        rows = pd.read_csv(item.path)
        source_column = _select_uncertainty_column(
            rows,
            label=item.label,
            requested=_lookup_requested_uncertainty_column(column_map, item.label),
            require_uncertainty=require_uncertainty,
        )
        out = rows.copy()
        if source_column is None:
            out[output_uncertainty_column] = float(fallback_sigma_m)
            source = "fallback"
        else:
            values = pd.to_numeric(out[source_column], errors="coerce")
            finite_positive = np.isfinite(values.to_numpy(float)) & (values > 0.0).to_numpy(bool)
            fallback_count = int((~finite_positive).sum())
            out[output_uncertainty_column] = values.where(finite_positive, float(fallback_sigma_m))
            source = source_column
        output_csv = normalized_dir / f"{_safe_label(item.label)}.csv"
        out.to_csv(output_csv, index=False)
        normalized_inputs.append(
            EstimateInput(label=item.label, path=output_csv, weight=item.weight)
        )
        sigma = pd.to_numeric(out[output_uncertainty_column], errors="coerce")
        records.append(
            {
                "label": item.label,
                "input_path": str(item.path),
                "output_path": str(output_csv),
                "weight": float(item.weight),
                "source_uncertainty_column": source,
                "output_uncertainty_column": output_uncertainty_column,
                "row_count": int(len(out)),
                "fallback_sigma_m": float(fallback_sigma_m),
                "fallback_row_count": int(len(out)) if source_column is None else fallback_count,
                "mean_sigma_m": _safe_mean(sigma),
                "p95_sigma_m": _safe_percentile(sigma, 95),
            }
        )
    return normalized_inputs, pd.DataFrame.from_records(records)


def write_uncertainty_column_adapter_outputs(
    *,
    estimate_inputs: Iterable[EstimateInput],
    output_dir: Path,
    uncertainty_columns: dict[str, str] | None = None,
    output_uncertainty_column: str = "predicted_sigma_m",
    fallback_sigma_m: float = 30.0,
    require_uncertainty: bool = False,
    template: pd.DataFrame | None = None,
    class_map: dict[str, str] | None = None,
    default_classification: int | str = 0,
    sigma_min_m: float = 1.0,
    sigma_max_m: float = 100.0,
    max_nearest_time_delta_s: float | None = None,
    run_ensemble: bool = False,
) -> dict[str, Path]:
    """Write normalized estimates and optional upload-ready uncertainty ensemble."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    normalized_inputs, summary = normalize_uncertainty_estimate_inputs(
        estimate_inputs,
        output_dir=output,
        uncertainty_columns=uncertainty_columns,
        output_uncertainty_column=output_uncertainty_column,
        fallback_sigma_m=fallback_sigma_m,
        require_uncertainty=require_uncertainty,
    )
    summary_csv = output / "mmuad_track5_uncertainty_column_adapter_summary.csv"
    summary.to_csv(summary_csv, index=False)
    paths: dict[str, Path] = {"summary_csv": summary_csv}
    ensemble_paths: dict[str, Path] = {}
    if run_ensemble:
        if template is None:
            raise ValueError("template is required when run_ensemble=True")
        ensemble_paths = write_track5_uncertainty_ensemble_outputs(
            estimate_inputs=normalized_inputs,
            template=template,
            output_dir=output / "uncertainty_ensemble",
            class_map=class_map or {},
            default_classification=default_classification,
            uncertainty_column=output_uncertainty_column,
            fallback_sigma_m=fallback_sigma_m,
            sigma_min_m=sigma_min_m,
            sigma_max_m=sigma_max_m,
            max_nearest_time_delta_s=max_nearest_time_delta_s,
        )
        paths.update({f"ensemble_{name}": path for name, path in ensemble_paths.items()})
    manifest = {
        "schema": "raft-uav-mmuad-track5-uncertainty-column-adapter-v1",
        "output_uncertainty_column": output_uncertainty_column,
        "fallback_sigma_m": float(fallback_sigma_m),
        "require_uncertainty": bool(require_uncertainty),
        "run_ensemble": bool(run_ensemble),
        "normalized_estimate_inputs": [
            {"label": item.label, "path": str(item.path), "weight": float(item.weight)}
            for item in normalized_inputs
        ],
        "summary_rows": summary.to_dict(orient="records"),
        "paths": {name: str(path) for name, path in paths.items()},
    }
    manifest_json = output / MANIFEST_JSON
    manifest_json.write_text(json.dumps(_jsonable(manifest), indent=2), encoding="utf-8")
    paths["manifest_json"] = manifest_json
    return paths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="raft-uav-mmuad-track5-uncertainty-column-adapter",
        description="normalize per-input uncertainty columns before Track 5 uncertainty ensembling",
    )
    parser.add_argument(
        "--estimate-csv",
        action="append",
        default=[],
        metavar="LABEL=PATH[@WEIGHT]",
        help="estimate trajectory to normalize; may be repeated",
    )
    parser.add_argument(
        "--uncertainty-column",
        action="append",
        default=[],
        metavar="LABEL=COLUMN",
        help="per-input uncertainty column; may be repeated",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-uncertainty-column", default="predicted_sigma_m")
    parser.add_argument("--fallback-sigma-m", type=float, default=30.0)
    parser.add_argument("--require-uncertainty", action="store_true")
    parser.add_argument("--run-ensemble", action="store_true")
    parser.add_argument("--template", type=Path)
    parser.add_argument("--class-map", type=Path)
    parser.add_argument("--default-classification", default="0")
    parser.add_argument("--sigma-min-m", type=float, default=1.0)
    parser.add_argument("--sigma-max-m", type=float, default=100.0)
    parser.add_argument("--max-nearest-time-delta-s", type=float)
    args = parser.parse_args(argv)

    if not args.estimate_csv:
        parser.error("provide at least one --estimate-csv LABEL=PATH[@WEIGHT]")
    if args.run_ensemble and args.template is None:
        parser.error("--template is required with --run-ensemble")
    inputs = [parse_estimate_spec(value) for value in args.estimate_csv]
    columns = _parse_uncertainty_column_map(args.uncertainty_column)
    template = (
        load_official_track5_template_file(args.template) if args.template is not None else None
    )
    class_map = load_sequence_class_map(args.class_map) if args.class_map is not None else {}
    paths = write_uncertainty_column_adapter_outputs(
        estimate_inputs=inputs,
        output_dir=args.output_dir,
        uncertainty_columns=columns,
        output_uncertainty_column=args.output_uncertainty_column,
        fallback_sigma_m=float(args.fallback_sigma_m),
        require_uncertainty=bool(args.require_uncertainty),
        template=template,
        class_map=class_map,
        default_classification=args.default_classification,
        sigma_min_m=float(args.sigma_min_m),
        sigma_max_m=float(args.sigma_max_m),
        max_nearest_time_delta_s=args.max_nearest_time_delta_s,
        run_ensemble=bool(args.run_ensemble),
    )
    print("mmuad_track5_uncertainty_column_adapter=ok")
    for name, path in paths.items():
        print(f"{name}={path}")
    return 0


def _parse_uncertainty_column_map(values: list[str]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"uncertainty-column spec must be LABEL=COLUMN: {value}")
        label, column = value.split("=", 1)
        label = _safe_label(label)
        column = column.strip()
        if not column:
            raise ValueError(f"empty uncertainty column for label {label}")
        mapping[label] = column
    return mapping


def _lookup_requested_uncertainty_column(mapping: dict[str, str], label: str) -> str | None:
    """Return an explicitly requested column for raw or filesystem-safe labels."""

    if label in mapping:
        return mapping[label]
    return mapping.get(_safe_label(label))


def _select_uncertainty_column(
    rows: pd.DataFrame,
    *,
    label: str,
    requested: str | None,
    require_uncertainty: bool,
) -> str | None:
    if requested is not None:
        if requested not in rows.columns:
            raise ValueError(f"requested uncertainty column for {label} not found: {requested}")
        return requested
    for column in DEFAULT_UNCERTAINTY_COLUMNS:
        if column in rows.columns:
            return column
    if require_uncertainty:
        raise ValueError(f"no uncertainty column found for {label}")
    return None


def _safe_label(value: str) -> str:
    label = str(value).strip().replace(" ", "_").replace("/", "_").replace(chr(92), "_")
    return label or "estimate"


def _positive_finite(value: float, *, name: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite and positive") from exc
    if not np.isfinite(parsed) or parsed <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return parsed


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
