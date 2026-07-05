"""Apply train-selected Track 5 estimate-ensemble weights to new splits.

``track5_estimate_ensemble_weight_search`` writes a JSON file with selected
weights.  This module consumes that JSON on validation or hidden-test estimate
CSVs, then delegates to the upload-ready estimate ensemble writer.  The command
is inference-safe: it reads fixed weights and a timestamp template only; it does
not consume truth labels.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import json
from typing import Any, Iterable

import numpy as np

from raft_uav.mmuad.submission import load_official_track5_template_file, load_sequence_class_map
from raft_uav.mmuad.track5_estimate_ensemble import EstimateInput
from raft_uav.mmuad.track5_estimate_ensemble import parse_estimate_spec
from raft_uav.mmuad.track5_estimate_ensemble import write_track5_estimate_ensemble_outputs

APPLY_MANIFEST_JSON = "mmuad_track5_ensemble_applied_weights_manifest.json"


def load_ensemble_weight_config(path: Path) -> dict[str, Any]:
    """Load and validate an ensemble-weight-search JSON config."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    weights = payload.get("weights")
    if not isinstance(weights, dict) or not weights:
        raise ValueError("weight config must contain a non-empty 'weights' object")
    parsed_weights: dict[str, float] = {}
    original_labels: dict[str, str] = {}
    for label, value in weights.items():
        safe_label = _safe_label(str(label))
        if safe_label in parsed_weights:
            previous = original_labels[safe_label]
            raise ValueError(
                "weight config labels must be unique after normalization; "
                f"{previous!r} and {str(label)!r} both map to {safe_label!r}"
            )
        weight = float(value)
        if not np.isfinite(weight) or weight < 0.0:
            raise ValueError(f"weight for {label!r} must be finite and non-negative")
        parsed_weights[safe_label] = weight
        original_labels[safe_label] = str(label)
    total = sum(parsed_weights.values())
    if total <= 0.0:
        raise ValueError("weight config sum must be positive")
    payload["weights"] = parsed_weights
    return payload


def apply_ensemble_weight_config(
    estimate_specs: Iterable[str | EstimateInput],
    weight_config: dict[str, Any],
    *,
    missing_weight_policy: str = "error",
    default_missing_weight: float = 0.0,
) -> list[EstimateInput]:
    """Return estimate inputs with weights from a selected-weight JSON config."""

    if missing_weight_policy not in {"error", "zero", "default"}:
        raise ValueError("missing_weight_policy must be one of: error, zero, default")
    weights = {str(key): float(value) for key, value in weight_config.get("weights", {}).items()}
    if not weights:
        raise ValueError("weight config has no weights")
    estimate_inputs: list[EstimateInput] = []
    seen_labels: set[str] = set()
    for spec in estimate_specs:
        item = spec if isinstance(spec, EstimateInput) else parse_estimate_spec(str(spec))
        label = _safe_label(item.label)
        if label in seen_labels:
            raise ValueError(f"duplicate estimate label after normalization: {label}")
        seen_labels.add(label)
        if label in weights:
            weight = weights[label]
        elif missing_weight_policy == "error":
            raise ValueError(f"missing selected ensemble weight for estimate label: {label}")
        elif missing_weight_policy == "zero":
            weight = 0.0
        else:
            weight = float(default_missing_weight)
        estimate_inputs.append(EstimateInput(label=label, path=item.path, weight=weight))
    return estimate_inputs


def write_apply_weights_outputs(
    *,
    estimate_inputs: Iterable[EstimateInput],
    weight_config: dict[str, Any],
    template_path: Path,
    output_dir: Path,
    class_map: dict[str, str] | None = None,
    default_classification: int | str = 0,
    aggregation_policy: str | None = None,
    trim_fraction: float | None = None,
    max_nearest_time_delta_s: float | None = None,
) -> dict[str, Path]:
    """Write weighted ensemble outputs plus an application manifest."""

    estimate_input_list = list(estimate_inputs)
    template = load_official_track5_template_file(template_path)
    selected_aggregation = str(
        aggregation_policy
        if aggregation_policy is not None
        else weight_config.get("aggregation_policy", "weighted-mean")
    )
    selected_trim = float(
        trim_fraction if trim_fraction is not None else weight_config.get("trim_fraction", 0.2)
    )
    selected_max_delta = (
        max_nearest_time_delta_s
        if max_nearest_time_delta_s is not None
        else weight_config.get("max_nearest_time_delta_s")
    )
    if selected_max_delta is not None:
        selected_max_delta = float(selected_max_delta)
    paths = write_track5_estimate_ensemble_outputs(
        estimate_inputs=estimate_input_list,
        template=template,
        output_dir=output_dir,
        class_map=class_map or {},
        default_classification=default_classification,
        max_nearest_time_delta_s=selected_max_delta,
        aggregation_policy=selected_aggregation,
        trim_fraction=selected_trim,
    )
    manifest_path = Path(output_dir) / APPLY_MANIFEST_JSON
    manifest = {
        "schema": "raft-uav-mmuad-track5-estimate-ensemble-apply-weights-v1",
        "weight_config_schema": weight_config.get("schema"),
        "applied_weights": {item.label: float(item.weight) for item in estimate_input_list},
        "aggregation_policy": selected_aggregation,
        "trim_fraction": selected_trim,
        "max_nearest_time_delta_s": selected_max_delta,
        "template_path": str(template_path),
        "output_paths": {name: str(path) for name, path in paths.items()},
    }
    manifest_path.write_text(json.dumps(_jsonable(manifest), indent=2), encoding="utf-8")
    paths["apply_manifest_json"] = manifest_path
    return paths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="raft-uav-mmuad-track5-apply-ensemble-weights",
        description="apply train-selected estimate ensemble weights to Track 5 estimates",
    )
    parser.add_argument(
        "--estimate-csv",
        action="append",
        default=[],
        metavar="LABEL=PATH",
        help="estimate trajectory to include; labels must match the weight config",
    )
    parser.add_argument("--weights-json", type=Path, required=True)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--class-map", type=Path)
    parser.add_argument("--default-classification", default="0")
    parser.add_argument(
        "--missing-weight-policy",
        choices=("error", "zero", "default"),
        default="error",
    )
    parser.add_argument("--default-missing-weight", type=float, default=0.0)
    parser.add_argument("--aggregation-policy")
    parser.add_argument("--trim-fraction", type=float)
    parser.add_argument("--max-nearest-time-delta-s", type=float)
    parser.add_argument("--require-leaderboard-ready", action="store_true")
    args = parser.parse_args(argv)

    if not args.estimate_csv:
        parser.error("provide at least one --estimate-csv LABEL=PATH")
    weight_config = load_ensemble_weight_config(args.weights_json)
    estimate_inputs = apply_ensemble_weight_config(
        args.estimate_csv,
        weight_config,
        missing_weight_policy=args.missing_weight_policy,
        default_missing_weight=float(args.default_missing_weight),
    )
    class_map = load_sequence_class_map(args.class_map) if args.class_map is not None else {}
    paths = write_apply_weights_outputs(
        estimate_inputs=estimate_inputs,
        weight_config=weight_config,
        template_path=args.template,
        output_dir=args.output_dir,
        class_map=class_map,
        default_classification=args.default_classification,
        aggregation_policy=args.aggregation_policy,
        trim_fraction=args.trim_fraction,
        max_nearest_time_delta_s=args.max_nearest_time_delta_s,
    )
    validation = json.loads(Path(paths["validation_json"]).read_text(encoding="utf-8"))
    print("mmuad_track5_apply_ensemble_weights=ok")
    for name, path in paths.items():
        print(f"{name}={path}")
    print(f"leaderboard_ready={validation.get('leaderboard_ready')}")
    print(f"codabench_upload_ready={validation.get('codabench_upload_ready')}")
    if args.require_leaderboard_ready and not validation.get("leaderboard_ready", False):
        reasons = ", ".join(validation.get("leaderboard_blocking_reasons", [])) or "unknown"
        raise SystemExit(f"applied-weight ensemble is not leaderboard-ready: {reasons}")
    return 0


def _safe_label(value: str) -> str:
    label = str(value).strip().replace(" ", "_").replace("/", "_").replace("\\", "_")
    return label or "estimate"


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
