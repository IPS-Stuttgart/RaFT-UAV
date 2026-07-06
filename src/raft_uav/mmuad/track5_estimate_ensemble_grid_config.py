"""Export reusable weight configs from Track 5 ensemble-grid manifests.

The ensemble-grid scorer writes the best weights and policy inside a manifest.
For leaderboard runs, the upload-time estimate ensemble expects a compact JSON
mapping that can be applied to hidden-test estimates.  This module converts the
truth-aware grid manifest into such an inference-safe config artifact.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

WEIGHT_CONFIG_SCHEMA = "raft-uav-mmuad-track5-estimate-ensemble-weight-config-v1"
DEFAULT_OUTPUT_JSON = "mmuad_track5_estimate_ensemble_best_weight_config.json"


def build_weight_config_from_grid_manifest(
    manifest: dict[str, Any],
    *,
    source_manifest: str | None = None,
) -> dict[str, Any]:
    """Build a reusable ensemble-weight config from a grid manifest dict."""

    inputs = manifest.get("estimate_inputs")
    weights = manifest.get("best_weights")
    if not isinstance(inputs, list) or not inputs:
        raise ValueError("grid manifest must contain non-empty estimate_inputs")
    if not isinstance(weights, list) or not weights:
        raise ValueError("grid manifest must contain non-empty best_weights")
    if len(inputs) != len(weights):
        raise ValueError(
            "grid manifest estimate_inputs and best_weights lengths differ: "
            f"{len(inputs)} != {len(weights)}"
        )
    labels: list[str] = []
    for index, item in enumerate(inputs):
        if not isinstance(item, dict):
            raise ValueError(f"estimate_inputs[{index}] must be an object")
        label = _safe_label(item.get("label", ""))
        if not label:
            raise ValueError(f"estimate_inputs[{index}] missing label")
        labels.append(label)
    if len(set(labels)) != len(labels):
        raise ValueError(f"estimate labels must be unique: {labels}")
    weight_map = {
        label: _finite_nonnegative_weight(weight, label=label)
        for label, weight in zip(labels, weights, strict=True)
    }
    total_weight = float(sum(weight_map.values()))
    if total_weight <= 0.0:
        raise ValueError("best_weights must have positive total weight")
    metrics = manifest.get("best", {}) if isinstance(manifest.get("best", {}), dict) else {}
    return {
        "schema": WEIGHT_CONFIG_SCHEMA,
        "weights": weight_map,
        "weight_sum": total_weight,
        "aggregation_policy": str(manifest.get("best_aggregation_policy", "weighted-mean")),
        "trim_fraction": float(manifest.get("best_trim_fraction", manifest.get("trim_fraction", 0.2))),
        "source_manifest": source_manifest,
        "grid_schema": manifest.get("schema"),
        "grid_row_count": manifest.get("grid_row_count"),
        "metrics": _jsonable(metrics),
    }


def write_weight_config_from_grid_manifest(
    *,
    manifest_json: Path,
    output_json: Path,
) -> dict[str, Any]:
    """Read a grid manifest and write a compact weight config JSON."""

    manifest_path = Path(manifest_json)
    output_path = Path(output_json)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("grid manifest JSON must be an object")
    config = build_weight_config_from_grid_manifest(
        manifest,
        source_manifest=str(manifest_path),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    return config


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="raft-uav-mmuad-track5-ensemble-grid-config",
        description="export reusable Track 5 estimate-ensemble weights from a grid manifest",
    )
    parser.add_argument("--grid-manifest-json", type=Path, required=True)
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args(argv)

    output_json = args.output_json
    if output_json is None:
        output_json = args.grid_manifest_json.parent / DEFAULT_OUTPUT_JSON
    config = write_weight_config_from_grid_manifest(
        manifest_json=args.grid_manifest_json,
        output_json=output_json,
    )
    print("mmuad_track5_ensemble_grid_config=ok")
    print(f"output_json={output_json}")
    print(f"aggregation_policy={config['aggregation_policy']}")
    print(f"weights={config['weights']}")
    return 0


def _finite_nonnegative_weight(value: object, *, label: str) -> float:
    weight = float(value)
    if not np.isfinite(weight) or weight < 0.0:
        raise ValueError(f"weight for {label} must be finite and non-negative: {value}")
    return weight


def _safe_label(value: object) -> str:
    text = "" if value is None else str(value)
    return text.strip().replace(" ", "_").replace("/", "_").replace("\\", "_")


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
