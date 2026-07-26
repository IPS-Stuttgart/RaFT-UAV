"""Reject lossy schema-version coercion in candidate-reservoir configs."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def _exact_integer_scalar(value: object, *, name: str) -> int:
    """Return an exact finite integer scalar without Boolean coercion."""

    error = f"{name} must be an exact integer scalar"
    if np.ma.is_masked(value) or isinstance(value, (bool, np.bool_)):
        raise ValueError(error)
    try:
        scalar = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(error) from exc
    if scalar.ndim != 0 or np.iscomplexobj(scalar):
        raise ValueError(error)
    try:
        item = scalar.item()
        if np.ma.is_masked(item) or isinstance(
            item,
            (bool, np.bool_, complex, np.complexfloating),
        ):
            raise ValueError(error)
        number = float(item)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(error) from exc
    if not np.isfinite(number) or not number.is_integer():
        raise ValueError(error)
    return int(number)


def install() -> None:
    """Install exact schema validation on the train-selected config loader."""

    from raft_uav.mmuad import candidate_reservoir_apply as apply_module

    if getattr(
        apply_module,
        "_raft_uav_exact_schema_validation_installed",
        False,
    ):
        return

    def validated_loader(path: Path) -> dict[str, object]:
        config_path = Path(path)
        payload = json.loads(config_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("candidate reservoir config JSON must contain an object")
        schema_version = _exact_integer_scalar(
            payload.get("schema_version", 1),
            name="candidate reservoir config schema_version",
        )
        if schema_version != 1:
            raise ValueError(
                f"unsupported candidate reservoir config schema: {schema_version}"
            )
        missing = [
            key for key in apply_module._REQUIRED_CONFIG_KEYS if key not in payload
        ]
        if missing:
            raise ValueError(
                f"candidate reservoir config missing required keys: {missing}"
            )
        validated = dict(payload)
        validated["schema_version"] = schema_version
        validated["branch_score_offsets"] = apply_module._float_mapping(
            payload.get("branch_score_offsets", {})
        )
        validated["source_score_offsets"] = apply_module._float_mapping(
            payload.get("source_score_offsets", {})
        )
        return validated

    apply_module.load_train_selected_reservoir_config = validated_loader
    apply_module._raft_uav_exact_schema_validation_installed = True
