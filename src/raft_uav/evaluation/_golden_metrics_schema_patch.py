"""Require golden-run metrics JSON to use the documented object schema."""

from __future__ import annotations

from importlib import import_module
import json
from pathlib import Path
from typing import Any


_golden_artifacts = import_module("raft_uav.evaluation.golden_artifacts")


def _check_metrics(path: Path) -> list[dict[str, Any]]:
    """Check metrics JSON without accepting list or string membership as keys."""

    rows: list[dict[str, Any]] = []
    try:
        metrics = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return [
            {
                "check": "metrics_json_parse",
                "file": str(path),
                "passed": False,
                "message": str(exc),
            }
        ]
    rows.append(
        {
            "check": "metrics_json_parse",
            "file": str(path),
            "passed": True,
            "message": "",
        }
    )
    is_object = isinstance(metrics, dict)
    rows.append(
        {
            "check": "metrics_json_object",
            "file": str(path),
            "passed": is_object,
            "message": "" if is_object else "metrics JSON root must be an object",
        }
    )
    if not is_object:
        return rows
    for key in ("posterior_records", "accepted_measurements", "position_error_3d"):
        rows.append(
            {
                "check": "metrics_required_key",
                "file": str(path),
                "key": key,
                "passed": key in metrics,
                "message": "" if key in metrics else f"missing key {key}",
            }
        )
    return rows


def install() -> None:
    """Install the top-level metrics-schema guard once."""

    if getattr(_golden_artifacts, "_metrics_schema_patch_applied", False):
        return
    _golden_artifacts._check_metrics = _check_metrics
    _golden_artifacts._metrics_schema_patch_applied = True


install()
