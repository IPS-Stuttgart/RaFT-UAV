"""Compatibility fixes for candidate-oracle target exports.

The maintained implementation lives in the sibling ``candidate_oracle_targets.py``
module. This package preserves the public import path while rejecting malformed
truth-matching time gates before they can silently widen or empty the training
export and while keeping distinct floating-point thresholds distinct in output
column labels.
"""

from __future__ import annotations

from dataclasses import replace
import importlib.util
from pathlib import Path
import sys
from typing import Any

import pandas as pd

from raft_uav.numeric import optional_float

_IMPL_PATH = Path(__file__).resolve().parent.parent / "candidate_oracle_targets.py"
_SPEC = importlib.util.spec_from_file_location(
    "raft_uav.mmuad._candidate_oracle_targets_legacy",
    _IMPL_PATH,
)
if _SPEC is None or _SPEC.loader is None:  # pragma: no cover
    raise ImportError(f"cannot load candidate-oracle targets from {_IMPL_PATH}")
_IMPL = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _IMPL
_SPEC.loader.exec_module(_IMPL)

_ORIGINAL_BUILD_CANDIDATE_ORACLE_TARGETS = _IMPL.build_candidate_oracle_targets


def _validated_config(
    config: _IMPL.CandidateOracleTargetConfig | None,
) -> _IMPL.CandidateOracleTargetConfig:
    """Return a config with a finite non-negative truth-time gate."""

    resolved = config or _IMPL.CandidateOracleTargetConfig()
    max_delta = optional_float(resolved.max_truth_time_delta_s)
    if max_delta is None or max_delta < 0.0:
        raise ValueError(
            "max_truth_time_delta_s must be a finite non-negative scalar"
        )
    return replace(resolved, max_truth_time_delta_s=max_delta)


def _threshold_label(value: float) -> str:
    """Return a column-safe shortest round-trip floating-point label."""

    text = repr(float(value))
    if text.endswith(".0") and "e" not in text.lower():
        text = text[:-2]
    return text.replace("-", "m").replace(".", "p").replace("+", "")


def build_candidate_oracle_targets(
    candidates: pd.DataFrame,
    truth: pd.DataFrame,
    *,
    config: _IMPL.CandidateOracleTargetConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Build targets after validating the truth-matching time gate."""

    return _ORIGINAL_BUILD_CANDIDATE_ORACLE_TARGETS(
        candidates,
        truth,
        config=_validated_config(config),
    )


_IMPL._threshold_label = _threshold_label
_IMPL.build_candidate_oracle_targets = build_candidate_oracle_targets

globals().update(
    {
        name: getattr(_IMPL, name)
        for name in dir(_IMPL)
        if not (name.startswith("__") and name.endswith("__"))
    }
)
globals()["_validated_config"] = _validated_config
globals()["_threshold_label"] = _threshold_label
globals()["build_candidate_oracle_targets"] = build_candidate_oracle_targets

__doc__ = _IMPL.__doc__
__all__ = [
    name for name in dir(_IMPL) if not (name.startswith("__") and name.endswith("__"))
]
