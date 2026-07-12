"""Compatibility fixes for entropy-adaptive pair-state candidate scoring.

The implementation remains in the adjacent module.  This package loads that
implementation under a private module name, re-exports its public API, and applies
the summary-column fix without duplicating the full algorithm.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
from typing import Any, Mapping

import pandas as pd

from raft_uav.mmuad.candidate_pair_forward_backward import (
    CandidatePairForwardBackwardConfig,
)
from raft_uav.mmuad.schema import CandidateFrame

_IMPLEMENTATION_PATH = (
    Path(__file__).resolve().parent.parent
    / "candidate_pair_forward_backward_adaptive.py"
)
_IMPLEMENTATION_NAME = f"{__name__}._implementation"
_IMPLEMENTATION_SPEC = importlib.util.spec_from_file_location(
    _IMPLEMENTATION_NAME,
    _IMPLEMENTATION_PATH,
)
if _IMPLEMENTATION_SPEC is None or _IMPLEMENTATION_SPEC.loader is None:
    raise ImportError(f"cannot load adaptive pair implementation from {_IMPLEMENTATION_PATH}")
_implementation = importlib.util.module_from_spec(_IMPLEMENTATION_SPEC)
sys.modules[_IMPLEMENTATION_NAME] = _implementation
_IMPLEMENTATION_SPEC.loader.exec_module(_implementation)

DEFAULT_OUTPUT_SCORE_COLUMN: str = _implementation.DEFAULT_OUTPUT_SCORE_COLUMN
EntropyAdaptivePairBlendConfig = _implementation.EntropyAdaptivePairBlendConfig
_candidate_rows = _implementation._candidate_rows
_different_fraction = _implementation._different_fraction
_jsonable = _implementation._jsonable
_posterior_sum_error = _implementation._posterior_sum_error
_safe_mean = _implementation._safe_mean
_safe_quantile = _implementation._safe_quantile
_top_candidate_ids = _implementation._top_candidate_ids
_original_write_entropy_adaptive_pair_outputs = (
    _implementation.write_entropy_adaptive_pair_outputs
)

for _public_name in dir(_implementation):
    if not _public_name.startswith("_"):
        globals().setdefault(_public_name, getattr(_implementation, _public_name))


_DEFAULT_PAIR_SCORE_COLUMN = "candidate_pair_forward_backward_score"


def entropy_adaptive_pair_summary(
    candidates: CandidateFrame | pd.DataFrame,
    *,
    output_score_column: str = DEFAULT_OUTPUT_SCORE_COLUMN,
    pair_score_column: str = _DEFAULT_PAIR_SCORE_COLUMN,
) -> dict[str, Any]:
    """Return diagnostics using the configured pair-posterior score column."""

    rows = _candidate_rows(candidates)
    if rows.empty:
        return {
            "row_count": 0,
            "sequence_count": 0,
            "frame_count": 0,
            "output_score_column": str(output_score_column),
            "pair_score_column": str(pair_score_column),
        }
    frame_columns = ["sequence_id", "time_s"]
    frame_first = rows.groupby(frame_columns, sort=False).first().reset_index()
    pair_weight = pd.to_numeric(
        frame_first["candidate_pair_forward_backward_adaptive_pair_weight"],
        errors="coerce",
    )
    confidence = pd.to_numeric(
        frame_first["candidate_pair_forward_backward_adaptive_confidence"],
        errors="coerce",
    )
    normalized_entropy = pd.to_numeric(
        frame_first["candidate_pair_forward_backward_adaptive_normalized_entropy"],
        errors="coerce",
    )
    adaptive_top = _top_candidate_ids(rows, output_score_column)
    local_top = _top_candidate_ids(
        rows,
        "candidate_pair_forward_backward_local_posterior",
    )
    pair_top = _top_candidate_ids(rows, pair_score_column)
    return {
        "row_count": int(len(rows)),
        "sequence_count": int(rows["sequence_id"].astype(str).nunique()),
        "frame_count": int(len(frame_first)),
        "output_score_column": str(output_score_column),
        "pair_score_column": str(pair_score_column),
        "posterior_sum_error_max": _posterior_sum_error(rows, output_score_column),
        "effective_pair_weight_mean": _safe_mean(pair_weight),
        "effective_pair_weight_p50": _safe_quantile(pair_weight, 0.50),
        "effective_pair_weight_p95": _safe_quantile(pair_weight, 0.95),
        "pair_confidence_mean": _safe_mean(confidence),
        "pair_normalized_entropy_mean": _safe_mean(normalized_entropy),
        "adaptive_top_differs_from_local_fraction": _different_fraction(
            adaptive_top,
            local_top,
        ),
        "adaptive_top_differs_from_pair_fraction": _different_fraction(
            adaptive_top,
            pair_top,
        ),
    }


def write_entropy_adaptive_pair_outputs(
    candidates: CandidateFrame,
    *,
    output_csv: Path,
    summary_json: Path | None = None,
    pair_config: CandidatePairForwardBackwardConfig | None = None,
    blend_config: EntropyAdaptivePairBlendConfig | None = None,
    extra_summary: Mapping[str, Any] | None = None,
) -> None:
    """Write outputs and recompute diagnostics with the configured pair column."""

    _original_write_entropy_adaptive_pair_outputs(
        candidates,
        output_csv=output_csv,
        summary_json=summary_json,
        pair_config=pair_config,
        blend_config=blend_config,
        extra_summary=extra_summary,
    )
    if summary_json is None:
        return

    pair_cfg = pair_config or CandidatePairForwardBackwardConfig()
    blend_cfg = blend_config or EntropyAdaptivePairBlendConfig()
    payload = json.loads(summary_json.read_text(encoding="utf-8"))
    payload["adaptive_summary"] = entropy_adaptive_pair_summary(
        candidates,
        output_score_column=blend_cfg.output_score_column,
        pair_score_column=pair_cfg.output_score_column,
    )
    summary_json.write_text(
        json.dumps(_jsonable(payload), indent=2),
        encoding="utf-8",
    )


_implementation.entropy_adaptive_pair_summary = entropy_adaptive_pair_summary
_implementation.write_entropy_adaptive_pair_outputs = (
    write_entropy_adaptive_pair_outputs
)

__all__ = sorted(
    name for name in dir(_implementation) if not name.startswith("_")
)
