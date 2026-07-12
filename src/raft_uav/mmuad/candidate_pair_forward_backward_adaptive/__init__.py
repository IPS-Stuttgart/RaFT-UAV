"""Compatibility fixes for entropy-adaptive pair-state candidate scoring.

The implementation remains in the adjacent module.  Loading it into this package
preserves the public API while allowing targeted compatibility fixes without
duplicating the implementation.
"""

from __future__ import annotations

from pathlib import Path as _Path

_IMPLEMENTATION_PATH = (
    _Path(__file__).resolve().parent.parent
    / "candidate_pair_forward_backward_adaptive.py"
)
exec(
    compile(
        _IMPLEMENTATION_PATH.read_text(encoding="utf-8"),
        str(_IMPLEMENTATION_PATH),
        "exec",
    ),
    globals(),
    globals(),
)

_DEFAULT_PAIR_SCORE_COLUMN = "candidate_pair_forward_backward_score"
_original_write_entropy_adaptive_pair_outputs = write_entropy_adaptive_pair_outputs


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
