"""Require complete sequence coverage for estimate-gate apply inputs."""

from __future__ import annotations

from importlib import import_module

import pandas as pd


_impl = import_module("raft_uav.mmuad.track5_estimate_sequence_gate_fit")
_PATCH_MARKER = "_raft_uav_estimate_sequence_apply_coverage_patch_applied"


def _render_sequence_ids(values: set[str]) -> str:
    """Render sequence identifiers deterministically for validation errors."""

    return ", ".join(repr(value) for value in sorted(values))


def _template_for_apply_estimates(
    template: pd.DataFrame,
    apply_base_estimates: pd.DataFrame,
    apply_alternate_estimates: pd.DataFrame,
) -> pd.DataFrame:
    """Return apply-template rows only after exact source coverage validation."""

    base_sequences = _impl._estimate_sequence_ids(
        apply_base_estimates,
        label="apply base estimates",
    )
    alternate_sequences = _impl._estimate_sequence_ids(
        apply_alternate_estimates,
        label="apply alternate estimates",
    )
    if base_sequences != alternate_sequences:
        details: list[str] = []
        missing_from_alternate = base_sequences - alternate_sequences
        missing_from_base = alternate_sequences - base_sequences
        if missing_from_alternate:
            details.append(
                "missing from alternate: "
                f"{_render_sequence_ids(missing_from_alternate)}"
            )
        if missing_from_base:
            details.append(
                f"missing from base: {_render_sequence_ids(missing_from_base)}"
            )
        raise ValueError(
            "apply base/alternate estimates must cover identical sequence sets "
            f"({'; '.join(details)})"
        )
    if not base_sequences:
        raise ValueError(
            "apply base/alternate estimates must share at least one sequence"
        )

    rows = pd.DataFrame(template).copy()
    sequence_column = _impl._first_present(rows, _impl.SEQUENCE_ALIASES)
    if sequence_column is None:
        raise ValueError("template must contain a sequence column")
    template_sequences = set(rows[sequence_column].dropna().astype(str))
    missing_from_template = base_sequences - template_sequences
    if missing_from_template:
        raise ValueError(
            "template is missing apply estimate sequences: "
            f"{_render_sequence_ids(missing_from_template)}"
        )

    filtered = rows.loc[
        rows[sequence_column].astype(str).isin(base_sequences)
    ].copy()
    if filtered.empty:
        raise ValueError("template has no rows for apply estimate sequences")
    return filtered


def install() -> None:
    """Install the strict apply-coverage boundary once per interpreter."""

    if getattr(_impl, _PATCH_MARKER, False):
        return
    _impl._template_for_apply_estimates = _template_for_apply_estimates
    setattr(_impl, _PATCH_MARKER, True)
