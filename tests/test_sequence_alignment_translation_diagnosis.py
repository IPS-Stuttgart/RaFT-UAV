from __future__ import annotations

import pandas as pd

from raft_uav.mmuad.sequence_alignment_audit import (
    build_sequence_alignment_decision_summary,
)


def _audit_row(
    *,
    variant: str,
    translation_mode: str,
    mean_m: float,
    p95_m: float,
    translation_x_m: float = 0.0,
) -> dict[str, object]:
    return {
        "sequence_id": "seqA",
        "sensor": "lidar_360",
        "variant": variant,
        "translation_mode": translation_mode,
        "translation_x_m": translation_x_m,
        "translation_y_m": 0.0,
        "translation_z_m": 0.0,
        "truth_frame_count": 10,
        "source_frame_count": 10,
        "candidate_count": 100,
        "raw_point_count": 1000,
        "source_time_matched_truth_frame_fraction": 1.0,
        "source_frame_with_candidates_fraction": 1.0,
        "mean_nearest_cluster_to_truth_distance_m": mean_m,
        "p95_nearest_cluster_to_truth_distance_m": p95_m,
        "fraction_frames_with_cluster_within_5m": 0.0,
        "fraction_frames_with_cluster_within_10m": 0.0,
        "fraction_frames_with_cluster_within_20m": 0.1,
    }


def test_translation_diagnosis_ignores_axis_variant_with_translation() -> None:
    audit = pd.DataFrame(
        [
            _audit_row(
                variant="as-is",
                translation_mode="none",
                mean_m=30.0,
                p95_m=50.0,
            ),
            _audit_row(
                variant="as-is+median-translation",
                translation_mode="per-sequence-median-diagnostic",
                mean_m=25.0,
                p95_m=40.0,
                translation_x_m=10.0,
            ),
            _audit_row(
                variant="swap-xy",
                translation_mode="none",
                mean_m=4.0,
                p95_m=5.0,
            ),
            _audit_row(
                variant="swap-xy+median-translation",
                translation_mode="per-sequence-median-diagnostic",
                mean_m=1.0,
                p95_m=2.0,
                translation_x_m=10.0,
            ),
        ]
    )

    summary = build_sequence_alignment_decision_summary(audit).iloc[0]

    assert summary["after_translation_nearest_mean"] == 25.0
    assert summary["after_translation_nearest_p95"] == 40.0
    assert summary["diagnosis"] == "axis_or_scale_suspected"
