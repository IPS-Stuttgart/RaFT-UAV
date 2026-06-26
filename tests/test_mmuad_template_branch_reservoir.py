from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pandas as pd


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
MODULE_PATH = SCRIPTS_DIR / "mmuad_template_branch_reservoir.py"
spec = importlib.util.spec_from_file_location(
    "mmuad_template_branch_reservoir",
    MODULE_PATH,
)
assert spec is not None and spec.loader is not None
template_reservoir = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = template_reservoir
spec.loader.exec_module(template_reservoir)


def _template_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Sequence": ["seq001", "seq001"],
            "Timestamp": [10.0, 20.0],
            "Position": ["(0,0,0)", "(0,0,0)"],
            "Classification": [0, 0],
        }
    )


def _candidate_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["seq001", "seq001", "seq001", "seq001"],
            "time_s": [9.96, 10.04, 10.08, 30.0],
            "source": ["lidar_360", "livox_avia", "radar", "radar"],
            "track_id": ["raw", "translated", "radar_near", "too_late"],
            "candidate_branch": ["raw", "source_translation", "radar", "radar"],
            "x_m": [0.0, 1.0, 2.0, 3.0],
            "y_m": [0.0, 0.0, 0.0, 0.0],
            "z_m": [0.0, 0.0, 0.0, 0.0],
            "ranker_score": [0.2, 0.9, 0.1, 0.99],
            "confidence": [0.2, 0.9, 0.1, 0.99],
        }
    )


def _different_scale_candidate_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["seq001", "seq001", "seq001"],
            "time_s": [9.96, 10.0, 10.04],
            "source": ["lidar_360", "lidar_360", "livox_avia"],
            "track_id": ["raw_high", "raw_low", "translated_high"],
            "candidate_branch": ["raw", "raw", "source_translation"],
            "x_m": [0.0, 10.0, 1.0],
            "y_m": [0.0, 0.0, 0.0],
            "z_m": [0.0, 0.0, 0.0],
            "ranker_score": [100.0, 99.0, 0.2],
            "confidence": [100.0, 99.0, 0.2],
        }
    )


def test_template_reservoir_selects_window_candidates_and_preserves_times() -> None:
    result = template_reservoir.build_template_branch_reservoir(
        _candidate_rows(),
        _template_rows(),
        max_time_delta_s=0.1,
        per_source_top_n=1,
        per_branch_top_n=1,
        global_top_n=1,
    )
    reservoir, frame_summary, branch_summary = result

    assert set(reservoir["track_id"]) == {"raw", "translated", "radar_near"}
    assert "too_late" not in set(reservoir["track_id"])
    assert set(reservoir["time_s"]) == {9.96, 10.04, 10.08}
    assert set(reservoir["template_timestamp_s"]) == {10.0}
    assert int(frame_summary.loc[0, "candidate_count_window"]) == 3
    assert int(frame_summary.loc[0, "reservoir_count"]) == 3
    assert int(frame_summary.loc[1, "candidate_count_window"]) == 0
    translated = branch_summary.loc[
        branch_summary["candidate_branch"] == "source_translation"
    ].iloc[0]
    assert int(translated["reservoir_count"]) == 1


def test_template_reservoir_branch_rank_score_normalization() -> None:
    reservoir, frame_summary, _ = template_reservoir.build_template_branch_reservoir(
        _different_scale_candidate_rows(),
        _template_rows().iloc[[0]],
        max_time_delta_s=0.1,
        per_source_top_n=0,
        per_branch_top_n=1,
        global_top_n=0,
        score_normalization="branch-rank",
    )

    retained = reservoir.set_index("track_id")
    assert set(retained.index) == {"raw_high", "translated_high"}
    assert float(retained.loc["raw_high", "raw_reservoir_score"]) == 100.0
    assert float(retained.loc["translated_high", "raw_reservoir_score"]) == 0.2
    assert float(retained.loc["raw_high", "normalized_reservoir_score"]) == 1.0
    assert float(retained.loc["translated_high", "normalized_reservoir_score"]) == 1.0
    assert set(retained["score_normalization"]) == {"branch-rank"}
    assert set(frame_summary["score_normalization"]) == {"branch-rank"}


def test_template_reservoir_cli_writes_artifacts(tmp_path: Path) -> None:
    template = tmp_path / "template.csv"
    raw = tmp_path / "raw.csv"
    translated = tmp_path / "translated.csv"
    output = tmp_path / "out"
    _template_rows().to_csv(template, index=False)
    rows = _candidate_rows()
    rows.loc[rows["candidate_branch"] == "raw"].to_csv(raw, index=False)
    rows.loc[rows["candidate_branch"] != "raw"].to_csv(translated, index=False)

    rc = template_reservoir.main(
        [
            "--template-csv",
            str(template),
            "--candidate-csv",
            f"raw={raw}",
            "--candidate-csv",
            f"translated={translated}",
            "--output-dir",
            str(output),
            "--max-time-delta-s",
            "0.1",
            "--per-source-top-n",
            "1",
            "--per-branch-top-n",
            "1",
            "--global-top-n",
            "1",
            "--score-normalization",
            "branch-rank",
        ]
    )

    assert rc == 0
    for name in (
        "mmuad_template_branch_reservoir_candidates.csv",
        "mmuad_template_branch_reservoir_frame_summary.csv",
        "mmuad_template_branch_reservoir_branch_summary.csv",
        "mmuad_template_branch_reservoir_provenance.json",
    ):
        assert (output / name).exists()
    reservoir = pd.read_csv(output / "mmuad_template_branch_reservoir_candidates.csv")
    assert "template_timestamp_s" in reservoir.columns
    assert set(reservoir["score_normalization"]) == {"branch-rank"}
    provenance_path = output / "mmuad_template_branch_reservoir_provenance.json"
    provenance = json.loads(provenance_path.read_text())
    assert provenance["template_rows"] == 2
    assert provenance["score_normalization"] == "branch-rank"
    assert provenance["provenance"]["candidate_inputs"][0]["branch"] == "raw"
