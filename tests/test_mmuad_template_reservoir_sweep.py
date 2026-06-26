from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pandas as pd


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
MODULE_PATH = SCRIPTS_DIR / "mmuad_template_reservoir_sweep.py"
spec = importlib.util.spec_from_file_location(
    "mmuad_template_reservoir_sweep",
    MODULE_PATH,
)
assert spec is not None and spec.loader is not None
template_sweep = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = template_sweep
spec.loader.exec_module(template_sweep)


def _template_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Sequence": ["seq001", "seq001"],
            "Timestamp": [10.0, 20.0],
            "Position": ["[0,0,0]", "[0,0,0]"],
            "Classification": [0, 0],
        }
    )


def _candidate_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["seq001", "seq001", "seq001"],
            "time_s": [9.96, 10.04, 20.7],
            "source": ["lidar_360", "livox_avia", "radar"],
            "track_id": ["raw", "translated", "fallback"],
            "candidate_branch": ["raw", "source_translation", "radar"],
            "x_m": [0.0, 1.0, 2.0],
            "y_m": [0.0, 0.0, 0.0],
            "z_m": [0.0, 0.0, 0.0],
            "ranker_score": [0.2, 0.9, 0.1],
            "confidence": [0.2, 0.9, 0.1],
        }
    )


def test_template_reservoir_sweep_writes_variant_summaries(tmp_path: Path) -> None:
    summary = template_sweep.run_template_reservoir_sweep(
        candidates=_candidate_rows(),
        template=_template_rows(),
        output_dir=tmp_path,
        max_time_delta_s_values=(0.1, 1.0),
        score_normalizations=("none", "branch-rank"),
        min_candidates_per_template_values=(0,),
        fallback_max_time_delta_s_values=(None,),
        per_source_top_n=1,
        per_branch_top_n=1,
        global_top_n=1,
    )

    assert len(summary) == 4
    assert (tmp_path / "mmuad_template_reservoir_sweep_summary.csv").exists()
    assert (tmp_path / "mmuad_template_reservoir_sweep_summary.json").exists()
    assert set(summary["score_normalization"]) == {"none", "branch-rank"}
    assert set(summary["max_time_delta_s"]) == {0.1, 1.0}
    for reservoir_csv in summary["reservoir_csv"]:
        assert Path(reservoir_csv).exists()


def test_template_reservoir_sweep_min_candidate_fallback_improves_coverage(
    tmp_path: Path,
) -> None:
    summary = template_sweep.run_template_reservoir_sweep(
        candidates=_candidate_rows(),
        template=_template_rows(),
        output_dir=tmp_path,
        max_time_delta_s_values=(0.1,),
        score_normalizations=("none",),
        min_candidates_per_template_values=(0, 1),
        fallback_max_time_delta_s_values=(None,),
        per_source_top_n=1,
        per_branch_top_n=1,
        global_top_n=1,
    )

    no_fallback = summary.loc[summary["min_candidates_per_template"] == 0].iloc[0]
    fallback = summary.loc[summary["min_candidates_per_template"] == 1].iloc[0]
    assert float(no_fallback["coverage_fraction"]) == 0.5
    assert float(fallback["coverage_fraction"]) == 1.0
    assert int(fallback["fallback_rows"]) == 1


def test_template_reservoir_sweep_cli_writes_summary(tmp_path: Path) -> None:
    template = tmp_path / "template.csv"
    raw = tmp_path / "raw.csv"
    translated = tmp_path / "translated.csv"
    output = tmp_path / "out"
    _template_rows().to_csv(template, index=False)
    rows = _candidate_rows()
    rows.loc[rows["candidate_branch"] == "raw"].to_csv(raw, index=False)
    rows.loc[rows["candidate_branch"] != "raw"].to_csv(translated, index=False)

    rc = template_sweep.main(
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
            "0.1,1.0",
            "--score-normalization",
            "none",
            "--score-normalization",
            "branch-rank",
            "--min-candidates-per-template",
            "0,1",
            "--per-source-top-n",
            "1",
            "--per-branch-top-n",
            "1",
            "--global-top-n",
            "1",
        ]
    )

    assert rc == 0
    summary_csv = output / "mmuad_template_reservoir_sweep_summary.csv"
    summary_json = output / "mmuad_template_reservoir_sweep_summary.json"
    assert summary_csv.exists()
    assert summary_json.exists()
    summary = pd.read_csv(summary_csv)
    assert len(summary) == 8
    payload = json.loads(summary_json.read_text())
    assert len(payload["rows"]) == 8
