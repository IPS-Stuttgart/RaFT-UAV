from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pandas as pd


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
MODULE_PATH = SCRIPTS_DIR / "mmuad_snap_official_results_to_template.py"
spec = importlib.util.spec_from_file_location(
    "mmuad_snap_official_results_to_template_empty_inputs",
    MODULE_PATH,
)
assert spec is not None and spec.loader is not None
snapper = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = snapper
spec.loader.exec_module(snapper)


def test_template_snap_zero_fills_schema_valid_header_only_results() -> None:
    results = pd.DataFrame(columns=["Sequence", "Timestamp", "Position", "Classification"])
    template = pd.DataFrame({"Sequence": ["seq001"], "Timestamp": [12.5]})

    snapped, diagnostics = snapper.snap_official_results_to_template(results, template)

    row = snapped.iloc[0]
    assert row["Sequence"] == "seq001"
    assert float(row["Timestamp"]) == 12.5
    assert row["Position"] == "(0,0,0)"
    assert int(row["Classification"]) == 0

    diagnostic = diagnostics.iloc[0]
    assert int(diagnostic["source_row_count"]) == 0
    assert diagnostic["method"] == "missing-zero"
    assert bool(diagnostic["extrapolated"]) is True
    assert bool(diagnostic["valid"]) is False


def test_template_snap_keeps_diagnostic_columns_without_template_rows() -> None:
    results = pd.DataFrame(
        {
            "Sequence": ["seq001"],
            "Timestamp": [0.0],
            "Position": ["(1,2,3)"],
            "Classification": [2],
        }
    )
    template = pd.DataFrame({"Sequence": [], "Timestamp": []})

    snapped, diagnostics = snapper.snap_official_results_to_template(results, template)

    assert snapped.empty
    assert diagnostics.empty
    assert list(snapped.columns) == list(snapper.OFFICIAL_UG2_RESULT_COLUMNS)
    assert list(diagnostics.columns) == list(snapper.DIAGNOSTIC_COLUMNS)
