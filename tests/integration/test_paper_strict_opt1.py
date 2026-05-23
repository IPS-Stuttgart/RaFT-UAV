from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import pytest

from raft_uav.diagnostics.paper_strict import (
    PAPER_REFERENCE_COUNTS,
    PAPER_REFERENCE_ERROR_M,
    run_paper_strict_reproduction,
)


pytestmark = pytest.mark.integration


def test_paper_strict_opt1_matches_reference_fingerprint(tmp_path: Path) -> None:
    """Dataset-backed Table-II parity check.

    This test is skipped in normal CI because the Dryad archive is not part of
    the repository.  Enable it locally with:

      RAFT_UAV_DATA_ROOT=/path/to/AADM2025Dryad \
      RAFT_UAV_ORIGINS_FILE=config/origins.toml \
      pytest tests/integration/test_paper_strict_opt1.py
    """

    dataset_root = os.environ.get("RAFT_UAV_DATA_ROOT")
    origin_config = os.environ.get("RAFT_UAV_ORIGINS_FILE")
    if not dataset_root or not origin_config:
        pytest.skip("set RAFT_UAV_DATA_ROOT and RAFT_UAV_ORIGINS_FILE to run paper parity")

    result = run_paper_strict_reproduction(
        dataset_root=Path(dataset_root),
        flights=["Opt1"],
        output_dir=tmp_path,
        count_mismatch_action="fail",
        origin_config=Path(origin_config),
        variant=os.environ.get("RAFT_UAV_PAPER_VARIANT", "rerun"),
    )
    summary = pd.read_csv(result["summary_csv"])
    parity = pd.read_csv(result["parity_summary_csv"])

    by_method = {str(row["method"]): row for _, row in summary.iterrows()}
    for method, reference_count in PAPER_REFERENCE_COUNTS.items():
        assert method in by_method
        assert int(by_method[method]["selected_count"]) == int(reference_count)

    parity_by_method = {str(row["method"]): row for _, row in parity.iterrows()}
    for method in PAPER_REFERENCE_COUNTS:
        assert str(parity_by_method[method]["count_matches_reference"]).lower() == "true"

    # Error tolerances are deliberately tight enough to catch timestamp/origin
    # drift but not so tight that harmless floating-point differences fail the
    # local audit.
    for method, reference in PAPER_REFERENCE_ERROR_M.items():
        assert method in by_method
        row = by_method[method]
        assert float(row["paper_error_mean_m"]) == pytest.approx(reference["mean"], abs=1.0)
        assert float(row["paper_error_std_m"]) == pytest.approx(reference["std"], abs=1.0)
        assert float(row["paper_error_max_m"]) == pytest.approx(reference["max"], abs=3.0)
