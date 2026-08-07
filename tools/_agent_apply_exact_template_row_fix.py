from __future__ import annotations

from pathlib import Path


MATCHER = '''def _template_time_matches(values: pd.Series, target: float):
    """Match rows copied from the template by exact timestamp identity."""

    numeric = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    return numeric == float(target)
'''


def replace_once(path: str, old: str, new: str) -> None:
    source = Path(path)
    text = source.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise RuntimeError(
            f"expected exactly one replacement anchor in {path}, found {count}: {old!r}"
        )
    source.write_text(text.replace(old, new, 1), encoding="utf-8")


def patch_fixed_weight_ensemble() -> None:
    path = "src/raft_uav/mmuad/track5_estimate_ensemble/__init__.py"
    replace_once(
        path,
        "\n\ndef _normalize_estimate_weight_mapping(",
        f"\n\n{MATCHER}\n\ndef _normalize_estimate_weight_mapping(",
    )
    replace_once(
        path,
        "_IMPL._validate_ensemble_weight = _validate_ensemble_weight",
        "_IMPL._template_time_matches = _template_time_matches\n"
        "_IMPL._validate_ensemble_weight = _validate_ensemble_weight",
    )
    replace_once(
        path,
        'globals()["_validate_ensemble_weight"] = _validate_ensemble_weight',
        'globals()["_template_time_matches"] = _template_time_matches\n'
        'globals()["_validate_ensemble_weight"] = _validate_ensemble_weight',
    )


def patch_uncertainty_ensemble() -> None:
    path = "src/raft_uav/mmuad/track5_uncertainty_ensemble/__init__.py"
    replace_once(
        path,
        "\n\ndef _positive_finite_real_scalar(",
        f"\n\n{MATCHER}\n\ndef _positive_finite_real_scalar(",
    )
    replace_once(
        path,
        "_IMPL._first_present = _first_present",
        "_IMPL._template_time_matches = _template_time_matches\n"
        "_IMPL._first_present = _first_present",
    )
    replace_once(
        path,
        'globals()["_first_present"] = _first_present',
        'globals()["_template_time_matches"] = _template_time_matches\n'
        'globals()["_first_present"] = _first_present',
    )


def patch_consensus_ensemble() -> None:
    path = "src/raft_uav/mmuad/track5_estimate_consensus_ensemble/__init__.py"
    replace_once(
        path,
        "\n\ndef _validate_unique_estimate_labels(",
        f"\n\n{MATCHER}\n\ndef _validate_unique_estimate_labels(",
    )
    replace_once(
        path,
        "_IMPL._first_present = _first_present",
        "_IMPL._template_time_matches = _template_time_matches\n"
        "_IMPL._first_present = _first_present",
    )
    replace_once(
        path,
        'globals()["_normalized_column_name"] = _normalized_column_name',
        'globals()["_template_time_matches"] = _template_time_matches\n'
        'globals()["_normalized_column_name"] = _normalized_column_name',
    )


def patch_spread_guard_ensemble() -> None:
    path = "src/raft_uav/mmuad/track5_estimate_ensemble_spread_guard/__init__.py"
    replace_once(
        path,
        "\n\ndef _materialize_unique_inputs(",
        f"\n\n{MATCHER}\n\ndef _materialize_unique_inputs(",
    )
    replace_once(
        path,
        "_IMPL.build_spread_guarded_estimate_ensemble = "
        "build_spread_guarded_estimate_ensemble",
        "_IMPL._template_time_matches = _template_time_matches\n"
        "_IMPL.build_spread_guarded_estimate_ensemble = "
        "build_spread_guarded_estimate_ensemble",
    )
    replace_once(
        path,
        'globals()["_materialize_unique_inputs"] = _materialize_unique_inputs',
        'globals()["_template_time_matches"] = _template_time_matches\n'
        'globals()["_materialize_unique_inputs"] = _materialize_unique_inputs',
    )


def write_regression_tests() -> None:
    Path("tests/test_mmuad_track5_exact_template_rows.py").write_text(
        '''from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from raft_uav.mmuad.track5_estimate_consensus_ensemble import (
    build_track5_consensus_estimate_ensemble,
)
from raft_uav.mmuad.track5_estimate_ensemble import (
    EstimateInput,
    build_track5_estimate_ensemble,
)
from raft_uav.mmuad.track5_estimate_ensemble_spread_guard import (
    build_spread_guarded_estimate_ensemble,
)
from raft_uav.mmuad.track5_uncertainty_ensemble import (
    build_track5_uncertainty_ensemble,
)


_CLOSE_TIMES = [0.0, 0.5e-9]


def _template() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Sequence": ["seq_close", "seq_close"],
            "Timestamp": _CLOSE_TIMES,
            "Position": ["(0,0,0)", "(0,0,0)"],
            "Classification": [1, 1],
        }
    )


def _estimates(*, with_sigma: bool = False) -> pd.DataFrame:
    rows = pd.DataFrame(
        {
            "sequence_id": ["seq_close", "seq_close"],
            "time_s": _CLOSE_TIMES,
            "state_x_m": [0.0, 100.0],
            "state_y_m": [0.0, 0.0],
            "state_z_m": [1.0, 1.0],
        }
    )
    if with_sigma:
        rows["predicted_sigma_m"] = 1.0
    return rows


def _assert_independent_rows(
    estimates: pd.DataFrame,
    diagnostics: pd.DataFrame,
    *,
    estimate_count_column: str,
    diagnostic_count_column: str,
) -> None:
    assert estimates["time_s"].tolist() == _CLOSE_TIMES
    assert estimates["state_x_m"].tolist() == pytest.approx([0.0, 100.0])
    assert estimates[estimate_count_column].tolist() == [1, 1]
    assert diagnostics[diagnostic_count_column].tolist() == [1, 1]


def test_fixed_weight_ensemble_keeps_close_template_rows_independent() -> None:
    estimates, diagnostics = build_track5_estimate_ensemble(
        [("only", _estimates(), 1.0)],
        _template(),
        max_nearest_time_delta_s=0.0,
    )

    _assert_independent_rows(
        estimates,
        diagnostics,
        estimate_count_column="ensemble_source_count",
        diagnostic_count_column="candidate_input_count",
    )


def test_uncertainty_ensemble_keeps_close_template_rows_independent(
    tmp_path: Path,
) -> None:
    estimate_csv = tmp_path / "estimate.csv"
    _estimates(with_sigma=True).to_csv(estimate_csv, index=False)

    estimates, diagnostics = build_track5_uncertainty_ensemble(
        [EstimateInput("only", estimate_csv, 1.0)],
        template=_template(),
        max_nearest_time_delta_s=0.0,
    )

    _assert_independent_rows(
        estimates,
        diagnostics,
        estimate_count_column="ensemble_source_count",
        diagnostic_count_column="candidate_input_count",
    )


def test_consensus_ensemble_keeps_close_template_rows_independent() -> None:
    estimates, diagnostics = build_track5_consensus_estimate_ensemble(
        [("only", _estimates(), 1.0)],
        _template(),
        consensus_radius_m=1.0,
        max_nearest_time_delta_s=0.0,
    )

    _assert_independent_rows(
        estimates,
        diagnostics,
        estimate_count_column="consensus_input_count",
        diagnostic_count_column="valid_input_count",
    )


def test_spread_guard_keeps_close_template_rows_independent() -> None:
    estimates, diagnostics = build_spread_guarded_estimate_ensemble(
        [("only", _estimates(), 1.0)],
        _template(),
        spread_threshold_m=1_000.0,
        max_nearest_time_delta_s=0.0,
    )

    assert estimates["time_s"].tolist() == _CLOSE_TIMES
    assert estimates["state_x_m"].tolist() == pytest.approx([0.0, 100.0])
    assert diagnostics["valid_input_count"].tolist() == [1, 1]
''',
        encoding="utf-8",
    )


def remove_temporary_files() -> None:
    for path in (
        ".github/workflows/agent-exact-template-row-fix.yml",
        ".github/workflows/agent-exact-template-row-fix-pr.yml",
        "tools/_agent_apply_exact_template_row_fix.py",
    ):
        Path(path).unlink()


def main() -> None:
    patch_fixed_weight_ensemble()
    patch_uncertainty_ensemble()
    patch_consensus_ensemble()
    patch_spread_guard_ensemble()
    write_regression_tests()
    remove_temporary_files()


if __name__ == "__main__":
    main()
