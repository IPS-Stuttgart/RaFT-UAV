from __future__ import annotations

import pandas as pd
import pytest

from raft_uav.mmuad.track5_estimate_ensemble import build_track5_estimate_ensemble


def test_ensemble_template_sequence_cells_are_officially_normalized() -> None:
    template = pd.DataFrame({'Sequence': [' seq0001 '], 'Timestamp': [1.0]})
    estimates = pd.DataFrame(
        {
            'sequence_id': ['seq0001', 'seq0001'],
            'time_s': [0.0, 2.0],
            'state_x_m': [0.0, 2.0],
            'state_y_m': [0.0, 0.0],
            'state_z_m': [5.0, 5.0],
        }
    )

    ensemble, diagnostics = build_track5_estimate_ensemble(
        [('base', estimates, 1.0)],
        template,
    )

    row = ensemble.iloc[0]
    assert row['sequence_id'] == 'seq0001'
    assert row['ensemble_source_count'] == 1
    assert row['state_x_m'] == pytest.approx(1.0)
    assert row['state_y_m'] == pytest.approx(0.0)
    assert row['state_z_m'] == pytest.approx(5.0)
    assert diagnostics.iloc[0]['valid_input_count'] == 1
