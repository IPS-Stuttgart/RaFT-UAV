from __future__ import annotations

from raft_uav.mmuad.track5_estimate_ensemble_apply_weights import _validate_trim_fraction


def test_null_trim_fraction_defaults_to_existing_apply_weights_default() -> None:
    assert _validate_trim_fraction(None) == 0.2
