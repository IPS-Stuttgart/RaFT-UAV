from __future__ import annotations

import numpy as np
import pytest

from raft_uav.uncertainty import VarianceHead


@pytest.mark.parametrize("imaginary_part", [0.0, 2.0])
def test_variance_head_rejects_complex_coefficient_arrays(
    imaginary_part: float,
) -> None:
    with pytest.raises(ValueError, match="coefficients must be finite numbers"):
        VarianceHead(
            source="rf",
            dimension="east",
            feature_names=("intercept",),
            coefficients=np.array([1.0 + imaginary_part * 1j]),
            min_std_m=1.0,
            max_std_m=10.0,
            training_rows=1,
        )
