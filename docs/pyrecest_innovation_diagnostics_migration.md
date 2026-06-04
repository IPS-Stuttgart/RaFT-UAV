# PyRecEst innovation diagnostics migration

Generic normalized-innovation-squared, chi-square gate, residual, and summary
diagnostics live in `pyrecest.tracking`. RaFT-UAV should keep only thin
source-specific adapters for RF/radar naming and CSV/report compatibility.

Use this adapter when a RaFT-UAV diagnostic needs NIS/residual statistics:

```python
from raft_uav.baselines.pyrecest_innovation_diagnostics import (
    raft_linear_innovation_diagnostic,
)

diag = raft_linear_innovation_diagnostic(
    mean=x,
    covariance_matrix=P,
    measurement_vector=z,
    observation_matrix=H,
    measurement_covariance=R,
    gate_threshold=gate,
    source="radar",
)
```

The main Kalman robust-update path can continue to use PyRecEst's robust update
planning; this adapter is for standalone diagnostics, replay scoring, and CSV
summaries.
