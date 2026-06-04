# PyRecEst measurement reliability migration

RaFT-UAV-specific RF reliability scoring should remain in RaFT-UAV.  The generic
operation that turns a reliability score into covariance inflation or hard
accept/reject decisions now lives in PyRecEst.

Use the adapter:

```python
from raft_uav.baselines.pyrecest_measurement_reliability import (
    apply_raft_measurement_reliability,
)

result = apply_raft_measurement_reliability(
    rf_covariance,
    reliability=rf_probability,
    mode="soft",              # off | soft | hard
    threshold=0.45,
    min_probability=0.05,
)

if result.accepted:
    covariance = result.covariance
else:
    # skip / coast / reject according to the caller's event accounting policy
    ...
```

Mapping:

- RaFT-UAV `off` -> PyRecEst `off`
- RaFT-UAV `soft` -> PyRecEst `inflate`
- RaFT-UAV `hard` -> PyRecEst `hard`

The RF score itself can still use local RF/RHO/CEP/spatial-density/radar-support
features.  The covariance scaling and hard reliability gate should use the
PyRecEst-backed adapter.
