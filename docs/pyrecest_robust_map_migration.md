# PyRecEst robust MAP smoothing migration

RaFT-UAV's `robust-map` and `fixed-lag-map` modes now delegate the generic
linear-Gaussian trajectory solve to PyRecEst.

## Ownership boundary

PyRecEst owns:

- sparse batch MAP assembly and solution;
- iteratively reweighted robust measurement factors;
- linear, Huber, soft-L1, Cauchy, and arctangent losses;
- monotone line search and convergence diagnostics;
- timestamp-based fixed-lag windowing;
- arbitrary state and measurement dimensions.

RaFT-UAV keeps:

- the six-dimensional ENU constant-velocity model;
- white-acceleration process noise and RaFT-specific regularization floors;
- RF/radar `TrackingMeasurement` conversion;
- global cardinality-first measurement-to-record matching;
- the posterior pseudo-measurement compatibility fallback;
- the existing tracking-record output schema and CLI options.

The public RaFT-UAV entry point remains:

```python
from raft_uav.baselines.robust_map import (
    RobustMapSmootherConfig,
    robust_map_smooth_records,
)
```

Existing callers may continue to choose `robust-map` or `fixed-lag-map` through
`smooth_tracking_records(...)`.

## Covariance semantics

PyRecEst deliberately returns `covariances=None` until selected MAP marginal
covariances are computed from the final approximate Hessian. RaFT-UAV therefore
continues to retain each filtered covariance in its output record and labels the
source with:

```text
map_covariance_source = filtered
```

The record also contains:

```text
map_solver = pyrecest.robust_linear_gaussian_map
```

This prevents filtered uncertainty from being presented as though it were a MAP
smoother marginal.

## Measurement matching

The matcher remains downstream because it depends on RaFT-UAV event semantics.
It solves one global assignment between measurement rows and posterior records,
with the objective ordered by:

1. maximum feasible match count;
2. source-consistent matches;
3. timestamp error;
4. deterministic record-order tie breaking.

When `accepted_measurements_only=True`, rejected records are excluded before the
assignment is solved, so a rejected nearest record cannot consume a measurement
that could match a nearby accepted record.
