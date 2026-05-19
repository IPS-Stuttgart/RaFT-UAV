# NIS covariance calibration

RaFT-UAV writes one `diagnostics.csv` per `raft-uav run-baseline` flight.  Each row includes the measurement source, measurement dimension, accepted/rejected status, normalized innovation squared (NIS), and any robust-update covariance scale.  This workflow fits source- and dimension-specific measurement covariance multipliers from those diagnostics.

The intended use is leave-flight-out or train/validation calibration: fit the covariance multipliers on training flights, then evaluate a held-out flight with the fitted JSON active.  This prevents tuning RF/radar uncertainty on the same flight used for the reported score.

## 1. Generate training diagnostics

Run the desired baseline or tracklet-Viterbi variant on the training flights, for example:

```bash
raft-uav run-baseline /path/to/dataset \
  --flight FlightA \
  --radar-association tracklet-viterbi \
  --robust-update nis-inflate \
  --output-dir outputs/train_nis
```

Repeat for every training flight.  The calibration command can consume the entire output directory and will discover nested `diagnostics.csv` files.

## 2. Fit covariance multipliers

Mean-NIS matching is the default:

```bash
raft-uav-calibrate-nis-covariance outputs/train_nis \
  --output-json outputs/nis_covariance_calibration.json \
  --output-summary-csv outputs/nis_covariance_calibration.csv \
  --method mean \
  --min-samples 20
```

This fits one multiplier for each `(source, measurement_dim)` group.  For example, `radar:3` matches the mean 3D radar NIS to the chi-square mean of 3.  A multiplier greater than one inflates that source's measurement covariance in subsequent runs; a multiplier below one sharpens it.

A tail-oriented alternative matches a quantile instead:

```bash
raft-uav-calibrate-nis-covariance outputs/train_nis \
  --output-json outputs/nis_covariance_calibration_q95.json \
  --method quantile \
  --quantile 0.95
```

By default only accepted updates are used.  Use `--include-rejected` only for diagnostic stress tests, because hard association failures can dominate the fitted scale.

## 3. Evaluate with the fitted calibration

Activate the calibration through the runtime environment variable printed by the calibration command:

```bash
RAFT_UAV_NIS_COVARIANCE_CALIBRATION_JSON=outputs/nis_covariance_calibration.json \
raft-uav run-baseline /path/to/dataset \
  --flight HeldOutFlight \
  --radar-association tracklet-viterbi \
  --robust-update nis-inflate \
  --output-dir outputs/eval_nis_calibrated
```

The scaling hook is applied when `TrackingMeasurement` objects are constructed, before gating and Kalman/IMM updates.  It therefore affects RF, radar, and any future source that uses the common measurement type, but only for source/dimension groups enabled in the JSON.

## Notes

- `min_samples` protects against high-variance scales from tiny groups; disabled groups are recorded with `applied_scale=1.0`.
- `min_scale` and `max_scale` bound the fitted multiplier to avoid pathological covariance collapse or explosion.
- Fit on training flights and report held-out metrics to avoid optimistic calibration.
