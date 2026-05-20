# Heteroscedastic RF/radar uncertainty

This adds a dependency-light first version of learned, row-wise sensor covariance for the RF/radar fusion baseline.

## Train

Use leave-one-flight-out training when evaluating a held-out flight:

```bash
python scripts/train_heteroscedastic_uncertainty.py data/raw/AADM2025Dryad \
  --exclude-flight Opt3 \
  --output outputs/uncertainty/no_opt3_uncertainty.json
```

The trainer aligns normalized RF and radar rows with the nearest truth sample and fits ridge-regularized log-linear variance heads for each measured axis.

## Evaluate

```bash
python scripts/run_heteroscedastic_baseline.py data/raw/AADM2025Dryad \
  --flight Opt3 \
  --uncertainty-model outputs/uncertainty/no_opt3_uncertainty.json \
  --radar-selection catprob \
  --smoother fixed-lag
```

The runner keeps the existing constant-velocity Kalman baseline and replaces static measurement covariance with per-row covariance columns predicted by the model.

## Leakage-safe SOTA row

The leave-one-flight-out SOTA runner can make heteroscedastic covariance the
main calibrated benchmark row. For each held-out flight it:

1. fits the heteroscedastic RF/radar uncertainty model on the training flights,
2. replays those training flights once to collect uncalibrated update NIS diagnostics,
3. fits source/dimension-specific NIS covariance multipliers from the training diagnostics, and
4. evaluates the held-out flight with `RAFT_UAV_NIS_COVARIANCE_CALIBRATION_JSON` set to the fold-specific calibration JSON.

```bash
python scripts/run_leave_flight_out_sota.py data/raw/AADM2025Dryad \
  --methods hetero_cv_lofo_nis_fixed_lag
```

## Leave-one-flight-out tracklet-Viterbi benchmark row

For the main benchmark table, prefer the leakage-safe LOFO row that combines
trained heteroscedastic covariance with the current strongest association/replay
path:

```bash
python scripts/run_leave_flight_out_sota.py data/raw/AADM2025Dryad \
  --methods hetero_imm_tracklet_viterbi_fixed_lag
```

The SOTA runner trains `outputs/leave_flight_out_sota/heldout_*/models/heteroscedastic_uncertainty.json`
on all non-held-out flights, then evaluates the held-out flight with
`raft_uav.heteroscedastic_tracklet_viterbi_cli`, `tracklet-viterbi`, the
`range-covariance` variant, IMM replay, robust updates, and fixed-lag smoothing.

The lower-level command is also exposed directly:

```bash
raft-uav-heteroscedastic-tracklet-viterbi run-baseline data/raw/AADM2025Dryad \
  --flight Opt3 --uncertainty-model outputs/uncertainty/no_opt3_uncertainty.json \
  --radar-association tracklet-viterbi --tracklet-variant range-covariance \
  --tracklet-replay-tracker imm --robust-update student-t --smoother fixed-lag
```

## Programmatic measurement conversion

For experiments that already normalize RF/radar rows and want to consume learned `cov_*` columns directly, use:

```python
from raft_uav.heteroscedastic_measurements import (
    radar_measurements_to_enu_with_uncertainty,
    rf_measurements_to_enu_with_uncertainty,
)

rf = model.apply_rf(rf)
radar = model.apply_radar(radar)
rf_measurements = rf_measurements_to_enu_with_uncertainty(rf)
radar_measurements = radar_measurements_to_enu_with_uncertainty(radar)
```

The radar converter keeps six-dimensional position-plus-velocity measurements when Fortem velocity components are present.  The learned covariance is used for the position block and the historical fixed velocity covariance is retained for the velocity block.

## Model outputs

RF rows receive:

- `cov_ee`
- `cov_nn`
- `cov_en`
- `std_east_m`
- `std_north_m`

Radar rows receive:

- `cov_ee`
- `cov_nn`
- `cov_uu`
- `cov_en`
- `cov_eu`
- `cov_nu`
- `std_east_m`
- `std_north_m`
- `std_up_m`

`covariance_from_row(...)` prefers `association_cov_*` columns over `cov_*` columns so PDA/MHT association code can override the measurement covariance with a mixture covariance.

## Notes

This implementation intentionally avoids adding scikit-learn or a neural dependency. It is meant as a safe first step: fit calibrated covariance, compare against the fixed-covariance CV baseline, and then use the same row-wise covariance in PDA/MHT/IMM extensions.
