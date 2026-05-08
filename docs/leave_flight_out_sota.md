# Leave-one-flight-out SOTA evaluation

This protocol evaluates each configured flight as a held-out test fold. Learned
components are trained only on the remaining configured flights, then evaluated
on the held-out flight. This prevents tuning or covariance calibration from
seeing the test trajectory.

## Recommended command

```bash
python scripts/run_leave_flight_out_sota.py data/raw/AADM2025Dryad \
  --methods cv_catprob cv_track_bank_fixed_lag imm_catprob hetero_cv_fixed_lag \
  --candidate-threshold 0.4 \
  --fixed-lag-s 20 \
  --skip-existing
```

By default the script discovers all flights with truth telemetry. To run a
smaller reproducibility subset:

```bash
python scripts/run_leave_flight_out_sota.py data/raw/AADM2025Dryad \
  --flights Opt1 Opt2 Opt3 \
  --methods cv_catprob cv_track_bank_fixed_lag imm_catprob hetero_cv_fixed_lag
```

## Outputs

The default output root is `outputs/leave_flight_out_sota/`.

For each fold:

```text
outputs/leave_flight_out_sota/
  heldout_<flight>/
    models/
      heteroscedastic_uncertainty.json
    <method>/
      <flight>/
        estimates.csv
        metrics.json
        selected_radar.csv
        ...
```

The top-level summaries are:

- `fold_summary.csv`: one row per held-out flight and method.
- `aggregate_summary.csv`: pooled errors and coverage per method.
- `report.json`: machine-readable protocol settings, fold rows, and aggregate rows.

## Metrics

The protocol reports:

- 2D and 3D RMSE, MAE, p50, p90, p95, p99, and max error.
- Truth-time coverage: fraction of truth timestamps with an estimate within the
  configured `--max-eval-time-delta-s` gate.
- NIS calibration diagnostics per source, when NIS is available in the estimate
  artifacts.
- Counts for posterior records, selected radar rows, accepted measurements, and
  rejected measurements.

Aggregate rows pool all per-fold error samples, instead of averaging fold-level
RMSE values. This gives a single pooled leaderboard while still preserving the
per-flight breakdown in `fold_summary.csv`.

## Leakage control

For heteroscedastic uncertainty runs, the script trains one model per held-out
fold by invoking:

```bash
python scripts/train_heteroscedastic_uncertainty.py ... --flight <train_1> --flight <train_2> ...
```

The held-out flight is omitted from that command. The resulting model is stored
inside the fold directory and passed only to the held-out evaluation run.

Classical methods do not train a model, but they are still executed inside the
same fold layout for comparable artifact paths and summaries.
