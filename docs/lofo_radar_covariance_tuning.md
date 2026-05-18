# LOFO radar covariance tuning

This script tunes the range-angle radar covariance parameters in a leave-one-flight-out manner.

For each held-out flight, it evaluates a grid on the remaining flights, selects the lowest training metric, then evaluates that setting on the held-out flight.

Main command:

```bash
python scripts/run_lofo_radar_covariance_tuning.py data/raw/AADM2025Dryad --flight Opt1 --flight Opt2 --flight Opt3 --skip-existing
```

Main outputs:

- `lofo_radar_covariance_summary.csv`
- `lofo_radar_covariance_all_training_rows.csv`
- `<holdout>/training_covariance_sweep.csv`
- `<holdout>/selected_covariance.json`

Important options include `--range-std-m`, `--azimuth-std-deg`, `--elevation-std-deg`, `--min-std-m`, `--max-std-m`, `--metric`, and `--aggregate`.
