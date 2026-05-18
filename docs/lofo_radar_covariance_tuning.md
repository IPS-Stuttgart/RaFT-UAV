# LOFO radar covariance tuning

`run_lofo_radar_covariance_tuning.py` performs leave-one-flight-out tuning of the
range-angle radar covariance parameters used by the existing runtime covariance
hook.

For each held-out flight, the script:

1. enumerates a grid of covariance hyperparameters,
2. runs `scripts/run_tracklet_viterbi_baseline.py` on all non-held-out flights,
3. chooses the candidate minimizing the requested training metric, and
4. evaluates the chosen candidate on the held-out flight.

Example:

```bash
python scripts/run_lofo_radar_covariance_tuning.py data/raw/AADM2025Dryad \
  --flight Opt1 --flight Opt2 --flight Opt3 \
  --metric position_error_3d.rmse_m \
  --range-std-m 3,5,10,20 \
  --azimuth-std-deg 1,2,3,4 \
  --elevation-std-deg 1,2,3,4 \
  --max-std-m 150,250,400 \
  --skip-existing
```

Useful outputs:

- `lofo_radar_covariance_summary.csv`
- `lofo_radar_covariance_all_training_rows.csv`
- `<holdout>/training_covariance_sweep.csv`
- `<holdout>/selected_covariance.json`

The script sets these environment variables for each candidate run:

- `RAFT_UAV_RADAR_COVARIANCE_MODE=range-angle`
- `RAFT_UAV_RADAR_RANGE_STD_M`
- `RAFT_UAV_RADAR_AZIMUTH_STD_DEG`
- `RAFT_UAV_RADAR_ELEVATION_STD_DEG`
- `RAFT_UAV_RADAR_COVARIANCE_MIN_STD_M`
- `RAFT_UAV_RADAR_COVARIANCE_MAX_STD_M`

Use `--baseline-arg` repeatedly to forward additional arguments to the tracklet
baseline, for example:

```bash
python scripts/run_lofo_radar_covariance_tuning.py data/raw/AADM2025Dryad \
  --baseline-arg --smoother --baseline-arg fixed-lag \
  --baseline-arg --smoother-lag-s --baseline-arg 20
```
