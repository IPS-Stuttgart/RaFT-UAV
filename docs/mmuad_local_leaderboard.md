# MMUAD local leaderboard workflow

This workflow aggregates one or more `mmaud_results.csv` or Codabench-style ZIP files
with the repository-local MMUAD evaluator.  It is intended for reproducible
method comparisons before using the official UG2+/Codabench runtime.

The generated table is **not** a claim of closed Codabench evaluator parity.
Every row records the metric protocol and leaderboard-readiness flags from the
local evaluator.

## Config file

JSON/YAML example:

```json
{
  "default_truth": "truth.csv",
  "default_metric_protocol": "public-track5",
  "methods": [
    {
      "method": "baseline",
      "results_csv": "baseline_mmaud_results.csv",
      "source_note": "exported baseline trajectory"
    },
    {
      "method": "raft_uav_pp",
      "results_zip": "raft_uav_pp_submission.zip",
      "source_note": "RaFT-UAV++ tracking backend"
    }
  ]
}
```

CSV example:

```csv
method,results_csv,truth_csv,metric_protocol,source_note
baseline,baseline_mmaud_results.csv,truth.csv,public-track5,exported baseline
raft_uav_pp,raft_uav_pp_mmaud_results.csv,truth.csv,public-track5,RaFT-UAV++
```

Paths are resolved relative to the config file location.

## Run

```bash
PYTHONPATH=src python scripts/build_mmuad_local_leaderboard.py \
  --config data/mmuad_export/leaderboard.json \
  --output-dir outputs/mmuad_local_leaderboard
```

Outputs:

- `mmuad_local_leaderboard.csv`
- `mmuad_local_leaderboard.json`
- `mmuad_local_leaderboard.md`

## Ranking

By default rows are ranked by `pose_mse_loss_m2`, then `p95_3d_m`, then
`max_3d_m`, then `uav_type_accuracy` descending, and finally method name.  A
custom ranking metric can be selected:

```bash
PYTHONPATH=src python scripts/build_mmuad_local_leaderboard.py \
  --config data/mmuad_export/leaderboard.json \
  --output-dir outputs/mmuad_local_leaderboard \
  --rank-metric mean_3d_m
```

## Remaining gaps

This workflow does not replace the official challenge evaluator.  Full support
still requires validating the exact Codabench metric semantics, the native
MMUAD sensor/calibration layout, and a real UAV-type classification pipeline.
