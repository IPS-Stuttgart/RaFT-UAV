# Third-wave RaFT-UAV result-improvement scaffolding

This patch adds additive research utilities for the next set of improvement
ideas: factor-graph smoothing, joint association/trajectory refinement,
tracklet-level classification, clutter and class-probability calibration,
native radar covariance modeling, bias modeling, RF quality scaling, flight-phase
segmentation, recovery mode, backward association repair, conformal uncertainty,
domain-shift diagnostics, leakage sentinels, constrained ablation selection,
latency curves, candidate-set recall, association regret, and reproducibility
bundles.

The additions are intentionally lightweight and do not change the default CLI or
baseline behavior. They are meant to be used in ablation scripts first, then
promoted into production paths only after leave-one-flight-out validation.

## New modules

- `raft_uav.research.diagnostics`: candidate-set recall, association regret,
  track-switch stability, domain-shift summaries, latency curves, and leakage
  sentinel checks.
- `raft_uav.research.factor_graph`: a small SciPy least-squares smoother and an
  offline coordinate-descent association/smoothing diagnostic.
- `raft_uav.research.tracklet_models`: tracklet-level features, frame-context
  features, logistic classifiers, Platt calibration, and simple clutter stats.
- `raft_uav.research.measurement_models`: native range/azimuth/elevation
  covariance transformation, linear radar bias correction, and RF-quality
  covariance scaling.
- `raft_uav.research.runtime_modes`: coarse flight-phase segmentation, recovery
  mode decisions, and backward-pass association repair.
- `raft_uav.research.uncertainty`: conformal error radii.
- `raft_uav.research.optimizer`: constrained experiment ranking and Pareto-front
  marking.
- `raft_uav.research.paper_bundle`: reproducibility bundle manifests and README
  generation.

## Useful commands

Candidate-set recall and association regret from normalized CSV artifacts:

```bash
python scripts/run_candidate_recall_regret_report.py \
  --radar outputs/normalized/radar.csv \
  --truth outputs/normalized/truth.csv \
  --selected outputs/run/selected_radar.csv \
  --output-dir outputs/diagnostics/recall_regret
```

Domain-shift report with optional leakage check:

```bash
python scripts/run_domain_shift_report.py \
  --train-csv outputs/train_opt1/radar.csv \
  --train-csv outputs/train_opt2/radar.csv \
  --heldout-csv outputs/heldout_opt3/radar.csv \
  --heldout-flight Opt3 \
  --metadata-json outputs/heldout_opt3/metrics.json \
  --output-csv outputs/diagnostics/domain_shift_opt3.csv
```

Factor-graph smoothing diagnostic:

```bash
python scripts/run_factor_graph_smoother.py \
  --radar outputs/normalized/radar.csv \
  --rf outputs/normalized/rf.csv \
  --output-dir outputs/factor_graph/opt3
```

Constrained ablation ranking:

```bash
python scripts/run_constrained_ablation_optimizer.py \
  outputs/leave_flight_out_sota/fold_summary.csv \
  --output-csv outputs/leave_flight_out_sota/constrained_rank.csv \
  --objective error_3d_rmse_m \
  --constraint truth_coverage_rate:>=:0.95 \
  --constraint track_switch_count:<=:10 \
  --pareto-minimize error_3d_rmse_m error_3d_p95_m \
  --pareto-maximize truth_coverage_rate
```

Paper bundle manifest:

```bash
python scripts/reproduce_paper_results.py data/raw/AADM2025Dryad \
  --output-dir outputs/paper_bundle
```

## Validation discipline

Use these tools to answer three questions before adding more model complexity:

1. Is the target present in the candidate set? Use candidate-set recall.
2. Is the selected row much worse than the best available row? Use association
   regret.
3. Is the remaining error caused by smoothing/filtering rather than association?
   Compare factor-graph diagnostics against current filtered/smoothed estimates.

Any learned classifier, bias model, class-probability calibration, or conformal
radius should be fitted only on training flights and evaluated on held-out
flights. The leakage sentinel is designed to catch obvious mistakes in the
resulting metadata.
