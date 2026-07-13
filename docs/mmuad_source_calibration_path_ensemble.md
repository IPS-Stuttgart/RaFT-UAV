# MMUAD source-calibration path ensemble

A train-fitted source transform can improve difficult sequences while degrading
already-good candidates on an unseen sequence. The existing source-calibration
branch union preserves the raw and fully calibrated endpoints. This experiment
adds optional intermediate hypotheses along the raw-to-calibrated coordinate
path:

```text
z(f) = z_raw + f * (z_calibrated - z_raw),  f in [0, 1]
```

The method does not use validation or test truth. Fractions should be selected
on training folds and frozen before public-validation or hidden-test inference.

## Run

```bash
python scripts/mmuad_source_calibration_path_ensemble.py \
  --candidates outputs/mmuad/raw_candidates.csv \
  --mmuad-source-calibration-json outputs/mmuad/source_calibration.json \
  --calibration-fractions 0,0.25,0.5,0.75,1 \
  --output-candidates outputs/mmuad/calibration_path_candidates.csv \
  --summary-json outputs/mmuad/calibration_path_summary.json \
  --reservoir-output-csv outputs/mmuad/calibration_path_reservoir.csv
```

Each derived row retains `mmuad_calibration_origin_row`. Use origin-group
multiplicity correction before pair-state inference so adding more coordinate
representations does not grant one physical observation excess path mass.

## Suggested train-CV ablation

```text
fractions:
  0,1
  0,0.5,1
  0,0.25,0.5,0.75,1

reservoir per-branch quota:
  1,2,3

pair-group correction strength:
  0.5,0.75,1

mixture top-K:
  10,20,40
```

Report full-pool and bounded-reservoir oracle recall together with frozen
learned-sigma Huber mixture-MAP MSE, P95, maximum error, candidate count, and
runtime. The experiment is useful only when intermediate fractions improve
train-CV assignment or preserve a better oracle ceiling without producing an
unmanageable candidate budget.

## Diagnostics

The output records:

- `mmuad_calibration_path_fraction`;
- `mmuad_calibration_path_interpolated`;
- `mmuad_source_calibration_effective_alpha` when the fitted transform exposes
  a source-translation alpha;
- raw-to-branch displacement components and norm;
- shared physical-origin IDs and branch-qualified track IDs.

This complements temporal-support and constant-velocity reservoir quotas: the
path ensemble expands coordinate hypotheses, while those quotas decide which
motion-supported hypotheses survive the bounded reservoir.
