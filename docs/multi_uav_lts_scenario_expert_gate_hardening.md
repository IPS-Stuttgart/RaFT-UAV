# Scenario expert gate evidence hardening

The scenario-level Multi-UAV LTS selector keeps raw predictions as an explicit
fallback and chooses a complete prediction source for each sequence prefix. The
hardened public path is:

```text
raft_uav.multi_uav_lts.scenario_expert_gate
```

It delegates to the maintained implementation and adds four safeguards.

## Direct compatibility with the RaFT-UAV metric CSV

The `metrics --sequence-summary-csv` output names the organizer HOTA value
`hota_at_005`. The scenario gate accepts that field directly, in addition to the
older `CODABENCH_HOTA` aliases.

## Family-wise prefix evidence

For every non-raw expert and prefix, the fitter bootstraps the sequence-level
HOTA-at-0.05 gain and Bonferroni-adjusts the interval across all transformed
experts. A candidate is eligible only when the adjusted lower bound clears
`--min-train-hota-ci-low`.

The cross-validated mixed policy receives a separate paired bootstrap interval.
It falls back globally to raw when that lower bound fails
`--min-cv-hota-ci-low`.

Useful controls are:

```text
--bootstrap-samples 5000
--familywise-alpha 0.05
--min-train-hota-ci-low 0.0
--min-cv-hota-ci-low 0.0
```

## Strict controls

Boolean, non-scalar, complex, NaN, and infinite programmatic controls are
rejected instead of being silently interpreted as integers or thresholds.

## Atomic prediction publication

Mixed predictions are first generated in a temporary sibling directory. The
selected output replaces an existing destination only after every candidate
manifest and every copied sequence has succeeded. Output paths must also be
disjoint from candidate inputs in both ancestry directions.

The final mixed training directory should still be compared against its exact
raw source with `raft_uav.multi_uav_lts.tournament`. The confidence gates reduce
selection risk; they do not establish a hidden-test or leaderboard gain.
