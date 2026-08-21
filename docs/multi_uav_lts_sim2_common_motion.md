# Reliability-gated Sim(2) common-motion experiment

## Scientific hypothesis

The existing proposal graph can subtract a shared translation before evaluating
individual UAV motion. That model is deliberately conservative, but it cannot
represent camera zoom or image-plane rotation. In a dense swarm, a small shared
rotation or scale change can make several locally plausible links look like
individual target maneuvers and can amplify ambiguity at crossings.

This experiment tests a strictly stronger but guarded hypothesis:

> Removing an observable shared similarity transform before delayed proposal
> association improves held-out HOTA and identity metrics relative to both the
> raw tracker and the translation-only common-motion candidate.

The detector, proposal bank, first-frame identities, delayed path cover, learned
edge likelihood, ambiguity beam, birth rules, and output format remain
unchanged. Only the coordinate system used during proposal association changes.

## Method

For every adjacent frame pair, the estimator:

1. keeps at most the highest-confidence 96 proposals per frame;
2. initializes a robust shared translation from the frame medians;
3. forms reciprocal nearest-neighbour proposal pairs using normalized center
   and size costs;
4. removes physically implausible pair steps;
5. fits a translation baseline and a proper-rotation Sim(2) transform

   \[
   p_{t+1}=s_t R(\theta_t)p_t+d_t;
   \]

6. iteratively removes high-residual Sim(2) pairs; and
7. accepts rotation and scale only when all reliability gates pass.

The default gates require:

- at least four reciprocal pairs;
- sufficient normalized spatial spread;
- scale deviation no larger than 0.15;
- absolute rotation no larger than 15 degrees;
- median normalized residual no larger than 1.5; and
- at least 0.05 normalized residual improvement over translation.

If Sim(2) is not observable or does not materially improve the fit, the step
falls back to the robust translation. If translation is also unreliable, the
step is the identity. These fallbacks are part of the candidate definition and
are reported in the output summary.

The accepted frame-to-frame transforms are composed into a map from frame one
to every later frame. Proposals are mapped into the frame-one coordinate system,
the existing experimental proposal graph is run unchanged, and selected output
boxes are mapped back into their native image coordinates.

## Run a first candidate

Use the same proposal bank and association controls as the current delayed/beam
candidate:

```bash
PYTHONPATH=src RAFT_UAV_SKIP_RUNTIME_HOOKS=1 \
python -m raft_uav.multi_uav_lts.sim2_proposal_graph_tracker \
  /path/to/proposals \
  --first-frame-label-dir /path/to/TrainLabels_FirstFrameOnly \
  --output-dir /path/to/sim2_candidate/predictions \
  --output-json /path/to/sim2_candidate/summary.json \
  --enable-delayed-path-cover \
  --enable-ambiguity-beam \
  --delayed-max-gap 0 \
  --delayed-lookahead-frames 2 \
  --delayed-successors-per-frame 3 \
  --delayed-continuation-weight 0.75
```

All normal experimental proposal-graph arguments are forwarded. The wrapper
forces `--no-sequence-cache` because it materializes a run-specific stabilized
proposal directory.

Do not combine this command with:

```text
--enable-common-motion
--common-motion-*
--birth-require-border-entry
--image-width
--image-height
```

Translation common motion would double-correct the proposals. Fixed native
image borders are not valid in the stabilized coordinate system, so
border-gated births and border-gap discounts must remain a separate control.
Ordinary persistent late births remain available.

## Evidence contract

The first complete comparison should contain exactly matched candidates:

1. `raw` seeded BoT-SORT control;
2. delayed/ambiguity proposal graph without common motion;
3. delayed/ambiguity graph with translation-only common motion; and
4. delayed/ambiguity graph after guarded Sim(2) stabilization.

Select no Sim(2) threshold from hidden-test or Codabench feedback. Use the
existing deterministic scenario-stratified folds. Materialize five disjoint
held-out prediction sets, combine them, and compare the complete out-of-fold
candidate against the exact raw control with
`raft_uav.multi_uav_lts.tournament`.

A Sim(2) candidate is eligible only when it satisfies the existing gates:

- positive held-out `CODABENCH_HOTA` gain;
- positive paired-bootstrap lower confidence bound;
- acceptable MOTA and IDF1 changes;
- no excessive worst-prefix regression;
- complete 102-sequence coverage; and
- exact source and prediction provenance.

Report canonical `AssA`, `DetA`, `LocA`, identity switches, output rows, and the
fractions of Sim(2), translation-fallback, and identity steps. A gain confined
to a few sequences with nearly every step falling back is evidence for a
scenario expert, not for making Sim(2) the global default.

## Interpretation

- **Sim(2) improves AssA/IDF1:** shared rotation or zoom was contaminating
  target-relative motion; use the stabilized candidate in the guarded expert
  tournament.
- **Sim(2) improves only localization:** inspect cumulative scale drift and box
  restoration before accepting the result.
- **Translation and Sim(2) tie:** retain the simpler translation model.
- **Sim(2) regresses tree or border-heavy prefixes:** keep it as a
  prefix/feature-gated expert rather than weakening the global evidence gate.
- **Most steps become identity:** proposal correspondences do not support a
  shared motion model; stop tuning Sim(2) and prioritize appearance-assisted
  ambiguity resolution.

The experiment does not claim a leaderboard improvement until a separately
prepared test submission is scored by Codabench.