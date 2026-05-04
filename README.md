# RaFT-UAV

Radar-RF Fusion Tracking for UAVs.

This repository contains implementation and evaluation code for tracking UAVs
with AERPAW Dataset-28 / Dryad DOI `10.5061/dryad.7d7wm3898`. The initial
baseline is an asynchronous constant-velocity Kalman fusion tracker built on
[PyRecEst](https://github.com/FlorianPfaff/PyRecEst).

Large datasets and generated bulk artifacts are intentionally not stored in
this repository.

## Setup

```bash
python -m pip install -e ".[dev]"
```

## Data Layout

Download and extract the Dryad archive outside git, for example:

```text
data/raw/AADM2025Dryad/
  RF Sensor and Radar/
    <flight>/
      AADM*.csv
      radar_data_*.json
      date_time_vehicleOut.txt
```

The loader starts with the RF Sensor and Radar folder because it contains the
modalities needed for the tracking work.

## First Commands

Inspect available flights:

```bash
python -m raft_uav.cli inspect data/raw/AADM2025Dryad
```

Run the initial fusion baseline on one flight:

```bash
python -m raft_uav.cli run-baseline data/raw/AADM2025Dryad --flight Opt2
```

The first baseline is deliberately conservative. It is meant to reproduce the
published constant-velocity Kalman fusion setup before adding robust gating,
learned sensor uncertainties, maneuvering models, and smoothing.
