# Dissertation experiment code

This repository provides a self-contained implementation of the reported
experiments. It contains only:

- Bayesian online changepoint detection (BOCPD), including the proposed
  GARCH(1,1) observation models and the reported comparator detectors;
- deep momentum network feature construction, tensor construction, random
  hyperparameter search, seed replication, and ensemble evaluation; and
- the uncertainty-gated overlays, asset-class analysis, and paired circular
  block bootstrap.

Raw vendor-data auditing, reconstruction, repair, cleaning, and archive
creation are deliberately absent. Classical momentum baselines and descriptive
data-appendix calculations are also outside this directory. The numerical
benchmark rows already reported in the dissertation are therefore not
regenerated here.

## Requirements

- CPython 3.12
- [`uv`](https://docs.astral.sh/uv/)
- an already prepared Pinnacle continuous-futures archive, as described below

From the project root, create the exact locked environment:

```sh
make env
```

The Makefile fixes Keras to the PyTorch backend and constrains numerical-library
thread counts so that concurrent training jobs do not oversubscribe the host.
Make targets use `.venv/bin/python`; activating the environment is optional.

## Prepared archive contract

Supply a ZIP archive with one member per contract:

```text
CLCDATA/CC_RAD.CSV
CLCDATA/DA_RAD.CSV
...
CLCDATA/SN_RAD.CSV
```

Every member must be a headerless CSV with the columns
`date,open,high,low,close,vol,oi`. Dates use `MM/DD/YYYY`, rows are strictly
increasing and unique by date, and prices used by the experiments must be
positive and finite. The complete dissertation run expects the ordered
50-contract universe defined in `experiment_data.py`, with observations through
2024-12-31. Observations from 1987-01-01 to 1989-12-31 may be included solely to
initialise lagged indicators. Estimation and evaluation begin on 1990-01-01.

The archive is commercial data and is ignored by Git. The experiment code does
not create, repair, or modify it. Pass its path explicitly:

```sh
make changepoint-smoke ARCHIVE=/absolute/path/to/pinnacle.zip
make deep-momentum-smoke ARCHIVE=/absolute/path/to/pinnacle.zip
```

## Tests

Run all unit and bounded integration tests with:

```sh
make test
```

The integration suite uses deterministic synthetic fixtures; it does not need
the commercial archive. The two `*-smoke` targets are broader acceptance runs
over a supplied prepared archive. They execute complete, reduced experiment
profiles and validate their output schemas and manifests.

Individual suites are available as `make changepoint-test`,
`make deep-momentum-test`, and `make integration-test`.

## BOCPD experiments

The full profile reproduces the reported synthetic and market experiments. It
uses 500 particles, at most 250 retained run lengths, a constant hazard of
0.004, and a 30-day recent-regime window. Synthetic thresholds target an
in-control average run length of 500. The market stage fits per-contract
GARCH(1,1) models, constructs asset-class priors, selects the Student-t degrees
of freedom using pre-2007 data, and evaluates the 2007--2024 event study.

Run the complete experiment:

```sh
make changepoint-full ARCHIVE=/absolute/path/to/pinnacle.zip
```

The same workflow can be resumed stage by stage:

```sh
make changepoint-prepare-full ARCHIVE=/absolute/path/to/pinnacle.zip
make changepoint-simulation-full ARCHIVE=/absolute/path/to/pinnacle.zip
make changepoint-nu-full ARCHIVE=/absolute/path/to/pinnacle.zip
make changepoint-events-full ARCHIVE=/absolute/path/to/pinnacle.zip
make changepoint-finalize-full ARCHIVE=/absolute/path/to/pinnacle.zip
```

`CPD_WORKERS` controls independent simulation-stream and contract jobs. Outputs
are written to `outputs/changepoint/<profile>/`; resumable intermediate arrays
are written to `build/changepoint/<profile>/`.

## Deep momentum and uncertainty gating

List the nine model specifications and their feature sets:

```sh
.venv/bin/python -m deep_momentum models
```

The full profile implements the six five-year out-of-sample windows beginning
in 1995, 2000, 2005, 2010, 2015, and 2020. It runs 50 random-search trials per
model and window, selects configurations on the chronological validation set,
and retrains each selected configuration with 20 deterministic seeds. Primary
reports use the top five validation-ranked members. The model registry contains
the six reported gross-objective models and the three reported 2-basis-point
cost-aware models.

The gating stage forms trial and seed ensembles, estimates model uncertainty
from cross-member dispersion, and uses the estimated uncertainty to decide
when to abstain from trading. It evaluates the reported abstention-rate and
capital-cap grids, applies transaction costs, reports results by asset class and
window, and performs the pre-specified paired circular-block bootstrap.

Run the complete workflow:

```sh
make deep-momentum-full ARCHIVE=/absolute/path/to/pinnacle.zip
```

Or resume it in order:

```sh
make deep-momentum-features-full ARCHIVE=/absolute/path/to/pinnacle.zip
make deep-momentum-train-full ARCHIVE=/absolute/path/to/pinnacle.zip
make deep-momentum-replicate-full ARCHIVE=/absolute/path/to/pinnacle.zip
make deep-momentum-report-full ARCHIVE=/absolute/path/to/pinnacle.zip
make deep-momentum-gate-full ARCHIVE=/absolute/path/to/pinnacle.zip
make deep-momentum-bootstrap-full ARCHIVE=/absolute/path/to/pinnacle.zip
make deep-momentum-validate-full ARCHIVE=/absolute/path/to/pinnacle.zip
```

`DMN_FEATURE_WORKERS` controls independent feature jobs.
`DMN_TRAIN_WORKERS` controls concurrent neural-network fits and is capped by
available memory by default. `DMN_ENSEMBLE` and `DMN_GATE_MEMBERS` default to
five. Caches are written to `build/deep_momentum/<profile>/`, and numerical
outputs and manifests to `outputs/deep_momentum/<profile>/`.

The full runs are computationally intensive. On a many-core host, BOCPD can
require several hours and the neural-network experiment can require longer.
Use the smoke profiles before committing resources to the full profiles.

## Parameters and reproducibility

All experiment parameters are defined in:

- `changepoint_detection/profiles/full.toml` and `smoke.toml`;
- `deep_momentum/profiles/full.toml` and `smoke.toml`;
- model architecture and feature constants in `deep_momentum/config.py`;
- the random-search grid in `deep_momentum/training.py`; and
- the gating grids in `deep_momentum/gating.py`.

Command-line help documents runtime overrides:

```sh
.venv/bin/python -m changepoint_detection --help
.venv/bin/python -m deep_momentum --help
```

Profiles, archive hashes, cache fingerprints, deterministic seeds, selected
hyperparameters, output schemas, and SHA-256 output manifests are recorded so
that completed runs can be audited and resumed without silently mixing
incompatible artefacts.
