UV ?= uv
ARCHIVE ?= $(CURDIR)/pinnacle.zip
UV_CACHE_DIR ?= $(CURDIR)/.uv-cache
UV_PROJECT_ENVIRONMENT ?= $(CURDIR)/.venv
PYTHON := $(UV_PROJECT_ENVIRONMENT)/bin/python
ENV_STAMP := $(UV_PROJECT_ENVIRONMENT)/pyvenv.cfg
CPD_WORKERS ?= $(shell nproc | awk '{print ($$1 > 2 ? $$1 - 2 : 1)}')
DMN_FEATURE_WORKERS ?= $(shell nproc | awk '{print ($$1 > 2 ? $$1 - 2 : 1)}')
DMN_MEMORY_WORKERS := $(shell awk '/MemAvailable/ {n=int($$2/2621440); print (n > 0 ? n : 1); exit}' /proc/meminfo 2>/dev/null || echo 1)
DMN_TRAIN_WORKERS ?= $(shell cpu=$$(nproc | awk '{print ($$1 > 2 ? $$1 - 2 : 1)}'); mem=$(DMN_MEMORY_WORKERS); test $$mem -lt $$cpu && echo $$mem || echo $$cpu)
DMN_ENSEMBLE ?= 5
DMN_GATE_MEMBERS ?= 5

export UV_CACHE_DIR
export UV_PROJECT_ENVIRONMENT
export KERAS_BACKEND := torch
export OMP_NUM_THREADS := 1
export OPENBLAS_NUM_THREADS := 1
export MKL_NUM_THREADS := 1

.PHONY: help env unit-test integration-test test changepoint-test deep-momentum-test \
	changepoint-smoke changepoint-prepare-full changepoint-simulation-full \
	changepoint-nu-full changepoint-events-full changepoint-finalize-full \
	changepoint-full changepoint-clean deep-momentum-smoke \
	deep-momentum-features-full deep-momentum-train-full \
	deep-momentum-replicate-full deep-momentum-report-full \
	deep-momentum-gate-full deep-momentum-bootstrap-full \
	deep-momentum-validate-full deep-momentum-full

help:
	@printf '%s\n' \
		'make env                    Synchronise the locked CPython 3.12 environment.' \
		'make test                   Run all unit and bounded integration tests.' \
		'make changepoint-smoke      Run the complete bounded BOCPD profile.' \
		'make deep-momentum-smoke    Run the complete bounded DMN and gating profile.' \
		'make changepoint-full       Run all dissertation BOCPD experiments.' \
		'make deep-momentum-full     Run all dissertation DMN and gating experiments.' \
		'Pass ARCHIVE=/absolute/path/to/pinnacle.zip to experiment targets.'

env: $(ENV_STAMP)

$(ENV_STAMP): pyproject.toml uv.lock
	$(UV) sync --locked --project "$(CURDIR)"

changepoint-test: | env
	$(PYTHON) -m unittest tests.test_changepoint_detection -v

deep-momentum-test: | env
	$(PYTHON) -m unittest tests.test_deep_momentum -v

unit-test: changepoint-test deep-momentum-test

integration-test: | env
	$(PYTHON) -m unittest tests.test_integration -v

test: unit-test integration-test

changepoint-smoke: changepoint-test | env
	@test -f "$(ARCHIVE)" || { printf '%s\n' 'Prepared archive not found: $(ARCHIVE)'; exit 2; }
	$(PYTHON) -m changepoint_detection smoke --archive "$(ARCHIVE)" --workers "$(CPD_WORKERS)"

changepoint-prepare-full: | env
	@test -f "$(ARCHIVE)" || { printf '%s\n' 'Prepared archive not found: $(ARCHIVE)'; exit 2; }
	$(PYTHON) -m changepoint_detection prepare-real-data --profile full --archive "$(ARCHIVE)" --workers "$(CPD_WORKERS)"

changepoint-simulation-full: changepoint-prepare-full | env
	$(PYTHON) -m changepoint_detection simulation --profile full --archive "$(ARCHIVE)" --workers "$(CPD_WORKERS)"

changepoint-nu-full: changepoint-prepare-full | env
	$(PYTHON) -m changepoint_detection select-nu --profile full --archive "$(ARCHIVE)" --workers "$(CPD_WORKERS)"

changepoint-events-full: changepoint-nu-full | env
	$(PYTHON) -m changepoint_detection events --profile full --archive "$(ARCHIVE)" --workers "$(CPD_WORKERS)"

changepoint-finalize-full: | env
	$(PYTHON) -m changepoint_detection finalize --profile full --archive "$(ARCHIVE)" --workers "$(CPD_WORKERS)"

changepoint-full: changepoint-simulation-full changepoint-events-full | env
	$(PYTHON) -m changepoint_detection finalize --profile full --archive "$(ARCHIVE)" --workers "$(CPD_WORKERS)"

changepoint-clean: | env
	$(PYTHON) -m changepoint_detection clean

deep-momentum-smoke: deep-momentum-test | env
	@test -f "$(ARCHIVE)" || { printf '%s\n' 'Prepared archive not found: $(ARCHIVE)'; exit 2; }
	$(PYTHON) -m deep_momentum smoke --archive "$(ARCHIVE)" --workers "$(DMN_FEATURE_WORKERS)" --train-workers "$(DMN_TRAIN_WORKERS)"

deep-momentum-features-full: deep-momentum-test | env
	@test -f "$(ARCHIVE)" || { printf '%s\n' 'Prepared archive not found: $(ARCHIVE)'; exit 2; }
	$(PYTHON) -m deep_momentum features --profile full --archive "$(ARCHIVE)" --workers "$(DMN_FEATURE_WORKERS)" --train-workers "$(DMN_TRAIN_WORKERS)"

deep-momentum-train-full: deep-momentum-features-full | env
	$(PYTHON) -m deep_momentum train --profile full --archive "$(ARCHIVE)" --workers "$(DMN_FEATURE_WORKERS)" --train-workers "$(DMN_TRAIN_WORKERS)"

deep-momentum-replicate-full: deep-momentum-train-full | env
	$(PYTHON) -m deep_momentum replicate --profile full --archive "$(ARCHIVE)" --workers "$(DMN_FEATURE_WORKERS)" --train-workers "$(DMN_TRAIN_WORKERS)"

deep-momentum-report-full: deep-momentum-replicate-full | env
	$(PYTHON) -m deep_momentum report --profile full --ensemble-size "$(DMN_ENSEMBLE)" --archive "$(ARCHIVE)" --workers "$(DMN_FEATURE_WORKERS)" --train-workers "$(DMN_TRAIN_WORKERS)"

deep-momentum-gate-full: deep-momentum-report-full | env
	$(PYTHON) -m deep_momentum gate --profile full --gate-members "$(DMN_GATE_MEMBERS)" --archive "$(ARCHIVE)" --workers "$(DMN_FEATURE_WORKERS)" --train-workers "$(DMN_TRAIN_WORKERS)"
	$(PYTHON) -m deep_momentum gate-by-class --profile full --gate-members "$(DMN_GATE_MEMBERS)" --archive "$(ARCHIVE)" --workers "$(DMN_FEATURE_WORKERS)" --train-workers "$(DMN_TRAIN_WORKERS)"

deep-momentum-bootstrap-full: deep-momentum-gate-full | env
	$(PYTHON) -m deep_momentum bootstrap --profile full --archive "$(ARCHIVE)" --workers "$(DMN_FEATURE_WORKERS)" --train-workers "$(DMN_TRAIN_WORKERS)"

deep-momentum-validate-full: deep-momentum-bootstrap-full | env
	$(PYTHON) -m deep_momentum validate --profile full --ensemble-size "$(DMN_ENSEMBLE)" --gate-members "$(DMN_GATE_MEMBERS)" --archive "$(ARCHIVE)" --workers "$(DMN_FEATURE_WORKERS)" --train-workers "$(DMN_TRAIN_WORKERS)"

deep-momentum-full: deep-momentum-validate-full

.DEFAULT_GOAL := help
