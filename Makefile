PYTHON ?= python

audit-data:
	$(PYTHON) -m forecast_select audit-data

build-model:
	$(PYTHON) -m forecast_select build-model

show-results:
	$(PYTHON) -m forecast_select show-results

check-project:
	$(PYTHON) -m forecast_select check-project

test:
	$(PYTHON) -m pytest
