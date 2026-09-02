PYTHON ?= python

install:
	$(PYTHON) -m pip install -e ".[dev]"

audit-data:
	$(PYTHON) -m forecast_select audit-data

build-model:
	$(PYTHON) -m forecast_select build-model

forecast:
	$(PYTHON) -m forecast_select forecast-next-three

show-results:
	$(PYTHON) -m forecast_select show-results

check-project:
	$(PYTHON) -m forecast_select check-project

lint:
	$(PYTHON) -m ruff check src tests

test:
	$(PYTHON) -m pytest

check: lint test
