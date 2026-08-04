PYTHON ?= python

audit-data:
	$(PYTHON) -m forecast_select audit-data

build-model:
	$(PYTHON) -m forecast_select build-model

show-results:
	$(PYTHON) -m forecast_select show-results

build-risk-gate:
	$(PYTHON) -m forecast_select build-risk-gate

show-risk-gate:
	$(PYTHON) -m forecast_select show-risk-gate

directional-downside:
	$(PYTHON) -m forecast_select build-directional-downside

build-context-selector:
	$(PYTHON) -m forecast_select build-context-selector

show-context-selector:
	$(PYTHON) -m forecast_select show-context-selector

forecast-next-three:
	$(PYTHON) -m forecast_select forecast-next-three

check-project:
	$(PYTHON) -m forecast_select check-project

test:
	$(PYTHON) -m pytest
