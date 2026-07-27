PYTHON ?= python

audit:
	$(PYTHON) -m forecast_select audit
test:
	$(PYTHON) -m pytest
test-leakage:
	$(PYTHON) -m pytest tests/leakage tests/unit/test_validation.py
baselines:
	$(PYTHON) -m forecast_select backtest --models baselines
classical-backtest:
	$(PYTHON) -m forecast_select backtest --models classical
catboost-backtest:
	$(PYTHON) -m forecast_select backtest --models catboost
ensemble-backtest:
	$(PYTHON) -m forecast_select ensemble
level-c:
	$(PYTHON) -m forecast_select level-c
freeze:
	$(PYTHON) -m forecast_select freeze
locked-audit:
	$(PYTHON) -m forecast_select locked-audit
train-final:
	$(PYTHON) -m forecast_select train-final
predict-month:
	$(PYTHON) -m forecast_select predict-month
report:
	$(PYTHON) -m forecast_select report
