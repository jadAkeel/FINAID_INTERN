# Unified Forecast Controller

This is a non-promoting meta-controller built on the frozen outputs of the
Directional Downside Selector, Downside Risk Gate, and Contextual Defensive
Selector.

The controller:

- uses the directional selector's Up/Down decision as its base;
- penalizes Up calls with relative downside risk;
- rewards Down calls when downside risk supports them; and
- gives a small stress-only bonus to the contextual defensive roles.

Weights are selected on tuning origins `120-179`. Validation origins `180-219`
and Confirmation origins `220-266` are reported separately. Locked origins
`268-315` are not read, and the active model is never changed automatically.

Run:

```powershell
python -m forecast_select build-unified-controller
python -m forecast_select show-unified-controller
```

The selected parameters and comparison metrics are written to `metrics/`, and
the prediction artifact is written to `artifacts/predictions.parquet`.

