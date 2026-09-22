# Uptrend Selector performance

This is the retained Uptrend Selector baseline.

- Selection hits / calls: `926 / 1500`
- Top-15 accuracy: `61.7333%`
- Up / Down calls: `1500 / 0`
- Registered result reproduced: `True`
- Confirmation read: `False`
- Locked evaluation read: `False`

The pipeline is Structured Logistic with corrected cross-sectional rank, followed by a frozen signed correlation graph and a causal top-indicator selector.
