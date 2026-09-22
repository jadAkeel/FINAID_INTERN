"""Write research metrics as standards-compliant JSON."""

import json
import math


def _finite(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _finite(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite(item) for item in value]
    return value


def strict_json_dumps(value, **kwargs):
    """Encode undefined numeric metrics as JSON null, preserving finite values."""
    return json.dumps(_finite(value), allow_nan=False, **kwargs)
