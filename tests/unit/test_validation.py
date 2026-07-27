import pandas as pd
import pytest

from forecast_select.validation import assert_no_same_month_training, make_layout


def test_layout_is_dynamic_for_316_rows():
    layout = make_layout(316, 48)
    assert layout.development_origins[0] == 120
    assert layout.development_origins[-1] == 267
    assert layout.audit_origins == tuple(range(268, 316))
    assert layout.production_origin == 316


def test_same_month_training_is_rejected():
    with pytest.raises(AssertionError):
        assert_no_same_month_training(pd.DataFrame({"origin_position": [5]}), 5)

