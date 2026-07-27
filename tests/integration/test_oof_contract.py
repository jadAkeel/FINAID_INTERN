from pathlib import Path

import pandas as pd

from forecast_select.schemas import validate_oof_columns


def test_completed_catboost_oof_matches_contract():
    path = Path("artifacts/oof_predictions/catboost_full_v2.parquet")
    assert path.exists()
    validate_oof_columns(pd.read_parquet(path))
