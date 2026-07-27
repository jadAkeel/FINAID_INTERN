from forecast_select.pipeline import catboost_chunk_origins
from forecast_select.validation import make_layout


def test_catboost_chunks_cover_development_origins_without_overlap():
    chunks = catboost_chunk_origins(make_layout(316, 48), chunk_size=8)
    flattened = [origin for chunk in chunks for origin in chunk]
    assert flattened == list(range(120, 268))
    assert len(flattened) == len(set(flattened))
    assert len(chunks[0]) == 8
    assert len(chunks[-1]) == 4

