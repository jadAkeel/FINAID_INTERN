"""The signed graph must drop self-loops without depending on array ownership."""

import numpy as np
import pandas as pd

from forecast_select.indicator_selection import without_self_loops


def _correlation() -> pd.DataFrame:
    rng = np.random.default_rng(20260922)
    changes = pd.DataFrame(
        rng.normal(size=(60, 5)),
        columns=[f"X{index}" for index in range(5)],
    )
    return changes.corr()


def _read_only(frame: pd.DataFrame) -> pd.DataFrame:
    values = frame.to_numpy()
    values.flags.writeable = False
    return pd.DataFrame(values, index=frame.index, columns=frame.columns)


def test_self_loops_are_dropped_on_a_read_only_frame():
    """pandas may back a frame with a read-only array; the graph still builds."""
    graph = without_self_loops(_read_only(_correlation()))
    assert np.array_equal(np.diag(graph.to_numpy()), np.zeros(len(graph)))


def test_off_diagonal_correlations_are_untouched():
    correlation = _correlation()
    graph = without_self_loops(correlation)
    off_diagonal = ~np.eye(len(correlation), dtype=bool)
    assert np.array_equal(
        graph.to_numpy()[off_diagonal], correlation.to_numpy()[off_diagonal]
    )
    assert list(graph.index) == list(correlation.index)
    assert list(graph.columns) == list(correlation.columns)


def test_the_source_frame_is_not_mutated():
    correlation = _correlation()
    before = correlation.to_numpy(copy=True)
    without_self_loops(correlation)
    assert np.array_equal(correlation.to_numpy(), before)
