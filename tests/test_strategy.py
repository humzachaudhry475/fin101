import pandas as pd
import numpy as np

from src.backtest.strategy import MovingAverageCrossStrategy


from src.backtest.strategy import MovingAverageCrossStrategy

def test_moving_average_cross_signals():
    # Create a price series that trends up then down
    dates = pd.date_range("2020-01-01", periods=10, freq="D")
    prices = pd.Series([1, 2, 3, 4, 5, 4, 3, 2, 1, 0.5], index=dates)

    strat = MovingAverageCrossStrategy(short_window=2, long_window=4)
    signals = strat.generate_signals(prices)

    assert isinstance(signals, pd.Series)
    # signals should be 0/1 and align with the index
    assert set(signals.unique()).issubset({0, 1})

