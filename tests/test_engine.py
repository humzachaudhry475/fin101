
import pandas as pd

from src.backtest.engine import Backtester


def test_backtester_basic():
    dates = pd.date_range("2020-01-01", periods=5, freq="D")
    prices = pd.Series([100, 101, 102, 101, 103], index=dates)

    # simple buy-and-hold signal (always long)
    signals = pd.Series(1, index=dates)

    bt = Backtester(prices)
    res = bt.run(signals)

    # final equity should equal cumulative product of returns
    assert abs(res.equity.iloc[-1] - (1 + prices.pct_change().fillna(0)).prod()) < 1e-8
