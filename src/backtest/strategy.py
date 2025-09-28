"""Strategy interface and example strategies."""
from typing import Protocol
import pandas as pd


class Strategy(Protocol):
    """Protocol for strategies used by the backtester.

    A strategy must implement generate_signals which returns a Series of positions:
    1 for long, 0 for flat.
    """

    def generate_signals(self, prices: pd.Series) -> pd.Series:
        ...


class MovingAverageCrossStrategy:
    """Simple moving-average crossover strategy.

    Parameters:
    - short_window: lookback for short MA
    - long_window: lookback for long MA
    """

    def __init__(self, short_window: int = 50, long_window: int = 200):
        if short_window >= long_window:
            raise ValueError("short_window must be less than long_window")
        self.short_window = short_window
        self.long_window = long_window

    def generate_signals(self, prices: pd.Series) -> pd.Series:
        short_ma = prices.rolling(self.short_window).mean()
        long_ma = prices.rolling(self.long_window).mean()
        signal = (short_ma > long_ma).astype(int)
        # shift signal to represent position from next open (simple approach)
        return signal.shift(1).fillna(0).astype(int)


class MomentumStrategy:
    """Simple momentum strategy.

    Parameters
    - lookback: number of periods to compare against
    - threshold: minimum pct change over lookback to consider bullish

    This strategy goes long (1) when price_pct_change over `lookback` is
    greater than `threshold`, otherwise flat (0). The returned series is
    shifted by 1 to represent taking the position at the next bar.
    """

    def __init__(self, lookback: int = 5, threshold: float = 0.0):
        if lookback < 1:
            raise ValueError("lookback must be >= 1")
        self.lookback = lookback
        self.threshold = threshold

    def generate_signals(self, prices: pd.Series) -> pd.Series:
        pct = prices.pct_change(self.lookback)
        sig = (pct > self.threshold).astype(int)
        return sig.shift(1).fillna(0).astype(int)
