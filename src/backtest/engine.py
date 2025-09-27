"""Backtesting engine: runs a strategy and computes results."""
from dataclasses import dataclass
import pandas as pd
import numpy as np
from typing import Optional


@dataclass
class BacktestResult:
    equity: pd.Series
    returns: pd.Series
    cumulative_return: float
    annualized_return: Optional[float] = None
    max_drawdown: Optional[float] = None


class Backtester:
    """Very small vectorized backtester.

    Assumptions/simplifications:
    - No transaction costs or slippage
    - Rebalance daily to target position (1 or 0)
    - Use close-to-close returns
    """

    def __init__(self, prices: pd.Series):
        if prices.empty:
            raise ValueError("prices series is empty")
        self.prices = prices.sort_index()

    def run(self, signals: pd.Series) -> BacktestResult:
        # align
        sig = signals.reindex(self.prices.index).fillna(0).astype(int)
        px = self.prices

        # daily returns
        daily_ret = px.pct_change().fillna(0)

        # strategy returns = position_{t} * return_{t}
        strat_ret = sig * daily_ret

        equity = (1 + strat_ret).cumprod()

        cumulative_return = equity.iloc[-1] - 1

        # annualized return (approx)
        days = (px.index[-1] - px.index[0]).days
        annualized_return = (1 + cumulative_return) ** (365.0 / max(days, 1)) - 1

        # max drawdown
        peak = equity.cummax()
        drawdown = (equity - peak) / peak
        max_drawdown = drawdown.min()

        return BacktestResult(
            equity=equity,
            returns=strat_ret,
            cumulative_return=float(cumulative_return),
            annualized_return=float(annualized_return),
            max_drawdown=float(max_drawdown),
        )
