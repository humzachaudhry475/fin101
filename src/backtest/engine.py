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

        def _to_scalar(x):
            """Convert a pandas/numpy scalar or single-element container to a Python float.

            Handles: pandas Series/Index with one element, numpy scalar/array with one element,
            and plain Python numeric types.
            """
            # pandas Series with a single value
            try:
                import numpy as _np
                import pandas as _pd
            except Exception:
                _np = None
                _pd = None

            if _pd is not None and isinstance(x, _pd.Series):
                if x.size == 0:
                    return float("nan")
                return float(x.iloc[0])
            # numpy array or scalar
            if _np is not None:
                if isinstance(x, _np.ndarray):
                    if x.size == 0:
                        return float("nan")
                    return float(x.ravel()[0])
                if _np.isscalar(x):
                    return float(x)
            # fallback for plain python numeric
            return float(x)

        return BacktestResult(
            equity=equity,
            returns=strat_ret,
            cumulative_return=_to_scalar(cumulative_return),
            annualized_return=_to_scalar(annualized_return),
            max_drawdown=_to_scalar(max_drawdown),
        )
