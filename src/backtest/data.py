"""Data loading utilities for backtesting."""
from typing import List
import pandas as pd
import yfinance as yf


def load_symbol(symbol: str, start: str = "2010-01-01", end: str = None) -> pd.DataFrame:
    """Download OHLCV data for a single symbol using yfinance.

    Returns a DataFrame indexed by Date with columns: Open, High, Low, Close, Adj Close, Volume
    """
    # Be explicit about adjustments so the returned frame contains the
    # 'Adj Close' column. yfinance changed the default of auto_adjust;
    # setting auto_adjust=False preserves the 'Adj Close' column.
    df = yf.download(symbol, start=start, end=end, progress=False, auto_adjust=False)
    if df.empty:
        raise ValueError(f"No data returned for {symbol}")
    return df


def download_sp500(start: str = "2010-01-01", end: str = None) -> pd.DataFrame:
    """Download historical prices for S&P 500 index (using ^GSPC) as a proxy.

    For live symbol lists you'd normally scrape or use an API; here we use the index itself.
    """
    return load_symbol("^GSPC", start=start, end=end)
