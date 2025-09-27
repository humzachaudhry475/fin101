"""Simple backtesting package for Fin101 project."""

from .data import download_sp500, load_symbol
from .strategy import Strategy, MovingAverageCrossStrategy
from .engine import BacktestResult, Backtester
