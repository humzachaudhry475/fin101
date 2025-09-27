#charts, summary of data
"""Report runner: example script to run a moving-average crossover on the S&P 500 index.

This script is intentionally simple. Run it after installing requirements in
`requirements.txt`.
"""

# Ensure the project root is on sys.path so `from src...` imports work
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
	sys.path.insert(0, str(ROOT))

from datetime import datetime
import matplotlib.pyplot as plt

from src.backtest.data import download_sp500
from src.backtest.strategy import MovingAverageCrossStrategy
from src.backtest.engine import Backtester


def run_example(start: str = "2010-01-01", end: str = None):
	print(f"Downloading S&P 500 data from {start} to {end}")
	df = download_sp500(start=start, end=end)
	# Prefer 'Adj Close' when available; fall back to 'Close' or the first
	# numeric column to be robust against changes in yfinance defaults.
	if "Adj Close" in df.columns:
		close = df["Adj Close"]
	elif "Close" in df.columns:
		close = df["Close"]
	else:
		# Pick the first numeric column
		numeric_cols = df.select_dtypes("number").columns
		if len(numeric_cols) == 0:
			raise RuntimeError("Downloaded data contains no numeric columns")
		close = df[numeric_cols[0]]

	strat = MovingAverageCrossStrategy(short_window=50, long_window=200)
	signals = strat.generate_signals(close)

	bt = Backtester(close)
	res = bt.run(signals)

	print(f"Cumulative return: {res.cumulative_return:.2%}")
	print(f"Annualized return (approx): {res.annualized_return:.2%}")
	print(f"Max drawdown: {res.max_drawdown:.2%}")

	# plot equity
	plt.figure(figsize=(10, 5))
	res.equity.plot(title="Strategy Equity Curve")
	plt.xlabel("Date")
	plt.ylabel("Equity (growth of 1.0)")
	plt.grid(True)
	plt.show()


if __name__ == "__main__":
	run_example()