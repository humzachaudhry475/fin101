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
from pathlib import Path
import matplotlib.pyplot as plt
import argparse

from src.backtest.data import download_sp500
from src.backtest.strategy import MovingAverageCrossStrategy, MomentumStrategy
from src.backtest.engine import Backtester
import pandas as pd


def run_example(
	start: str = "2010-01-01",
	end: str | None = None,
	annotate: bool = True,
	exec_price: str = "close",  # 'close' or 'next_open'
	shade_positions: bool = True,
	save_plots: bool = False,
	save_dir: str | Path | None = None,
	interactive: bool = True,
	strategy: str = "ma",  # 'ma' or 'momentum'
):
	"""Run example and plot results.

	Parameters
	- annotate: add date/price annotations next to buy/sell markers
	- exec_price: 'close' to mark trades at the same-day close, or
	  'next_open' to use the Open price on the trade date (if available)
	- shade_positions: shade background while position == 1
	- save_plots: save PNG and PDF copies of the generated plots
	- save_dir: directory to save plots into (defaults to './reports_output')
	"""

	print(f"Downloading S&P 500 data from {start} to {end}")
	if end is None:
		df = download_sp500(start=start)
	else:
		df = download_sp500(start=start, end=end)

	# Determine a sensible close series
	if "Adj Close" in df.columns:
		close = df["Adj Close"]
	elif "Close" in df.columns:
		close = df["Close"]
	else:
		numeric_cols = df.select_dtypes("number").columns
		if len(numeric_cols) == 0:
			raise RuntimeError("Downloaded data contains no numeric columns")
		close = df[numeric_cols[0]]


	# Prepare save directory early so comparison branch can write files
	# Ensure save_path is always defined before any strategy branches
	save_path = None
	if save_plots:
		save_path = Path(save_dir or Path(__file__).resolve().parents[2] / "reports_output")
		save_path.mkdir(parents=True, exist_ok=True)

	# choose strategy
	if strategy.lower() in ("ma", "moving_average", "moving_average_cross"):
		strat = MovingAverageCrossStrategy(short_window=50, long_window=200)
		signals = strat.generate_signals(close)
		bt = Backtester(close)
		res = bt.run(signals)
	elif strategy.lower() in ("momentum", "mom"):
		strat = MomentumStrategy(lookback=5, threshold=0.0)
		signals = strat.generate_signals(close)
		bt = Backtester(close)
		res = bt.run(signals)
	elif strategy.lower() in ("compare", "both"):
		# run both strategies and plot equity curves together
		strat_ma = MovingAverageCrossStrategy(short_window=50, long_window=200)
		strat_mom = MomentumStrategy(lookback=5, threshold=0.0)
		signals_ma = strat_ma.generate_signals(close)
		signals_mom = strat_mom.generate_signals(close)
		bt = Backtester(close)
		res_ma = bt.run(signals_ma)
		res_mom = bt.run(signals_mom)

		print("--- Moving Average Strategy ---")
		print(f"Cumulative return: {res_ma.cumulative_return:.2%}")
		print(f"Annualized return (approx): {res_ma.annualized_return:.2%}")
		print(f"Max drawdown: {res_ma.max_drawdown:.2%}")
		print("--- Momentum Strategy ---")
		print(f"Cumulative return: {res_mom.cumulative_return:.2%}")
		print(f"Annualized return (approx): {res_mom.annualized_return:.2%}")
		print(f"Max drawdown: {res_mom.max_drawdown:.2%}")

		# plot comparison equity curves
		plt.figure(figsize=(10, 5))
		res_ma.equity.plot(label="MA Equity")
		res_mom.equity.plot(label="Momentum Equity")
		plt.title("Strategy equity comparison")
		plt.xlabel("Date")
		plt.ylabel("Equity (growth of 1.0)")
		plt.legend()
		plt.grid(True)
		if save_path:
			png = save_path / f"equity_compare_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
			pdf = save_path / f"equity_compare_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
			plt.savefig(png, bbox_inches="tight")
			plt.savefig(pdf, bbox_inches="tight")
			print(f"Saved comparison equity plots to: {png} and {pdf}")
		plt.show()

		# optionally save trade logs for each strategy
		def _build_and_save_trades(signals, name_suffix):
			pos2 = signals.reindex(close.index).fillna(0).astype(int).squeeze()
			tr2 = pos2.diff().fillna(0)
			buys2 = tr2[tr2 == 1].index
			sells2 = tr2[tr2 == -1].index
			rows2 = []
			for t in buys2:
				# robustly extract a single price value (DataFrame/Series -> scalar)
				if 'Open' in df.columns:
					val = df['Open'].reindex([t]).squeeze()
					try:
						price_val = float(val)
					except Exception:
						# fallback to iloc for single-element series
						price_val = float(val.iloc[0]) if hasattr(val, 'iloc') else float(val)
				else:
					val = close.reindex([t]).squeeze()
					price_val = float(val)
				rows2.append({"timestamp": t, "side": "buy", "price": price_val, "size": 1.0})
			for t in sells2:
				# robustly extract a single price value (DataFrame/Series -> scalar)
				if 'Open' in df.columns:
					val = df['Open'].reindex([t]).squeeze()
					try:
						price_val = float(val)
					except Exception:
						price_val = float(val.iloc[0]) if hasattr(val, 'iloc') else float(val)
				else:
					val = close.reindex([t]).squeeze()
					price_val = float(val)
				rows2.append({"timestamp": t, "side": "sell", "price": price_val, "size": 1.0})
			df_tr = pd.DataFrame(rows2).sort_values("timestamp")
			if save_path and not df_tr.empty:
				pth = save_path / f"detected_trades_{name_suffix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
				df_tr.to_csv(pth, index=False)
				print(f"Saved trades for {name_suffix} to: {pth}")

		_build_and_save_trades(signals_ma, "ma")
		_build_and_save_trades(signals_mom, "momentum")

		# Also plot the price series with buy/sell markers for both strategies
		plt.figure(figsize=(12, 6))
		plt.plot(close.index, close.to_numpy(), label="S&P 500 Close", color="#1f77b4")

		# helper to get price Series for an index (prefer 'Open' if available)
		def _price_series_for(idx):
			if len(idx) == 0:
				return pd.Series(dtype=float, index=idx)
			if "Open" in df.columns:
				return df["Open"].reindex(idx)
			return close.reindex(idx)

		# MA markers
		pos_ma = signals_ma.reindex(close.index).fillna(0).astype(int).squeeze()
		tr_ma = pos_ma.diff().fillna(0)
		buys_ma = tr_ma[tr_ma == 1].index
		sells_ma = tr_ma[tr_ma == -1].index
		buy_prices_ma = _price_series_for(buys_ma)
		sell_prices_ma = _price_series_for(sells_ma)

		# Momentum markers
		pos_mom = signals_mom.reindex(close.index).fillna(0).astype(int).squeeze()
		tr_mom = pos_mom.diff().fillna(0)
		buys_mom = tr_mom[tr_mom == 1].index
		sells_mom = tr_mom[tr_mom == -1].index
		buy_prices_mom = _price_series_for(buys_mom)
		sell_prices_mom = _price_series_for(sells_mom)

		# plot markers (use distinct shapes/colors)
		if len(buys_ma) > 0:
			plt.scatter(buys_ma, buy_prices_ma.to_numpy(), marker="^", color="green", s=80, label="MA Buy")
		if len(sells_ma) > 0:
			plt.scatter(sells_ma, sell_prices_ma.to_numpy(), marker="v", color="darkgreen", s=80, label="MA Sell")
		if len(buys_mom) > 0:
			plt.scatter(buys_mom, buy_prices_mom.to_numpy(), marker="s", color="orange", s=60, label="Momentum Buy")
		if len(sells_mom) > 0:
			plt.scatter(sells_mom, sell_prices_mom.to_numpy(), marker="x", color="red", s=60, label="Momentum Sell")

		plt.title("S&P 500 Price with Strategy Buys/Sells (MA vs Momentum)")
		plt.xlabel("Date")
		plt.ylabel("Price")
		plt.legend()
		plt.grid(True)
		if save_path:
			png = save_path / f"price_trades_compare_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
			pdf = save_path / f"price_trades_compare_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
			plt.savefig(png, bbox_inches="tight")
			plt.savefig(pdf, bbox_inches="tight")
			print(f"Saved comparison price/trade plots to: {png} and {pdf}")
		plt.show()

		# comparison done — return early (skip single-strategy plotting below)
		return
	else:
		raise ValueError("Unknown strategy. Use 'ma', 'momentum' or 'compare'.")

	bt = Backtester(close)
	res = bt.run(signals)

	print(f"Cumulative return: {res.cumulative_return:.2%}")
	print(f"Annualized return (approx): {res.annualized_return:.2%}")
	print(f"Max drawdown: {res.max_drawdown:.2%}")



	# detect plotly availability early so we can fallback gracefully
	plotly_available = False
	if interactive:
		try:
			import plotly.graph_objects as go  # type: ignore
			import plotly.io as pio  # type: ignore
			plotly_available = True
		except Exception:
			print("Plotly is not installed; falling back to static matplotlib plots. To enable interactive plots install plotly: pip install plotly")
			plotly_available = False

	# --- Equity plot ---
	plt.figure(figsize=(10, 5))
	res.equity.plot(title="Strategy Equity Curve")
	plt.xlabel("Date")
	plt.ylabel("Equity (growth of 1.0)")
	plt.grid(True)
	if save_path:
		png = save_path / f"equity_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
		pdf = save_path / f"equity_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
		plt.savefig(png, bbox_inches="tight")
		plt.savefig(pdf, bbox_inches="tight")
		print(f"Saved equity plots to: {png} and {pdf}")
	plt.show()

	# --- Price + trades plot ---
	# Align signals to price index and compute trade transitions
	# ensure we have a 1-D Series (squeeze in case signals is a DataFrame)
	pos = signals.reindex(close.index).fillna(0).astype(int).squeeze()
	trades = pos.diff().fillna(0)
	buys = trades[trades == 1].index
	sells = trades[trades == -1].index

	# Build trades DataFrame for export

	# choose execution price for annotation/markers
	def exec_price_of(idx):
		# idx is a DatetimeIndex; return a Series of prices for those timestamps
		if exec_price == "next_open":
			# prefer 'Open' from downloaded df; if missing, fallback to close
			if "Open" in df.columns:
				# Use the Open price on the trade date
				return df["Open"].reindex(idx)
			else:
				return close.loc[idx]
		else:
			return close.loc[idx]

	# Build trades DataFrame for export
	trade_rows = []
	def get_scalar_price_at(ts):
		"""Return a float price for a single Timestamp ts using exec_price_of.
		Handles Series, DataFrame, or scalar returns robustly."""
		res = exec_price_of(pd.DatetimeIndex([ts]))
		# squeeze DataFrame/Series to scalar or 1-element Series
		try:
			res = res.squeeze()
		except Exception:
			pass
		if isinstance(res, pd.Series):
			return float(res.iat[0])
		# otherwise assume scalar
		return float(res)

	for t in buys:
		price = get_scalar_price_at(t)
		trade_rows.append({"timestamp": t, "side": "buy", "price": price, "size": 1.0})
	for t in sells:
		price = get_scalar_price_at(t)
		trade_rows.append({"timestamp": t, "side": "sell", "price": price, "size": 1.0})

	trades_df = pd.DataFrame(trade_rows).sort_values("timestamp")
	if save_path and not trades_df.empty:
		csv_path = save_path / f"detected_trades_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
		trades_df.to_csv(csv_path, index=False)
		print(f"Saved detected trades CSV to: {csv_path}")

	# prepare marker price series for plotting
	buy_prices = exec_price_of(buys) if len(buys) > 0 else pd.Series(dtype=float)
	sell_prices = exec_price_of(sells) if len(sells) > 0 else pd.Series(dtype=float)

	# Use either matplotlib or an interactive Plotly figure
	if interactive and plotly_available:
		fig = go.Figure()
		fig.add_trace(go.Scatter(x=close.index, y=close.values, mode="lines", name="S&P 500 Close"))

		# shade long regions
		if shade_positions:
			# create rectangles for each contiguous long region using diffs
			long_mask = pos == 1
			diff = long_mask.astype(int).diff().fillna(0)
			starts = long_mask.index[diff == 1].tolist()
			ends = long_mask.index[diff == -1].tolist()
			# if mask starts with True, prepend first index
			if long_mask.iloc[0]:
				starts.insert(0, long_mask.index[0])
			# if mask ends with True, append last index to ends
			if long_mask.iloc[-1] and (len(ends) < len(starts)):
				ends.append(long_mask.index[-1])
			for s, e in zip(starts, ends):
				fig.add_vrect(x0=s, x1=e, fillcolor="#d0f0c0", opacity=0.2, layer="below", line_width=0)

		# add markers (hover shows date & price)
		if len(buys) > 0:
			fig.add_trace(go.Scatter(x=buys, y=buy_prices, mode="markers", marker_symbol="triangle-up", marker_color="green", marker_size=10, name="Buy", hovertemplate="%{x}<br>Buy: %{y:.2f}"))
		if len(sells) > 0:
			fig.add_trace(go.Scatter(x=sells, y=sell_prices, mode="markers", marker_symbol="triangle-down", marker_color="red", marker_size=10, name="Sell", hovertemplate="%{x}<br>Sell: %{y:.2f}"))

		fig.update_layout(title="S&P 500 Price with Strategy Buys/Sells", xaxis_title="Date", yaxis_title="Price")
		if save_path:
			html_path = save_path / f"price_trades_interactive_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
			pio.write_html(fig, file=html_path, auto_open=False)
			print(f"Saved interactive HTML plot to: {html_path}")
		fig.show()
	else:
		plt.figure(figsize=(12, 6))
		plt.plot(close.index, close.to_numpy(), label="S&P 500 Close", color="#1f77b4")

		# Shade positions when long
		if shade_positions:
			mask = pos == 1
			plt.fill_between(
				close.index,
				close.min() * 0.98,
				close.max() * 1.02,
				where=mask.to_numpy(),
				color="#d0f0c0",
				alpha=0.3,
				label="Long position",
			)

		if len(buys) > 0:
			plt.scatter(buys, buy_prices, marker="^", color="green", s=80, label="Buy")
		if len(sells) > 0:
			plt.scatter(sells, sell_prices, marker="v", color="red", s=80, label="Sell")

		# Annotate each trade with date and price if requested
		if annotate:
			for t in buys:
				p = get_scalar_price_at(t)
				plt.annotate(f"{t.date()}\n{p:.2f}", xy=(t, p), xytext=(0, 10), textcoords="offset points", ha="center", color="green", fontsize=8)
			for t in sells:
				p = get_scalar_price_at(t)
				plt.annotate(f"{t.date()}\n{p:.2f}", xy=(t, p), xytext=(0, -18), textcoords="offset points", ha="center", color="red", fontsize=8)

	# Shade positions when long
	if shade_positions:
		# Create a mask for positions and fill between
		mask = pos == 1
		# We need numeric x-values: use the index and convert to matplotlib dates
		plt.fill_between(
			close.index,
			close.min() * 0.98,
			close.max() * 1.02,
			where=mask.to_numpy(),
			color="#d0f0c0",
			alpha=0.3,
			label="Long position",
		)

	if len(buys) > 0:
		plt.scatter(buys, buy_prices, marker="^", color="green", s=80, label="Buy")
	if len(sells) > 0:
		plt.scatter(sells, sell_prices, marker="v", color="red", s=80, label="Sell")

	# Annotate each trade with date and price if requested
	if annotate:
		for t in buys:
			p = get_scalar_price_at(t)
			plt.annotate(f"{t.date()}\n{p:.2f}", xy=(t, p), xytext=(0, 10), textcoords="offset points", ha="center", color="green", fontsize=8)
		for t in sells:
			p = get_scalar_price_at(t)
			plt.annotate(f"{t.date()}\n{p:.2f}", xy=(t, p), xytext=(0, -18), textcoords="offset points", ha="center", color="red", fontsize=8)

	plt.title("S&P 500 Price with Strategy Buys/Sells")
	plt.xlabel("Date")
	plt.ylabel("Price")
	plt.legend()
	plt.grid(True)
	if save_path:
		png = save_path / f"price_trades_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
		pdf = save_path / f"price_trades_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
		plt.savefig(png, bbox_inches="tight")
		plt.savefig(pdf, bbox_inches="tight")
		print(f"Saved price/trade plots to: {png} and {pdf}")
	plt.show()


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Run example strategies and plot results.")
	parser.add_argument("--start", type=str, default="2010-01-01", help="Start date")
	parser.add_argument("--end", type=str, default=None, help="End date")
	parser.add_argument("--strategy", type=str, default="ma", help="Strategy: ma, momentum, or compare")
	parser.add_argument("--no-interactive", dest="interactive", action="store_false", help="Disable interactive (Plotly) plots")
	parser.add_argument("--save-plots", dest="save_plots", action="store_true", help="Save plots and trade CSVs to reports_output")
	args = parser.parse_args()

	run_example(start=args.start, end=args.end, interactive=args.interactive, save_plots=args.save_plots, strategy=args.strategy)