# fin101 backtesting starter

Small starter project to experiment with backtesting strategies on the S&P 500.

Quick start

1. Create and activate a virtual environment (recommended).
2. Install dependencies:

   pip install -r requirements.txt

3. Run the example report:

   python -m reports.main

4. Run tests:

   pytest -q

Files of interest

- `src/backtest/` - core backtesting modules (data, strategy, engine)
- `reports/main.py` - example runner that downloads ^GSPC and runs a MA crossover
- `tests/` - unit tests for engine and strategy
# fin101
Trying things with friends
