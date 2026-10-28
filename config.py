"""Central configuration for momentum backtester."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

TICKER = "^GSPC"
TICKER_LABEL = "S&P 500"
START_DATE = "2010-01-01"
END_DATE = "2024-12-31"
CACHE_FILE = DATA_DIR / "sp500_prices.csv"

LOOKBACK = 60
TRANSACTION_COST_BPS = 10
RISK_FREE_RATE = 0.0
TRADING_DAYS_PER_YEAR = 252

BOOTSTRAP_ITERATIONS = 10000
BOOTSTRAP_CI = 0.95
