"""S&P 500 price data fetching and caching via yfinance."""

import pandas as pd
import yfinance as yf

from config import TICKER, START_DATE, END_DATE, CACHE_FILE


def fetch_prices(use_cache: bool = True) -> pd.DataFrame:
    """Fetch S&P 500 daily prices. Uses local CSV cache by default."""
    if use_cache and CACHE_FILE.exists():
        df = pd.read_csv(CACHE_FILE, index_col=0, parse_dates=True)
        if not df.empty:
            return df

    ticker = yf.Ticker(TICKER)
    df = ticker.history(start=START_DATE, end=END_DATE, auto_adjust=True)
    df = df[["Close"]].rename(columns={"Close": "close"})
    df.index = df.index.tz_localize(None)
    df.index.name = "date"
    df = df.dropna()

    df.to_csv(CACHE_FILE)
    return df


def get_prices() -> pd.DataFrame:
    """Return cached prices, fetching if needed."""
    return fetch_prices(use_cache=True)
