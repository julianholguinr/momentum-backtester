# Momentum Strategy Backtester

60-day momentum strategy backtester on S&P 500 historical data (2010-2024). Goes long when trailing 60-day return is positive, moves to cash otherwise. Evaluates risk-return trade-offs with transaction cost modeling, drawdown analysis, and statistical significance testing.

## Results (60-day lookback, 10bps cost)

| Metric | Strategy | Buy & Hold |
|--------|----------|------------|
| Annualized Return | 5.8% | 11.6% |
| Max Drawdown | -21.9% | -33.9% |
| Volatility | 11.2% | 17.3% |
| Sharpe Ratio | 0.52 | 0.67 |

The strategy reduces maximum drawdown by 12pp and volatility by 35%, trading return for downside protection in a secular bull market.

## Dashboard

Interactive Streamlit dashboard with 4 tabs:
- **Equity Curves** — strategy vs buy-and-hold with market exposure overlay
- **Drawdown Analysis** — underwater chart and worst drawdown periods
- **Performance Diagnostics** — rolling Sharpe, monthly heatmap, return distributions
- **Statistical Tests** — t-test, bootstrap confidence intervals, cumulative excess return

## Tech Stack

Python, Pandas, NumPy, yfinance, SciPy, Streamlit, Plotly

## Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Tests

```bash
pytest tests/ -v
```
