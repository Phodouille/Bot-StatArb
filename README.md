# StatArb Bot

Statistical arbitrage bot using PCA + mean-reversion on 40-50 liquid US tech stocks (S&P 500). Two differentiating features over a classic stat arb baseline:

- **Adaptive PCA window** — adjusts dynamically to market volatility instead of a fixed lookback
- **Macro filters** — VIX + credit spreads reduce/suspend trading during stress regimes when mean-reversion breaks down

Paper trading on Alpaca (cash account). Four configurable benchmark levels in one codebase: buy-and-hold S&P500 → equal-weight sector → classic PCA stat arb → adaptive model with macro filters.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill in your Alpaca paper API keys
```

## Run the dashboard

```bash
streamlit run dashboard/app.py
```

Opens at `http://localhost:8501` — three pages:
- **Backtest** — compare all 4 benchmark levels, NAV chart, performance metrics, drawdown
- **Signals** — live z-score heatmap, macro filter status, active positions
- **Risk Monitor** — kill switch status, drawdown gauge, risk limits

## Run the tests

```bash
python -m pytest tests/ -v
```

## Repo structure

```
data/        Price data pipeline (Alpaca fetch, cleaning, parquet storage)
signals/     PCA, z-score, adaptive window, macro filters
backtest/    Backtest engine — all 4 benchmark levels
execution/   Order execution via Alpaca
risk/        Position sizing, drawdown kill switch, exposure checks
dashboard/   Streamlit app
config/      Universe, risk parameters
tests/       Unit tests
```

## Risk guardrails

- Max position per ticker: 5–10% of portfolio
- Kill switch triggers at 5% drawdown
- Market-neutral: long exposure ≈ short exposure
- Capital: 100k$ paper account on Alpaca
