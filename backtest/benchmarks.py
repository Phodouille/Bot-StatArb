"""Benchmark portfolio construction helpers for levels 1 and 2.

Level 1 — buy_and_hold_spy: a single purchase of SPY held to end.
Level 2 — equal_weight_rebalance: 1/N across all tickers, rebalanced on
a configurable schedule.

These helpers return a pd.Series of daily portfolio NAV and a trades
DataFrame that the engine assembles into a BacktestResult.
"""
from __future__ import annotations

import logging
from typing import Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_TRADES_COLUMNS = ["date", "ticker", "direction", "weight_delta", "cost_dollars"]


def _rebalance_dates(index: pd.DatetimeIndex, freq: str) -> pd.DatetimeIndex:
    """Return subset of trading dates that are rebalance triggers.

    Always includes the first date in the index so the portfolio is
    initialised on day 1.

    Args:
        index: Full sorted DatetimeIndex of trading days.
        freq: One of ``"daily"``, ``"weekly"``, ``"monthly"``.

    Returns:
        DatetimeIndex of rebalance dates.

    Raises:
        ValueError: If freq is not recognised.
    """
    if index.empty:
        return index

    if freq == "daily":
        return index

    if freq == "weekly":
        # First trading day of each ISO week
        iso_week = index.isocalendar().week.values
        # Group by (year, week) to handle year boundaries correctly
        iso_year = index.isocalendar().year.values
        keys = [f"{y}-{w:02d}" for y, w in zip(iso_year, iso_week)]
        groups: dict[str, pd.Timestamp] = {}
        for ts, key in zip(index, keys):
            if key not in groups:
                groups[key] = ts
        return pd.DatetimeIndex(sorted(groups.values()))

    if freq == "monthly":
        groups_m: dict[str, pd.Timestamp] = {}
        for ts in index:
            key = ts.strftime("%Y-%m")
            if key not in groups_m:
                groups_m[key] = ts
        return pd.DatetimeIndex(sorted(groups_m.values()))

    raise ValueError(
        f"Unknown rebalance_freq: {freq!r}. Use 'daily', 'weekly', or 'monthly'."
    )


def buy_and_hold_spy(
    spy_returns: pd.Series,
    initial_capital: float,
    tx_cost_bps: float,
) -> Tuple[pd.Series, pd.DataFrame]:
    """Buy SPY on the first day and hold to the end (level 1 benchmark).

    Transaction cost is applied once at purchase (one-way cost on the
    full position). No rebalancing cost after that.

    Args:
        spy_returns: Daily log-returns of SPY. index=date (DatetimeIndex).
        initial_capital: Starting portfolio value in dollars.
        tx_cost_bps: One-way transaction cost in basis points.

    Returns:
        Tuple of:
            - portfolio_value: pd.Series of daily NAV. index=date.
            - trades: pd.DataFrame with columns matching _TRADES_COLUMNS.
    """
    if spy_returns.empty:
        empty_trades = pd.DataFrame(columns=_TRADES_COLUMNS)
        return pd.Series(dtype=float), empty_trades

    cost_fraction = tx_cost_bps / 10_000.0
    entry_cost = initial_capital * cost_fraction
    capital_after_cost = initial_capital - entry_cost

    # Convert log-returns to cumulative simple price ratio
    cum_log_ret = spy_returns.cumsum()
    price_ratio = np.exp(cum_log_ret)
    portfolio_value = capital_after_cost * price_ratio

    trades = pd.DataFrame(
        [
            {
                "date": spy_returns.index[0],
                "ticker": "SPY",
                "direction": "long",
                "weight_delta": 1.0,
                "cost_dollars": entry_cost,
            }
        ],
        columns=_TRADES_COLUMNS,
    )

    return portfolio_value, trades


def equal_weight_rebalance(
    returns: pd.DataFrame,
    rebalance_freq: str,
    initial_capital: float,
    tx_cost_bps: float,
) -> Tuple[pd.Series, pd.DataFrame]:
    """Equal-weight (1/N) portfolio rebalanced on schedule (level 2 benchmark).

    Portfolio construction
    ----------------------
    Weights are tracked as fractions.  Between rebalances the portfolio
    drifts freely: each ticker's weight evolves with its return.  At each
    scheduled rebalance the weights are reset to 1/N and transaction costs
    are charged on the sum of absolute weight changes.

    NAV evolution
    -------------
    On day t, NAV_t = NAV_{t-1} * (1 + sum_i w_{i,t-1} * r_{i,t})
    where r_{i,t} = exp(log_return_{i,t}) - 1.

    Transaction cost model
    ----------------------
    cost_t = NAV_{t-1} * sum_i |delta_w_i| * cost_bps / 10000
    Cost is deducted from NAV before updating weights so the position
    sizing correctly reflects the reduced capital.

    Args:
        returns: Wide log-return DataFrame. index=date, columns=tickers.
        rebalance_freq: One of ``"daily"``, ``"weekly"``, ``"monthly"``.
        initial_capital: Starting portfolio value in dollars.
        tx_cost_bps: One-way transaction cost in basis points per unit of
            weight moved.

    Returns:
        Tuple of:
            - portfolio_value: pd.Series of daily NAV. index=date.
            - trades: pd.DataFrame with columns matching _TRADES_COLUMNS.
    """
    if returns.empty:
        empty_trades = pd.DataFrame(columns=_TRADES_COLUMNS)
        return pd.Series(dtype=float), empty_trades

    tickers = list(returns.columns)
    n = len(tickers)
    target_weight = 1.0 / n
    cost_fraction = tx_cost_bps / 10_000.0

    reb_date_set = set(_rebalance_dates(returns.index, rebalance_freq))

    # weights[i] is the fraction of NAV invested in tickers[i]
    # Starts at 0; gets initialised on the first rebalance date.
    weights = np.zeros(n)
    nav = initial_capital

    nav_series: dict[pd.Timestamp, float] = {}
    trade_records: list[dict] = []

    for today in returns.index:
        # --- Drift phase: update NAV and weights using today's log-returns ---
        today_log_rets = returns.loc[today].values  # aligned to tickers order
        simple_rets = np.expm1(today_log_rets)      # exp(r) - 1

        # NAV change = sum of (weight_i * simple_return_i)
        portfolio_return = float(np.dot(weights, simple_rets))
        nav *= 1.0 + portfolio_return

        # Drift weights: each weight grows by its own return
        # w_i_new = w_i_old * (1 + r_i) / (1 + portfolio_return)
        # This keeps weights as fractions summing to 1 after drift.
        if abs(1.0 + portfolio_return) > 1e-12:
            weights = weights * (1.0 + simple_rets) / (1.0 + portfolio_return)

        # --- Rebalance phase ---
        if today in reb_date_set:
            target = np.full(n, target_weight)
            deltas = target - weights

            total_turnover = float(np.abs(deltas).sum())
            # Deduct cost from NAV before snapshotting the post-rebalance state
            rebalance_cost = nav * total_turnover * cost_fraction
            nav -= rebalance_cost

            # nav_before_cost is what we multiply weight-deltas by to get dollar cost
            nav_before_cost = nav + rebalance_cost
            for i, ticker in enumerate(tickers):
                delta = float(deltas[i])
                if abs(delta) < 1e-9:
                    continue
                # Proportional share of total rebalance cost for this ticker
                ticker_cost = abs(delta) * nav_before_cost * cost_fraction
                direction = "long" if delta > 0 else "short"
                trade_records.append(
                    {
                        "date": today,
                        "ticker": ticker,
                        "direction": direction,
                        "weight_delta": delta,
                        "cost_dollars": ticker_cost,
                    }
                )

            weights = target.copy()

        nav_series[today] = nav

    portfolio_value = pd.Series(nav_series, dtype=float)
    portfolio_value.index = pd.DatetimeIndex(portfolio_value.index)

    trades = pd.DataFrame(trade_records, columns=_TRADES_COLUMNS)
    return portfolio_value, trades
