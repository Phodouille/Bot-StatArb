"""BacktestResult dataclass and performance metric computations.

All metrics are computed from daily portfolio value and trade records.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from backtest.engine import BacktestConfig

logger = logging.getLogger(__name__)

SHARPE_WARN_THRESHOLD = 3.0
TRADING_DAYS_PER_YEAR = 252


@dataclass
class BacktestResult:
    """Container for all backtest outputs and derived performance metrics.

    Attributes:
        portfolio_value: Daily portfolio NAV. index=date.
        positions: Target weights at each rebalance date.
            index=date, columns=ticker, values=signed weight
            (positive=long, negative=short).
        trades: Record of every trade executed.
            Columns: date, ticker, direction, weight_delta, cost_dollars.
        config: The BacktestConfig that produced this result.
    """

    portfolio_value: pd.Series
    positions: pd.DataFrame
    trades: pd.DataFrame
    config: "BacktestConfig"

    def _daily_returns(self) -> pd.Series:
        return self.portfolio_value.pct_change().dropna()

    def total_return(self) -> float:
        """Return (final_value / initial_value) - 1.

        Returns:
            Total return as a decimal fraction.
        """
        if self.portfolio_value.empty:
            return 0.0
        return float(self.portfolio_value.iloc[-1] / self.portfolio_value.iloc[0]) - 1.0

    def annualized_return(self) -> float:
        """Compound annual growth rate over the backtest horizon.

        Returns:
            CAGR as a decimal fraction.
        """
        dr = self._daily_returns()
        if dr.empty:
            return 0.0
        n_years = len(dr) / TRADING_DAYS_PER_YEAR
        if n_years <= 0:
            return 0.0
        total = self.total_return()
        return float((1.0 + total) ** (1.0 / n_years) - 1.0)

    def annualized_volatility(self) -> float:
        """Annualized standard deviation of daily returns.

        Returns:
            Annualized volatility as a decimal fraction.
        """
        dr = self._daily_returns()
        if len(dr) < 2:
            return 0.0
        return float(dr.std() * np.sqrt(TRADING_DAYS_PER_YEAR))

    def sharpe_ratio(self, risk_free_rate: float = 0.05) -> float:
        """Annualized Sharpe ratio.

        Args:
            risk_free_rate: Annual risk-free rate as a decimal fraction.

        Returns:
            Sharpe ratio. Emits a WARNING if the value exceeds
            SHARPE_WARN_THRESHOLD (likely indicates look-ahead bias or a
            data leak).
        """
        vol = self.annualized_volatility()
        if vol == 0.0:
            return 0.0
        sharpe = (self.annualized_return() - risk_free_rate) / vol
        if sharpe > SHARPE_WARN_THRESHOLD:
            logger.warning(
                "Sharpe > 3 detected (%.2f) — check for look-ahead bias. "
                "Common causes: (1) PCA fitted on full sample then applied "
                "in-sample, (2) z-scores computed with future data in the "
                "rolling window, (3) rebalance signals generated one step "
                "ahead of when they would be available in production.",
                sharpe,
            )
        return float(sharpe)

    def max_drawdown(self) -> float:
        """Maximum peak-to-trough drawdown (always <= 0).

        Returns:
            Max drawdown as a negative decimal fraction.
        """
        pv = self.portfolio_value
        if pv.empty:
            return 0.0
        rolling_peak = pv.cummax()
        drawdown = (pv - rolling_peak) / rolling_peak
        return float(drawdown.min())

    def calmar_ratio(self) -> float:
        """Annualized return divided by absolute max drawdown.

        Returns:
            Calmar ratio. Returns 0.0 when max drawdown is zero.
        """
        mdd = self.max_drawdown()
        if mdd == 0.0:
            return 0.0
        return float(self.annualized_return() / abs(mdd))

    def turnover(self) -> float:
        """Annualized average turnover.

        Computed as the total sum of absolute weight changes across all
        rebalances, divided by the number of years in the backtest.

        Returns:
            Annualized turnover (e.g. 2.0 means 200% per year).
        """
        if self.trades.empty:
            return 0.0
        total_abs_delta = self.trades["weight_delta"].abs().sum()
        dr = self._daily_returns()
        n_years = len(dr) / TRADING_DAYS_PER_YEAR
        if n_years <= 0:
            return 0.0
        return float(total_abs_delta / n_years)

    def hit_rate(self) -> float:
        """Fraction of days with a positive portfolio return.

        Returns:
            Hit rate in [0, 1].
        """
        dr = self._daily_returns()
        if dr.empty:
            return float("nan")
        return float((dr > 0).mean())

    def total_transaction_costs(self) -> float:
        """Total transaction costs paid over the backtest in dollars.

        Returns:
            Total cost in the same currency units as initial_capital.
        """
        if self.trades.empty:
            return 0.0
        return float(self.trades["cost_dollars"].sum())

    def summary(self) -> dict:
        """All performance metrics in a single dict.

        Returns:
            Dict with keys: level, total_return, annualized_return,
            annualized_volatility, sharpe_ratio, max_drawdown, calmar_ratio,
            turnover, hit_rate, total_transaction_costs_dollars,
            n_trades, sharpe_warning.
        """
        sharpe = self.sharpe_ratio()
        result = {
            "level": self.config.level,
            "total_return": self.total_return(),
            "annualized_return": self.annualized_return(),
            "annualized_volatility": self.annualized_volatility(),
            "sharpe_ratio": sharpe,
            "max_drawdown": self.max_drawdown(),
            "calmar_ratio": self.calmar_ratio(),
            "turnover": self.turnover(),
            "hit_rate": self.hit_rate(),
            "total_transaction_costs_dollars": self.total_transaction_costs(),
            "n_trades": len(self.trades),
            "sharpe_warning": sharpe > SHARPE_WARN_THRESHOLD,
        }
        return result
