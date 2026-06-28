"""Sensitivity analysis utility for the academic report.

Takes the backtest engine's output across a grid of risk parameter values
and computes how Sharpe ratio and maximum drawdown respond to each parameter.
This module is a pure analysis tool — it does NOT generate signals or place
orders.  It feeds the academic report, not the live bot.

Expected usage:
    results = run_sensitivity(
        backtest_fn=my_backtest,
        base_params=load_risk_params(),
        param_grid={
            "position.max_weight_per_ticker": [0.05, 0.08, 0.10, 0.15],
            "drawdown.kill_switch_threshold": [0.03, 0.05, 0.08],
        },
    )
    df = results_to_dataframe(results)
"""

from __future__ import annotations

import copy
import logging
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Annualisation factor for daily returns (252 trading days).
_TRADING_DAYS_PER_YEAR: int = 252


def _set_nested(d: dict, dotted_key: str, value: Any) -> dict:
    """Set a value in a nested dict using a dotted key path.

    Args:
        d: Dict to mutate (already a deep copy).
        dotted_key: Key path like ``"position.max_weight_per_ticker"``.
        value: Value to assign.

    Returns:
        The mutated dict.
    """
    parts = dotted_key.split(".")
    node = d
    for part in parts[:-1]:
        node = node[part]
    node[parts[-1]] = value
    return d


def compute_sharpe(equity_curve: pd.Series, risk_free_rate: float = 0.0) -> float:
    """Compute annualised Sharpe ratio from a daily equity curve.

    Args:
        equity_curve: Time-indexed series of portfolio values.
        risk_free_rate: Annualised risk-free rate as a decimal (default 0).

    Returns:
        Annualised Sharpe ratio.  Returns ``float('nan')`` if the curve has
        fewer than two data points or zero volatility.
    """
    if len(equity_curve) < 2:
        return float("nan")

    daily_returns = equity_curve.pct_change().dropna()
    if daily_returns.std() == 0:
        return float("nan")

    daily_rf = risk_free_rate / _TRADING_DAYS_PER_YEAR
    excess = daily_returns - daily_rf
    return float(excess.mean() / excess.std() * np.sqrt(_TRADING_DAYS_PER_YEAR))


def compute_max_drawdown(equity_curve: pd.Series) -> float:
    """Compute the maximum drawdown from a daily equity curve.

    Args:
        equity_curve: Time-indexed series of portfolio values.

    Returns:
        Maximum drawdown as a positive fraction (e.g. ``0.05`` means 5%).
        Returns ``0.0`` for a curve with fewer than two points.
    """
    if len(equity_curve) < 2:
        return 0.0

    rolling_max = equity_curve.cummax()
    drawdowns = (equity_curve - rolling_max) / rolling_max
    return float(abs(drawdowns.min()))


def run_sensitivity(
    backtest_fn: Callable[[dict[str, Any]], pd.Series],
    base_params: dict[str, Any],
    param_grid: dict[str, list[Any]],
    risk_free_rate: float = 0.0,
) -> list[dict[str, Any]]:
    """Run the backtest over a grid of risk parameter values.

    For each combination of parameter → value in ``param_grid``, the function
    substitutes that single value into a copy of ``base_params``, calls
    ``backtest_fn``, and records Sharpe + max drawdown.  Parameters are varied
    one at a time (not a full cross-product) to keep the search tractable and
    the report interpretable.

    Args:
        backtest_fn: Callable that accepts a risk params dict and returns a
            daily equity curve as a ``pd.Series`` (index=date, values=equity).
        base_params: Baseline risk params (typically from ``load_risk_params()``).
        param_grid: Dict mapping dotted-key param names to a list of candidate
            values.  Example::

                {
                    "position.max_weight_per_ticker": [0.05, 0.08, 0.10, 0.15],
                    "drawdown.kill_switch_threshold": [0.03, 0.05, 0.08],
                }

        risk_free_rate: Annualised risk-free rate for Sharpe computation.

    Returns:
        List of result dicts, each with keys:
            - ``param_name`` (str)
            - ``param_value`` (Any)
            - ``sharpe`` (float)
            - ``max_drawdown`` (float)
            - ``equity_curve`` (pd.Series)
    """
    results: list[dict[str, Any]] = []

    for param_name, values in param_grid.items():
        for value in values:
            trial_params = _set_nested(copy.deepcopy(base_params), param_name, value)

            logger.info(
                "Sensitivity run: %s = %s", param_name, value
            )

            try:
                equity_curve: pd.Series = backtest_fn(trial_params)
            except Exception:
                logger.exception(
                    "Sensitivity backtest failed for %s=%s — skipping.", param_name, value
                )
                continue

            sharpe = compute_sharpe(equity_curve, risk_free_rate)
            mdd = compute_max_drawdown(equity_curve)

            results.append(
                {
                    "param_name": param_name,
                    "param_value": value,
                    "sharpe": sharpe,
                    "max_drawdown": mdd,
                    "equity_curve": equity_curve,
                }
            )

            logger.info(
                "  → Sharpe=%.3f  MaxDD=%.3f%%", sharpe, mdd * 100
            )

    return results


def results_to_dataframe(results: list[dict[str, Any]]) -> pd.DataFrame:
    """Convert sensitivity results to a tidy DataFrame for reporting.

    Args:
        results: Output of ``run_sensitivity()``.

    Returns:
        DataFrame with columns ``["param_name", "param_value", "sharpe",
        "max_drawdown"]``, sorted by ``param_name`` then ``param_value``.
        The ``equity_curve`` column is dropped for readability.
    """
    rows = [
        {
            "param_name": r["param_name"],
            "param_value": r["param_value"],
            "sharpe": r["sharpe"],
            "max_drawdown": r["max_drawdown"],
        }
        for r in results
    ]
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["param_name", "param_value"]).reset_index(drop=True)
    return df
