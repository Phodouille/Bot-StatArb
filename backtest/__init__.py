"""Backtest package for the StatArb bot.

Public API
----------
BacktestConfig
    Dataclass holding all parameters for a single run (level, dates,
    cost model, signal thresholds, …).

BacktestEngine
    Single engine class; instantiate with a BacktestConfig and call .run().

BacktestResult
    Returned by BacktestEngine.run().  Carries portfolio_value, positions,
    trades, and all computed performance metrics via .summary().

Usage example
-------------
>>> from backtest import BacktestConfig, BacktestEngine
>>> cfg = BacktestConfig(level=2, rebalance_freq="monthly")
>>> engine = BacktestEngine(cfg)
>>> result = engine.run(returns_df)
>>> print(result.summary())
"""
from backtest.engine import BacktestConfig, BacktestEngine
from backtest.metrics import BacktestResult

__all__ = ["BacktestConfig", "BacktestEngine", "BacktestResult"]
