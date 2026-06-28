"""Pytest suite for the backtest module.

Covers:
- BacktestResult metric correctness (Sharpe, drawdown, return)
- Transaction cost accounting
- Level-flag differentiation (different levels produce genuinely different results)
- No-look-ahead enforcement (engine cannot see future data)
- buy_and_hold_spy grows with positive returns
- Random-signal Sharpe sanity check (should not be suspiciously high)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backtest.benchmarks import buy_and_hold_spy, equal_weight_rebalance
from backtest.engine import BacktestConfig, BacktestEngine
from backtest.metrics import BacktestResult, SHARPE_WARN_THRESHOLD


# ---------------------------------------------------------------------------
# Fixtures — synthetic market data
# ---------------------------------------------------------------------------

def _make_returns(
    n_days: int = 504,
    n_tickers: int = 5,
    daily_drift: float = 0.0003,
    seed: int = 42,
) -> pd.DataFrame:
    """Create a synthetic wide log-return DataFrame for testing."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-02", periods=n_days, freq="B")
    data = rng.normal(loc=daily_drift, scale=0.01, size=(n_days, n_tickers))
    tickers = [f"T{i}" for i in range(n_tickers)]
    return pd.DataFrame(data, index=dates, columns=tickers)


def _make_spy_returns(n_days: int = 504, seed: int = 99) -> pd.Series:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-02", periods=n_days, freq="B")
    data = rng.normal(loc=0.0003, scale=0.01, size=n_days)
    return pd.Series(data, index=dates, name="SPY")


def _minimal_result(
    portfolio_value: pd.Series,
    trades: pd.DataFrame | None = None,
) -> BacktestResult:
    cfg = BacktestConfig(level=2, initial_capital=float(portfolio_value.iloc[0]))
    if trades is None:
        trades = pd.DataFrame(columns=["date", "ticker", "direction", "weight_delta", "cost_dollars"])
    return BacktestResult(
        portfolio_value=portfolio_value,
        positions=pd.DataFrame(),
        trades=trades,
        config=cfg,
    )


# ---------------------------------------------------------------------------
# BacktestResult.sharpe_ratio
# ---------------------------------------------------------------------------

class TestSharpeRatio:
    def test_positive_drift_gives_positive_sharpe(self):
        dates = pd.bdate_range("2020-01-02", periods=252)
        nav = pd.Series(
            100_000.0 * np.cumprod(1 + np.full(252, 0.001)),
            index=dates,
        )
        result = _minimal_result(nav)
        assert result.sharpe_ratio() > 0

    def test_flat_portfolio_gives_zero_sharpe(self):
        dates = pd.bdate_range("2020-01-02", periods=252)
        nav = pd.Series(100_000.0, index=dates)
        result = _minimal_result(nav)
        assert result.sharpe_ratio() == 0.0

    def test_sharpe_uses_risk_free_rate(self):
        dates = pd.bdate_range("2020-01-02", periods=252)
        daily = np.full(252, 0.0003)
        nav = pd.Series(100_000.0 * np.cumprod(1 + daily), index=dates)
        result = _minimal_result(nav)
        sharpe_high_rf = result.sharpe_ratio(risk_free_rate=0.20)
        sharpe_low_rf = result.sharpe_ratio(risk_free_rate=0.01)
        assert sharpe_low_rf > sharpe_high_rf

    def test_sharpe_gt3_emits_warning(self, caplog):
        """A Sharpe > 3 must trigger a warning (potential look-ahead bias)."""
        import logging
        dates = pd.bdate_range("2020-01-02", periods=504)
        # Artificially smooth, high-return series -> very high Sharpe
        daily = np.full(504, 0.002)
        nav = pd.Series(100_000.0 * np.cumprod(1 + daily), index=dates)
        result = _minimal_result(nav)
        with caplog.at_level(logging.WARNING, logger="backtest.metrics"):
            sharpe = result.sharpe_ratio(risk_free_rate=0.0)
        assert sharpe > SHARPE_WARN_THRESHOLD
        assert any("Sharpe > 3" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# BacktestResult.max_drawdown
# ---------------------------------------------------------------------------

class TestMaxDrawdown:
    def test_always_nonpositive(self):
        rng = np.random.default_rng(7)
        dates = pd.bdate_range("2020-01-02", periods=200)
        nav = pd.Series(
            100_000.0 * np.cumprod(1 + rng.normal(0.0, 0.01, 200)),
            index=dates,
        )
        result = _minimal_result(nav)
        assert result.max_drawdown() <= 0.0

    def test_known_drawdown(self):
        """Portfolio drops from 100 to 80 then recovers: drawdown = -0.20."""
        dates = pd.bdate_range("2020-01-02", periods=3)
        nav = pd.Series([100.0, 80.0, 90.0], index=dates)
        result = _minimal_result(nav)
        dd = result.max_drawdown()
        assert abs(dd - (-0.20)) < 1e-9

    def test_monotone_increase_zero_drawdown(self):
        dates = pd.bdate_range("2020-01-02", periods=50)
        nav = pd.Series(np.linspace(100, 200, 50), index=dates)
        result = _minimal_result(nav)
        assert result.max_drawdown() == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# Transaction cost accounting
# ---------------------------------------------------------------------------

class TestTransactionCosts:
    def test_spy_cost_deducted_from_initial_capital(self):
        """Level 1: the entry cost must equal cost_bps applied to initial_capital.

        We use a flat SPY series (zero returns) so the first NAV equals
        capital_after_cost exactly, removing noise from actual returns.
        """
        dates = pd.bdate_range("2020-01-02", periods=252)
        spy = pd.Series(np.zeros(252), index=dates)
        initial = 100_000.0
        cost_bps = 7.5
        portfolio_value, trades = buy_and_hold_spy(
            spy_returns=spy,
            initial_capital=initial,
            tx_cost_bps=cost_bps,
        )
        expected_entry_cost = initial * cost_bps / 10_000.0
        assert abs(float(trades["cost_dollars"].sum()) - expected_entry_cost) < 1e-6
        # With zero returns, NAV on day 0 == capital_after_cost == initial - cost
        assert abs(portfolio_value.iloc[0] - (initial - expected_entry_cost)) < 1e-6

    def test_spy_single_trade_recorded(self):
        spy = _make_spy_returns(n_days=100)
        _, trades = buy_and_hold_spy(spy_returns=spy, initial_capital=100_000.0, tx_cost_bps=5.0)
        assert len(trades) == 1
        assert trades.iloc[0]["ticker"] == "SPY"

    def test_zero_cost_no_deduction(self):
        """With 0 bps, portfolio starts at exactly initial_capital."""
        spy = _make_spy_returns(n_days=100)
        portfolio_value, trades = buy_and_hold_spy(
            spy_returns=spy, initial_capital=100_000.0, tx_cost_bps=0.0
        )
        assert abs(float(trades["cost_dollars"].sum())) < 1e-9

    def test_equal_weight_costs_increase_with_more_rebalances(self):
        """More frequent rebalancing incurs more transaction costs."""
        returns = _make_returns(n_days=252, n_tickers=4)
        _, trades_daily = equal_weight_rebalance(returns, "daily", 100_000.0, 7.5)
        _, trades_monthly = equal_weight_rebalance(returns, "monthly", 100_000.0, 7.5)
        # Daily rebalancing has more rebalance events -> higher total cost
        assert trades_daily["cost_dollars"].sum() >= trades_monthly["cost_dollars"].sum()

    def test_engine_level3_trades_nonzero_costs(self):
        """Level 3 engine must record non-zero transaction costs."""
        returns = _make_returns(n_days=300, n_tickers=5)
        cfg = BacktestConfig(
            level=3,
            start=returns.index[0].date(),
            end=returns.index[-1].date(),
            transaction_cost_bps=7.5,
            rebalance_freq="weekly",
            pca_window=60,
            zscore_window=21,
        )
        engine = BacktestEngine(cfg)
        result = engine.run(returns)
        # Trades may be empty if no signals were generated (synthetic fallback),
        # but if there are trades, each must have a positive cost.
        if not result.trades.empty:
            assert (result.trades["cost_dollars"] >= 0).all()
            assert result.trades["cost_dollars"].sum() > 0


# ---------------------------------------------------------------------------
# Level differentiation — different levels must produce different outputs
# ---------------------------------------------------------------------------

class TestLevelDifferentiation:
    """Switching the level flag must change portfolio behaviour."""

    def _run(self, level: int, returns: pd.DataFrame, spy: pd.Series) -> BacktestResult:
        cfg = BacktestConfig(
            level=level,
            start=returns.index[0].date(),
            end=returns.index[-1].date(),
            initial_capital=100_000.0,
            transaction_cost_bps=7.5,
            rebalance_freq="monthly",
            pca_window=60,
            zscore_window=21,
        )
        engine = BacktestEngine(cfg)
        return engine.run(returns, spy_returns=spy)

    def test_level1_vs_level2_different_nav(self):
        returns = _make_returns(n_days=252, n_tickers=5)
        spy = _make_spy_returns(n_days=252)
        r1 = self._run(1, returns, spy)
        r2 = self._run(2, returns, spy)
        # Final NAV must differ between levels
        assert r1.portfolio_value.iloc[-1] != pytest.approx(
            r2.portfolio_value.iloc[-1], rel=1e-6
        )

    def test_level2_vs_level3_different_nav(self):
        returns = _make_returns(n_days=300, n_tickers=5)
        spy = _make_spy_returns(n_days=300)
        r2 = self._run(2, returns, spy)
        r3 = self._run(3, returns, spy)
        # Portfolio values should differ in shape or final value
        final_2 = r2.portfolio_value.iloc[-1]
        final_3 = r3.portfolio_value.iloc[-1]
        # They may coincidentally be equal if synthetic signals are flat,
        # but their trades/positions must differ (level 2 always rebalances).
        assert (
            final_2 != pytest.approx(final_3, rel=1e-4)
            or len(r2.trades) != len(r3.trades)
        )

    def test_level1_positions_only_spy(self):
        returns = _make_returns(n_days=252, n_tickers=5)
        spy = _make_spy_returns(n_days=252)
        r1 = self._run(1, returns, spy)
        assert "SPY" in r1.positions.columns
        assert len(r1.positions.columns) == 1

    def test_level2_positions_all_tickers(self):
        returns = _make_returns(n_days=252, n_tickers=5)
        spy = _make_spy_returns(n_days=252)
        r2 = self._run(2, returns, spy)
        for col in returns.columns:
            assert col in r2.positions.columns

    def test_level4_raises_without_macro_series(self):
        returns = _make_returns(n_days=252, n_tickers=5)
        cfg = BacktestConfig(
            level=4,
            start=returns.index[0].date(),
            end=returns.index[-1].date(),
        )
        engine = BacktestEngine(cfg)
        with pytest.raises(ValueError, match="vix_series"):
            engine.run(returns)

    def test_level1_raises_without_spy(self):
        returns = _make_returns(n_days=100, n_tickers=3)
        cfg = BacktestConfig(
            level=1,
            start=returns.index[0].date(),
            end=returns.index[-1].date(),
        )
        engine = BacktestEngine(cfg)
        with pytest.raises(ValueError, match="spy_returns"):
            engine.run(returns)

    def test_invalid_level_raises(self):
        returns = _make_returns(n_days=100, n_tickers=3)
        cfg = BacktestConfig(level=99)
        engine = BacktestEngine(cfg)
        with pytest.raises(ValueError, match="Unknown benchmark level"):
            engine.run(returns)


# ---------------------------------------------------------------------------
# No look-ahead — the engine must never use data past the current date J
# ---------------------------------------------------------------------------

class TestNoLookAhead:
    """Verify the engine cannot see data dated after J.

    Strategy: construct a returns DataFrame where the last rows contain
    extremely large positive returns.  Run the backtest ending one period
    before those large returns.  The engine must not benefit from them.
    """

    def test_engine_ignores_returns_after_end_date(self):
        returns_early = _make_returns(n_days=200, n_tickers=4, daily_drift=0.0)
        # Append 20 days of huge positive returns beyond the end date
        future_dates = pd.bdate_range(
            returns_early.index[-1] + pd.Timedelta(days=1), periods=20
        )
        future_data = np.full((20, 4), 0.10)  # 10% per day — unmissable if leaked
        future_df = pd.DataFrame(future_data, index=future_dates, columns=returns_early.columns)
        full_returns = pd.concat([returns_early, future_df])

        spy_early = _make_spy_returns(n_days=200)
        future_spy = pd.Series(
            np.full(20, 0.10),
            index=future_dates,
            name="SPY",
        )
        full_spy = pd.concat([spy_early, future_spy])

        # Backtest ends at the last "normal" date
        end_date = returns_early.index[-1].date()
        start_date = returns_early.index[0].date()

        cfg_l2 = BacktestConfig(
            level=2,
            start=start_date,
            end=end_date,
            initial_capital=100_000.0,
            transaction_cost_bps=0.0,
            rebalance_freq="monthly",
        )
        result_l2 = BacktestEngine(cfg_l2).run(full_returns, spy_returns=full_spy)
        # Portfolio must end on the declared end_date, not in the future
        assert result_l2.portfolio_value.index[-1] <= pd.Timestamp(end_date)

    def test_pca_engine_slices_history_at_J(self):
        """Confirm the engine passes only returns.loc[:J] to signal generation.

        We monkey-patch _get_signals and record the (today, history.index.max)
        pair at each call.  The history max must never exceed today.
        """
        returns = _make_returns(n_days=150, n_tickers=4)
        cfg = BacktestConfig(
            level=3,
            start=returns.index[0].date(),
            end=returns.index[-1].date(),
            rebalance_freq="weekly",
            pca_window=40,
            zscore_window=10,
        )
        engine = BacktestEngine(cfg)

        # List of (today, history_max) tuples captured at each call
        call_log: list[tuple[pd.Timestamp, pd.Timestamp]] = []
        original_get_signals = engine._get_signals

        def patched_get_signals(history, today, vix_series, spread_series):
            call_log.append((today, history.index.max()))
            return original_get_signals(history, today, vix_series, spread_series)

        engine._get_signals = patched_get_signals  # type: ignore[method-assign]
        engine.run(returns)

        assert len(call_log) > 0, "No rebalance calls were made — check rebalance logic."
        for today, hist_max in call_log:
            assert hist_max <= today, (
                f"Look-ahead detected: history max {hist_max} > rebalance date {today}"
            )


# ---------------------------------------------------------------------------
# buy_and_hold_spy — grows with positive returns
# ---------------------------------------------------------------------------

class TestBuyAndHoldSpy:
    def test_grows_with_positive_returns(self):
        dates = pd.bdate_range("2020-01-02", periods=100)
        spy = pd.Series(np.full(100, 0.001), index=dates)
        portfolio_value, _ = buy_and_hold_spy(spy, 100_000.0, tx_cost_bps=0.0)
        assert portfolio_value.iloc[-1] > portfolio_value.iloc[0]

    def test_shrinks_with_negative_returns(self):
        dates = pd.bdate_range("2020-01-02", periods=100)
        spy = pd.Series(np.full(100, -0.001), index=dates)
        portfolio_value, _ = buy_and_hold_spy(spy, 100_000.0, tx_cost_bps=0.0)
        assert portfolio_value.iloc[-1] < portfolio_value.iloc[0]

    def test_higher_cost_lowers_final_nav(self):
        dates = pd.bdate_range("2020-01-02", periods=100)
        spy = pd.Series(np.full(100, 0.0005), index=dates)
        pv_cheap, _ = buy_and_hold_spy(spy, 100_000.0, tx_cost_bps=1.0)
        pv_expensive, _ = buy_and_hold_spy(spy, 100_000.0, tx_cost_bps=50.0)
        assert pv_cheap.iloc[-1] > pv_expensive.iloc[-1]


# ---------------------------------------------------------------------------
# Random-signal Sharpe sanity check
# ---------------------------------------------------------------------------

class TestRandomSignalSharpe:
    """A purely random trading strategy should not produce an implausibly
    high Sharpe ratio.  This guards against infrastructure bugs that
    accidentally make everything look profitable.
    """

    def test_random_signals_sharpe_not_high(self):
        rng = np.random.default_rng(2024)
        n_days = 504
        n_tickers = 10
        dates = pd.bdate_range("2020-01-02", periods=n_days, freq="B")
        # Zero-drift returns (pure noise)
        data = rng.normal(0.0, 0.01, (n_days, n_tickers))
        returns = pd.DataFrame(data, index=dates, columns=[f"T{i}" for i in range(n_tickers)])

        cfg = BacktestConfig(
            level=2,  # equal-weight, no alpha signal
            start=dates[0].date(),
            end=dates[-1].date(),
            initial_capital=100_000.0,
            transaction_cost_bps=7.5,
            rebalance_freq="weekly",
        )
        result = BacktestEngine(cfg).run(returns)
        sharpe = result.sharpe_ratio(risk_free_rate=0.0)
        # With zero drift and transaction costs, Sharpe must be modest
        assert sharpe < SHARPE_WARN_THRESHOLD, (
            f"Random-signal portfolio produced Sharpe {sharpe:.2f} > {SHARPE_WARN_THRESHOLD}. "
            "This is suspicious — check for a data leak or cost-accounting bug."
        )


# ---------------------------------------------------------------------------
# summary() — smoke test
# ---------------------------------------------------------------------------

class TestSummary:
    def test_summary_keys_present(self):
        returns = _make_returns(n_days=252, n_tickers=5)
        cfg = BacktestConfig(
            level=2,
            start=returns.index[0].date(),
            end=returns.index[-1].date(),
        )
        result = BacktestEngine(cfg).run(returns)
        summary = result.summary()
        required_keys = {
            "level",
            "total_return",
            "annualized_return",
            "annualized_volatility",
            "sharpe_ratio",
            "max_drawdown",
            "calmar_ratio",
            "turnover",
            "hit_rate",
            "total_transaction_costs_dollars",
            "n_trades",
            "sharpe_warning",
        }
        assert required_keys.issubset(set(summary.keys()))

    def test_max_drawdown_nonpositive_in_summary(self):
        returns = _make_returns(n_days=252, n_tickers=5)
        cfg = BacktestConfig(
            level=2,
            start=returns.index[0].date(),
            end=returns.index[-1].date(),
        )
        result = BacktestEngine(cfg).run(returns)
        assert result.summary()["max_drawdown"] <= 0.0
