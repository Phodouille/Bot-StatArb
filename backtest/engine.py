"""Backtest engine for all 4 benchmark levels in a single configurable class.

Level 1 — buy-and-hold SPY.
Level 2 — equal-weight sector rebalance.
Level 3 — classic PCA stat arb (fixed window, no macro filter).
Level 4 — adaptive-window PCA stat arb + macro filter (the full model).

The engine iterates strictly forward in time: at date J it only uses data
with index <= J.  No future data is ever referenced.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date
from typing import Dict, Optional

import numpy as np
import pandas as pd

from backtest.benchmarks import buy_and_hold_spy, equal_weight_rebalance, _rebalance_dates
from backtest.metrics import BacktestResult

# Conditional import: signals module may not exist yet at bootstrap time.
try:
    from signals.generator import generate_signals  # type: ignore[import]

    _SIGNALS_AVAILABLE = True
except ImportError:
    _SIGNALS_AVAILABLE = False

logger = logging.getLogger(__name__)

_TRADES_COLUMNS = ["date", "ticker", "direction", "weight_delta", "cost_dollars"]
SHARPE_WARN_THRESHOLD = 3.0


@dataclass
class BacktestConfig:
    """Configuration for a single backtest run.

    Attributes:
        level: Benchmark level 1-4.
            1 = buy-and-hold SPY
            2 = equal-weight sector
            3 = PCA stat arb, fixed window, no macro filter
            4 = PCA stat arb, adaptive window + macro filter
        start: Inclusive start date for the backtest.
        end: Inclusive end date for the backtest.
        initial_capital: Starting NAV in dollars.
        transaction_cost_bps: Round-trip cost per unit of weight traded, in
            basis points.  Applied on |delta_weight| at each rebalance.
        rebalance_freq: How often to rebalance — ``"daily"``, ``"weekly"``,
            or ``"monthly"``.
        pca_window: Lookback in trading days for the fixed-window PCA
            (level 3).  Ignored for levels 1-2.
        n_components: Number of PCA factors to retain (levels 3-4).
        zscore_window: Rolling window for z-score normalisation (levels 3-4).
        entry_threshold: |z-score| at which a position is opened (levels 3-4).
        exit_threshold: |z-score| at which a position is closed (levels 3-4).
        vix_threshold: VIX level above which macro filter suspends trading
            (level 4 only).
        spread_threshold: Credit spread (%) above which macro filter suspends
            trading (level 4 only).
        max_weight_per_ticker: Hard cap on absolute position weight per ticker.
    """

    level: int = 4
    start: date = field(default_factory=lambda: date(2020, 1, 1))
    end: date = field(default_factory=lambda: date(2024, 12, 31))
    initial_capital: float = 100_000.0
    transaction_cost_bps: float = 7.5
    rebalance_freq: str = "weekly"
    pca_window: int = 252
    n_components: int = 5
    zscore_window: int = 21
    entry_threshold: float = 2.0
    exit_threshold: float = 0.5
    vix_threshold: float = 30.0
    spread_threshold: float = 3.0
    max_weight_per_ticker: float = 0.10


class BacktestEngine:
    """Unified backtest engine that executes any of the 4 benchmark levels.

    Args:
        config: A BacktestConfig instance controlling which level runs and all
            model/cost parameters.

    Example:
        >>> cfg = BacktestConfig(level=2, rebalance_freq="monthly")
        >>> engine = BacktestEngine(cfg)
        >>> result = engine.run(returns_df)
        >>> print(result.summary())
    """

    def __init__(self, config: BacktestConfig) -> None:
        self.config = config

    def run(
        self,
        returns: pd.DataFrame,
        spy_returns: Optional[pd.Series] = None,
        vix_series: Optional[pd.Series] = None,
        spread_series: Optional[pd.Series] = None,
    ) -> BacktestResult:
        """Execute the backtest and return a BacktestResult.

        Iterates date by date in forward order.  At rebalance date J, only
        ``returns.loc[:J]`` (and similarly for vix/spread) is visible.

        Args:
            returns: Wide log-return DataFrame produced by ``load_returns()``.
                index=DatetimeIndex, columns=ticker.
            spy_returns: SPY log-returns. Required for level 1.  index must
                align with ``returns``.
            vix_series: Daily VIX levels. Required for level 4.
            spread_series: Daily credit spread (%). Required for level 4.

        Returns:
            BacktestResult populated with portfolio_value, positions, trades,
            and the original config.

        Raises:
            ValueError: If a required input is missing for the requested level,
                or if the level is not in 1-4.
        """
        cfg = self.config
        returns = self._filter_date_range(returns)

        if returns.empty:
            logger.warning("No returns data in the requested date range.")
            return self._empty_result()

        if cfg.level == 1:
            return self._run_level1(returns, spy_returns)
        if cfg.level == 2:
            return self._run_level2(returns)
        if cfg.level in (3, 4):
            return self._run_pca(returns, vix_series, spread_series)
        raise ValueError(f"Unknown benchmark level: {cfg.level}. Must be 1-4.")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _filter_date_range(self, returns: pd.DataFrame) -> pd.DataFrame:
        """Slice returns to [config.start, config.end] inclusive."""
        start_ts = pd.Timestamp(self.config.start)
        end_ts = pd.Timestamp(self.config.end)
        mask = (returns.index >= start_ts) & (returns.index <= end_ts)
        return returns.loc[mask]

    def _empty_result(self) -> BacktestResult:
        return BacktestResult(
            portfolio_value=pd.Series(dtype=float),
            positions=pd.DataFrame(),
            trades=pd.DataFrame(columns=_TRADES_COLUMNS),
            config=self.config,
        )

    def _run_level1(
        self,
        returns: pd.DataFrame,
        spy_returns: Optional[pd.Series],
    ) -> BacktestResult:
        """Level 1: buy-and-hold SPY."""
        if spy_returns is None:
            raise ValueError("spy_returns is required for level=1.")

        spy_slice = spy_returns.loc[
            spy_returns.index >= pd.Timestamp(self.config.start)
        ]
        spy_slice = spy_slice.loc[
            spy_slice.index <= pd.Timestamp(self.config.end)
        ]

        portfolio_value, trades = buy_and_hold_spy(
            spy_returns=spy_slice,
            initial_capital=self.config.initial_capital,
            tx_cost_bps=self.config.transaction_cost_bps,
        )
        positions = pd.DataFrame(
            {"SPY": [1.0]},
            index=[spy_slice.index[0]] if not spy_slice.empty else pd.DatetimeIndex([]),
        )
        result = BacktestResult(
            portfolio_value=portfolio_value,
            positions=positions,
            trades=trades,
            config=self.config,
        )
        self._check_sharpe(result)
        return result

    def _run_level2(self, returns: pd.DataFrame) -> BacktestResult:
        """Level 2: equal-weight sector rebalance."""
        portfolio_value, trades = equal_weight_rebalance(
            returns=returns,
            rebalance_freq=self.config.rebalance_freq,
            initial_capital=self.config.initial_capital,
            tx_cost_bps=self.config.transaction_cost_bps,
        )
        # Build positions snapshot: at each rebalance all weights are 1/N
        n = len(returns.columns)
        reb_dates = _rebalance_dates(returns.index, self.config.rebalance_freq)
        positions = pd.DataFrame(
            1.0 / n,
            index=reb_dates,
            columns=returns.columns,
        )
        result = BacktestResult(
            portfolio_value=portfolio_value,
            positions=positions,
            trades=trades,
            config=self.config,
        )
        self._check_sharpe(result)
        return result

    def _run_pca(
        self,
        returns: pd.DataFrame,
        vix_series: Optional[pd.Series],
        spread_series: Optional[pd.Series],
    ) -> BacktestResult:
        """Levels 3 and 4: rolling PCA stat arb (with or without macro filter).

        Walk-forward: for each rebalance date J we call generate_signals()
        with data up to and including J.  If the signals module is not yet
        available, we fall back to a synthetic flat-weight signal so the
        engine still runs in isolation.
        """
        cfg = self.config
        cost_fraction = cfg.transaction_cost_bps / 10_000.0

        if cfg.level == 4:
            if vix_series is None or spread_series is None:
                raise ValueError("vix_series and spread_series are required for level=4.")

        reb_dates = _rebalance_dates(returns.index, cfg.rebalance_freq)

        nav = cfg.initial_capital
        # current positions: ticker -> signed weight (positive=long, negative=short)
        current_positions: Dict[str, float] = {}

        nav_series: dict[pd.Timestamp, float] = {}
        position_records: list[dict] = []
        trade_records: list[dict] = []

        reb_date_set = set(reb_dates)

        for today in returns.index:
            # Drift current NAV using today's returns for each held position
            today_rets = returns.loc[today]
            pnl = 0.0
            for ticker, signed_weight in current_positions.items():
                if ticker in today_rets.index:
                    pnl += signed_weight * float(np.expm1(today_rets[ticker]))
            nav *= 1.0 + pnl

            # Rebalance if scheduled
            if today in reb_date_set:
                # Generate target signals using only history <= today
                history = returns.loc[:today]

                target_signals = self._get_signals(
                    history=history,
                    today=today,
                    vix_series=vix_series,
                    spread_series=spread_series,
                )

                # Convert signal dict to signed weight dict
                target_positions: Dict[str, float] = {}
                for ticker, sig in target_signals.items():
                    w = float(sig.get("weight", 0.0))
                    w = min(w, cfg.max_weight_per_ticker)
                    if sig.get("direction") == "long":
                        target_positions[ticker] = w
                    elif sig.get("direction") == "short":
                        target_positions[ticker] = -w

                # Compute delta, charge transaction costs
                all_tickers = set(current_positions) | set(target_positions)
                rebalance_cost = 0.0
                for ticker in all_tickers:
                    old_w = current_positions.get(ticker, 0.0)
                    new_w = target_positions.get(ticker, 0.0)
                    delta = new_w - old_w
                    if abs(delta) < 1e-9:
                        continue
                    ticker_cost = abs(delta) * nav * cost_fraction
                    rebalance_cost += ticker_cost
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

                nav -= rebalance_cost
                current_positions = target_positions

                # Snapshot positions
                snap = {t: w for t, w in current_positions.items()}
                snap["_date"] = today
                position_records.append(snap)

            nav_series[today] = nav

        portfolio_value = pd.Series(nav_series)
        portfolio_value.index = pd.DatetimeIndex(portfolio_value.index)

        trades = pd.DataFrame(trade_records, columns=_TRADES_COLUMNS)

        if position_records:
            positions = pd.DataFrame(position_records).set_index("_date")
            positions.index.name = None
            positions = positions.fillna(0.0)
        else:
            positions = pd.DataFrame()

        result = BacktestResult(
            portfolio_value=portfolio_value,
            positions=positions,
            trades=trades,
            config=self.config,
        )
        self._check_sharpe(result)
        return result

    def _get_signals(
        self,
        history: pd.DataFrame,
        today: pd.Timestamp,
        vix_series: Optional[pd.Series],
        spread_series: Optional[pd.Series],
    ) -> Dict[str, dict]:
        """Call signals.generator.generate_signals() if available, else synthetic.

        At date J this only passes history.loc[:J] into the signal generator,
        enforcing the no-look-ahead constraint at the call boundary.

        Args:
            history: Returns up to and including today (never beyond).
            today: The current simulation date.
            vix_series: Full VIX series (we slice to :today inside).
            spread_series: Full spread series (we slice to :today inside).

        Returns:
            Signals dict matching CONTRAT 1.
        """
        cfg = self.config

        # Macro filter scalars (level 4 only)
        use_macro_filter = cfg.level == 4
        vix_now: Optional[float] = None
        spread_now: Optional[float] = None

        if use_macro_filter and vix_series is not None:
            vix_to_date = vix_series.loc[:today]
            if not vix_to_date.empty:
                vix_now = float(vix_to_date.iloc[-1])

        if use_macro_filter and spread_series is not None:
            spread_to_date = spread_series.loc[:today]
            if not spread_to_date.empty:
                spread_now = float(spread_to_date.iloc[-1])

        if _SIGNALS_AVAILABLE:
            try:
                signals = generate_signals(
                    returns=history,
                    level=cfg.level,
                    window=cfg.pca_window,
                    n_components=cfg.n_components,
                    zscore_window=cfg.zscore_window,
                    entry_threshold=cfg.entry_threshold,
                    exit_threshold=cfg.exit_threshold,
                    vix=vix_now,
                    credit_spread=spread_now,
                    vix_threshold=cfg.vix_threshold,
                    spread_threshold=cfg.spread_threshold,
                )
                return signals
            except Exception:
                logger.exception(
                    "generate_signals() raised an exception at %s; using empty signals.",
                    today,
                )
                return {}

        # Fallback synthetic signals when signals module is not yet available
        return self._synthetic_signals(history)

    def _synthetic_signals(self, history: pd.DataFrame) -> Dict[str, dict]:
        """Minimal synthetic signals used when signals/ module is not present.

        Produces equal-weight long/short on the two extreme z-score tickers
        computed from a simple rolling mean/std over the available history.
        This is intentionally naive — it exists only so the backtest engine
        can be tested in isolation before signals/ is complete.

        IMPORTANT: This synthetic path must never be used in production.
        The engine always prefers the real generate_signals() when available.
        """
        cfg = self.config
        if len(history) < max(cfg.zscore_window, 5):
            return {}

        # Use last zscore_window rows to compute a simple cross-sectional z-score
        window_data = history.iloc[-cfg.zscore_window :]
        mean_ret = window_data.mean()
        std_ret = window_data.std()

        # Replace near-zero std to avoid division issues
        std_ret = std_ret.replace(0, np.nan)
        z = (history.iloc[-1] - mean_ret) / std_ret
        z = z.dropna()

        if z.empty:
            return {}

        signals: Dict[str, dict] = {}
        max_w = cfg.max_weight_per_ticker

        most_negative = z.idxmin()
        if float(z[most_negative]) < -cfg.entry_threshold:
            signals[most_negative] = {
                "direction": "long",
                "z_score": float(z[most_negative]),
                "weight": max_w,
            }

        most_positive = z.idxmax()
        if float(z[most_positive]) > cfg.entry_threshold:
            signals[most_positive] = {
                "direction": "short",
                "z_score": float(z[most_positive]),
                "weight": max_w,
            }

        return signals

    @staticmethod
    def _check_sharpe(result: BacktestResult) -> None:
        """Emit a warning (already handled inside sharpe_ratio) by calling it."""
        result.sharpe_ratio()
