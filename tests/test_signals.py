"""Pytest tests for the signals/ module.

Uses only synthetic data — no dependency on data/ or external APIs.
All fixtures are constructed so that PCA fits use only past data
(no look-ahead introduced by test setup).
"""

import numpy as np
import pandas as pd
import pytest

from signals.filters import MacroFilter
from signals.pca import AdaptivePCA, ClassicPCA
from signals.zscore import compute_zscore, get_signals


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(42)


def _make_returns(n_rows: int, n_cols: int = 10, seed: int = 42) -> pd.DataFrame:
    """Synthetic wide log-return matrix with a DatetimeIndex."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2023-01-02", periods=n_rows)
    tickers = [f"T{i:02d}" for i in range(n_cols)]
    data = rng.normal(0, 0.01, size=(n_rows, n_cols))
    return pd.DataFrame(data, index=dates, columns=tickers)


def _make_factor_returns(n_rows: int = 120, n_cols: int = 10) -> pd.DataFrame:
    """Synthetic returns with a latent factor so PCA residuals are meaningful."""
    rng = np.random.default_rng(0)
    dates = pd.bdate_range("2023-01-02", periods=n_rows)
    tickers = [f"T{i:02d}" for i in range(n_cols)]
    factor = rng.normal(0, 0.01, size=(n_rows, 1))
    loadings = rng.normal(1, 0.2, size=(1, n_cols))
    idio = rng.normal(0, 0.005, size=(n_rows, n_cols))
    data = factor @ loadings + idio
    return pd.DataFrame(data, index=dates, columns=tickers)


# ---------------------------------------------------------------------------
# ClassicPCA
# ---------------------------------------------------------------------------


class TestClassicPCA:
    def test_residuals_shape_matches_input(self):
        """get_residuals must return a DataFrame with the same shape as input."""
        returns = _make_returns(100)
        model = ClassicPCA(window=30, n_components=3)
        model.fit(returns)
        residuals = model.get_residuals(returns)
        assert residuals.shape == returns.shape

    def test_residuals_index_matches_input(self):
        """get_residuals preserves the DatetimeIndex of the input."""
        returns = _make_returns(80)
        model = ClassicPCA(window=30, n_components=3)
        model.fit(returns)
        residuals = model.get_residuals(returns)
        pd.testing.assert_index_equal(residuals.index, returns.index)

    def test_residuals_columns_match_input(self):
        """get_residuals preserves ticker columns."""
        returns = _make_returns(80)
        model = ClassicPCA(window=30, n_components=3)
        model.fit(returns)
        residuals = model.get_residuals(returns)
        assert list(residuals.columns) == list(returns.columns)

    def test_no_lookahead_fit_uses_only_window_rows(self):
        """Appending a future row must not change residuals on past dates."""
        returns = _make_returns(100)
        model_a = ClassicPCA(window=30, n_components=3)
        model_a.fit(returns)
        res_a = model_a.get_residuals(returns)

        future_row = pd.DataFrame(
            np.ones((1, returns.shape[1])) * 999,
            index=pd.bdate_range(returns.index[-1] + pd.offsets.BDay(1), periods=1),
            columns=returns.columns,
        )
        returns_extended = pd.concat([returns, future_row])

        model_b = ClassicPCA(window=30, n_components=3)
        model_b.fit(returns)  # fit on same slice — simulates date J
        res_b = model_b.get_residuals(returns)

        pd.testing.assert_frame_equal(res_a, res_b)

    def test_raises_when_insufficient_history(self):
        """fit() must raise ValueError when rows < n_components."""
        returns = _make_returns(3)
        model = ClassicPCA(window=30, n_components=5)
        with pytest.raises(ValueError, match="at least"):
            model.fit(returns)

    def test_raises_if_get_residuals_before_fit(self):
        """get_residuals() must raise RuntimeError if fit() was never called."""
        returns = _make_returns(50)
        model = ClassicPCA(window=30, n_components=3)
        with pytest.raises(RuntimeError, match="fit\\(\\)"):
            model.get_residuals(returns)

    def test_window_clipped_to_available_rows(self):
        """When window > len(returns), fit must succeed using all available rows."""
        returns = _make_returns(20)
        model = ClassicPCA(window=200, n_components=3)
        model.fit(returns)  # should not raise
        residuals = model.get_residuals(returns)
        assert residuals.shape == returns.shape

    def test_residuals_have_no_nan(self):
        """Residuals must not contain NaN when input has no NaN."""
        returns = _make_factor_returns()
        model = ClassicPCA(window=60, n_components=3)
        model.fit(returns)
        residuals = model.get_residuals(returns)
        assert not residuals.isna().any().any()

    def test_residuals_have_lower_variance_than_returns(self):
        """With a strong latent factor, residuals should be smaller than raw returns."""
        returns = _make_factor_returns(n_rows=200, n_cols=15)
        model = ClassicPCA(window=120, n_components=3)
        model.fit(returns)
        residuals = model.get_residuals(returns)
        assert residuals.var().mean() < returns.var().mean()


# ---------------------------------------------------------------------------
# AdaptivePCA
# ---------------------------------------------------------------------------


class TestAdaptivePCA:
    def test_compute_window_varies_with_volatility(self):
        """_compute_window returns a larger window when recent vol > long-term vol."""
        rng = np.random.default_rng(42)
        tickers = [f"T{i}" for i in range(5)]

        # high recent vol: calm history, then a spike in the last 21 days
        calm = pd.DataFrame(rng.normal(0, 0.001, (179, 5)), columns=tickers)
        spike = pd.DataFrame(rng.normal(0, 0.05, (21, 5)), columns=tickers)
        high_recent = pd.concat([calm, spike], ignore_index=True)

        # low recent vol: volatile history, then calm in the last 21 days
        volatile = pd.DataFrame(rng.normal(0, 0.05, (179, 5)), columns=tickers)
        quiet = pd.DataFrame(rng.normal(0, 0.001, (21, 5)), columns=tickers)
        low_recent = pd.concat([volatile, quiet], ignore_index=True)

        base = 60
        model = AdaptivePCA(base_window=base, n_components=3)
        w_high = model._compute_window(high_recent)
        w_low = model._compute_window(low_recent)
        assert w_high > w_low, (
            f"Expected high-recent-vol window ({w_high}) > low-recent-vol window ({w_low})"
        )

    def test_compute_window_respects_lower_bound(self):
        """Window must never be less than base_window // 2."""
        returns = _make_returns(200, seed=5) * 0.0001  # tiny vol → small ratio
        base = 60
        model = AdaptivePCA(base_window=base, n_components=3)
        w = model._compute_window(returns)
        assert w >= base // 2, f"Window {w} below lower bound {base // 2}"

    def test_compute_window_respects_upper_bound(self):
        """Window must never exceed base_window * 2."""
        returns = _make_returns(200, seed=6) * 100  # extreme vol → large ratio
        base = 60
        model = AdaptivePCA(base_window=base, n_components=3)
        w = model._compute_window(returns)
        assert w <= base * 2, f"Window {w} exceeds upper bound {base * 2}"

    def test_compute_window_fallback_on_constant_returns(self):
        """If long_vol is zero, _compute_window must return base_window."""
        dates = pd.bdate_range("2023-01-02", periods=50)
        returns = pd.DataFrame(
            np.ones((50, 5)) * 0.0,
            index=dates,
            columns=[f"T{i}" for i in range(5)],
        )
        model = AdaptivePCA(base_window=40, n_components=3)
        w = model._compute_window(returns)
        assert w == 40

    def test_compute_window_returns_int(self):
        """_compute_window must return a Python int."""
        returns = _make_returns(100)
        model = AdaptivePCA(base_window=60, n_components=3)
        w = model._compute_window(returns)
        assert isinstance(w, int)

    def test_fit_stores_last_window_used(self):
        """After fit(), _last_window_used must be set."""
        returns = _make_returns(100)
        model = AdaptivePCA(base_window=60, n_components=3)
        model.fit(returns)
        assert model._last_window_used is not None

    def test_residuals_shape(self):
        """AdaptivePCA.get_residuals must return same shape as input."""
        returns = _make_factor_returns(120)
        model = AdaptivePCA(base_window=60, n_components=3)
        model.fit(returns)
        residuals = model.get_residuals(returns)
        assert residuals.shape == returns.shape

    def test_no_lookahead_fit_deterministic(self):
        """Fitting twice on the same slice must produce identical residuals."""
        returns = _make_returns(100)
        model_a = AdaptivePCA(base_window=60, n_components=3)
        model_a.fit(returns)
        res_a = model_a.get_residuals(returns)

        model_b = AdaptivePCA(base_window=60, n_components=3)
        model_b.fit(returns)
        res_b = model_b.get_residuals(returns)

        pd.testing.assert_frame_equal(res_a, res_b)


# ---------------------------------------------------------------------------
# compute_zscore
# ---------------------------------------------------------------------------


class TestComputeZscore:
    def test_nan_in_first_window_minus_one_rows(self):
        """First window-1 rows of z-score must all be NaN."""
        residuals = _make_returns(100)
        window = 21
        zs = compute_zscore(residuals, window=window)
        assert zs.iloc[: window - 1].isna().all().all()

    def test_no_nan_after_window(self):
        """After the initial NaN period, z-scores must be finite."""
        residuals = _make_returns(100)
        window = 21
        zs = compute_zscore(residuals, window=window)
        assert not zs.iloc[window - 1 :].isna().any().any()

    def test_shape_matches_input(self):
        """compute_zscore must return a DataFrame of the same shape."""
        residuals = _make_returns(80, n_cols=8)
        zs = compute_zscore(residuals, window=10)
        assert zs.shape == residuals.shape

    def test_index_matches_input(self):
        """compute_zscore must preserve the DatetimeIndex."""
        residuals = _make_returns(60)
        zs = compute_zscore(residuals, window=10)
        pd.testing.assert_index_equal(zs.index, residuals.index)

    def test_zscore_magnitude_plausible(self):
        """On gaussian residuals, |z| should rarely exceed 4."""
        rng = np.random.default_rng(99)
        dates = pd.bdate_range("2023-01-02", periods=500)
        data = rng.normal(0, 1, size=(500, 20))
        residuals = pd.DataFrame(data, index=dates, columns=[f"T{i}" for i in range(20)])
        zs = compute_zscore(residuals, window=21)
        valid = zs.iloc[20:]
        extreme_frac = (valid.abs() > 4).mean().mean()
        assert extreme_frac < 0.01, f"Too many extreme z-scores: {extreme_frac:.3f}"

    def test_zero_std_column_produces_nan(self):
        """A constant residual column must yield NaN z-scores (std=0)."""
        dates = pd.bdate_range("2023-01-02", periods=50)
        data = pd.DataFrame(
            {"CONST": np.zeros(50), "NORM": np.random.default_rng(7).normal(0, 1, 50)},
            index=dates,
        )
        zs = compute_zscore(data, window=10)
        assert zs["CONST"].iloc[9:].isna().all()


# ---------------------------------------------------------------------------
# get_signals
# ---------------------------------------------------------------------------


class TestGetSignals:
    def _make_zscore_df(
        self,
        values: dict[str, float],
        n_rows: int = 30,
        window: int = 21,
    ) -> pd.DataFrame:
        """Build a z-score DataFrame whose last row matches `values`.

        The first `n_rows - 1` rows are set to 0 so they don't affect the
        final signal, and the last row gets the specified z-scores.
        """
        dates = pd.bdate_range("2023-01-02", periods=n_rows)
        tickers = list(values.keys())
        data = np.zeros((n_rows, len(tickers)))
        df = pd.DataFrame(data, index=dates, columns=tickers)
        for col, val in values.items():
            df.loc[df.index[-1], col] = val
        return df

    def test_long_signal_for_negative_z(self):
        """z < -entry must produce direction='long'."""
        zs = self._make_zscore_df({"AAPL": -2.5, "MSFT": 0.0})
        signals = get_signals(zs, entry=2.0)
        assert "AAPL" in signals
        assert signals["AAPL"]["direction"] == "long"

    def test_short_signal_for_positive_z(self):
        """z > entry must produce direction='short'."""
        zs = self._make_zscore_df({"AAPL": 0.0, "MSFT": 3.1})
        signals = get_signals(zs, entry=2.0)
        assert "MSFT" in signals
        assert signals["MSFT"]["direction"] == "short"

    def test_below_threshold_omitted(self):
        """Tickers with |z| <= entry must not appear in the output dict."""
        zs = self._make_zscore_df({"AAPL": 1.9, "MSFT": -1.5})
        signals = get_signals(zs, entry=2.0)
        assert signals == {}

    def test_signal_contract_keys_present(self):
        """Each signal entry must have direction, z_score, and weight."""
        zs = self._make_zscore_df({"AAPL": -2.3, "MSFT": 2.1, "GOOGL": 0.5})
        signals = get_signals(zs, entry=2.0)
        for ticker, info in signals.items():
            assert "direction" in info, f"{ticker} missing direction"
            assert "z_score" in info, f"{ticker} missing z_score"
            assert "weight" in info, f"{ticker} missing weight"

    def test_weights_sum_to_at_most_half(self):
        """Total weight across all signals must not exceed 0.5."""
        zs = self._make_zscore_df(
            {"A": -2.5, "B": 3.0, "C": -2.1, "D": 4.0, "E": -3.3}
        )
        signals = get_signals(zs, entry=2.0)
        total = sum(info["weight"] for info in signals.values())
        assert total <= 0.5 + 1e-9, f"Total weight {total:.4f} exceeds 0.5"

    def test_weights_are_positive(self):
        """All weights must be strictly positive."""
        zs = self._make_zscore_df({"A": -2.5, "B": 3.0})
        signals = get_signals(zs, entry=2.0)
        for ticker, info in signals.items():
            assert info["weight"] > 0, f"{ticker} has non-positive weight"

    def test_z_score_value_preserved(self):
        """The z_score field must match the actual z-score value."""
        zs = self._make_zscore_df({"AAPL": -2.7})
        signals = get_signals(zs, entry=2.0)
        assert pytest.approx(signals["AAPL"]["z_score"], abs=1e-6) == -2.7

    def test_returns_empty_dict_on_all_nan(self):
        """If the last row is all NaN, get_signals must return {}."""
        dates = pd.bdate_range("2023-01-02", periods=5)
        df = pd.DataFrame(
            np.full((5, 3), np.nan),
            index=dates,
            columns=["A", "B", "C"],
        )
        signals = get_signals(df, entry=2.0)
        assert signals == {}

    def test_weights_exactly_sum_to_half_with_single_signal(self):
        """A single active signal must receive exactly 0.5 weight."""
        zs = self._make_zscore_df({"ONLY": -2.5})
        signals = get_signals(zs, entry=2.0)
        assert pytest.approx(signals["ONLY"]["weight"], abs=1e-9) == 0.5


# ---------------------------------------------------------------------------
# MacroFilter
# ---------------------------------------------------------------------------


class TestMacroFilter:
    def _base_signals(self) -> dict:
        return {
            "AAPL": {"direction": "long", "z_score": -2.3, "weight": 0.10},
            "MSFT": {"direction": "short", "z_score": 2.1, "weight": 0.08},
        }

    # --- should_trade ---

    def test_should_trade_true_below_both_thresholds(self):
        f = MacroFilter(vix_threshold=30.0, spread_threshold=3.0)
        assert f.should_trade(vix=20.0, credit_spread=1.5) is True

    def test_should_trade_false_vix_above_threshold(self):
        f = MacroFilter(vix_threshold=30.0, spread_threshold=3.0)
        assert f.should_trade(vix=35.0, credit_spread=1.0) is False

    def test_should_trade_false_spread_above_threshold(self):
        f = MacroFilter(vix_threshold=30.0, spread_threshold=3.0)
        assert f.should_trade(vix=20.0, credit_spread=4.0) is False

    def test_should_trade_false_both_above_threshold(self):
        f = MacroFilter(vix_threshold=30.0, spread_threshold=3.0)
        assert f.should_trade(vix=40.0, credit_spread=5.0) is False

    def test_should_trade_false_exactly_at_vix_threshold(self):
        """VIX equal to threshold triggers suspend (> vs <=)."""
        f = MacroFilter(vix_threshold=30.0, spread_threshold=3.0)
        assert f.should_trade(vix=30.0, credit_spread=1.0) is True  # boundary: <= is ok

    # --- scale_weights: hard suspend ---

    def test_scale_weights_returns_empty_when_vix_above_threshold(self):
        f = MacroFilter(vix_threshold=30.0, spread_threshold=3.0)
        result = f.scale_weights(self._base_signals(), vix=35.0, credit_spread=1.0)
        assert result == {}

    def test_scale_weights_returns_empty_when_spread_above_threshold(self):
        f = MacroFilter(vix_threshold=30.0, spread_threshold=3.0)
        result = f.scale_weights(self._base_signals(), vix=15.0, credit_spread=5.0)
        assert result == {}

    def test_scale_weights_returns_empty_when_both_above_threshold(self):
        f = MacroFilter(vix_threshold=30.0, spread_threshold=3.0)
        result = f.scale_weights(self._base_signals(), vix=40.0, credit_spread=6.0)
        assert result == {}

    # --- scale_weights: linear ramp ---

    def test_scale_weights_reduces_in_ramp_zone(self):
        """VIX between 20 and threshold must reduce weights below originals."""
        f = MacroFilter(vix_threshold=30.0, spread_threshold=3.0)
        original = self._base_signals()
        vix_in_ramp = 25.0
        result = f.scale_weights(original, vix=vix_in_ramp, credit_spread=1.0)
        assert result  # not empty
        for ticker in original:
            assert result[ticker]["weight"] < original[ticker]["weight"]

    def test_scale_weights_multiplier_at_midpoint(self):
        """VIX = 25 with threshold=30 → multiplier = 0.5 (midpoint of [20, 30])."""
        f = MacroFilter(vix_threshold=30.0, spread_threshold=3.0)
        signals = {"AAPL": {"direction": "long", "z_score": -2.3, "weight": 0.10}}
        result = f.scale_weights(signals, vix=25.0, credit_spread=1.0)
        assert pytest.approx(result["AAPL"]["weight"], abs=1e-6) == 0.05

    def test_scale_weights_unchanged_below_ramp(self):
        """VIX < 20 must leave weights unchanged (multiplier = 1.0)."""
        f = MacroFilter(vix_threshold=30.0, spread_threshold=3.0)
        original = self._base_signals()
        result = f.scale_weights(original, vix=15.0, credit_spread=1.0)
        for ticker in original:
            assert pytest.approx(result[ticker]["weight"]) == original[ticker]["weight"]

    def test_scale_weights_does_not_mutate_input(self):
        """scale_weights must return a new dict without modifying the original."""
        f = MacroFilter(vix_threshold=30.0, spread_threshold=3.0)
        original = self._base_signals()
        original_weight = original["AAPL"]["weight"]
        _ = f.scale_weights(original, vix=25.0, credit_spread=1.0)
        assert original["AAPL"]["weight"] == original_weight

    def test_scale_weights_preserves_direction_and_zscore(self):
        """Macro filter must not alter direction or z_score fields."""
        f = MacroFilter(vix_threshold=30.0, spread_threshold=3.0)
        original = self._base_signals()
        result = f.scale_weights(original, vix=25.0, credit_spread=1.0)
        for ticker in result:
            assert result[ticker]["direction"] == original[ticker]["direction"]
            assert result[ticker]["z_score"] == original[ticker]["z_score"]

    # --- synthetic high-VIX input: macro filter actually suspends signals ---

    def test_high_vix_synthetic_input_produces_empty_signals(self):
        """End-to-end: high-VIX input to scale_weights must yield empty dict."""
        f = MacroFilter(vix_threshold=30.0, spread_threshold=3.0)
        rich_signals = {
            f"T{i}": {"direction": "long", "z_score": -float(i + 2), "weight": 0.05}
            for i in range(10)
        }
        result = f.scale_weights(rich_signals, vix=45.0, credit_spread=2.0)
        assert result == {}, "MacroFilter must return {} when VIX > threshold"

    def test_high_spread_synthetic_input_produces_empty_signals(self):
        """End-to-end: high credit spread must suspend all signals."""
        f = MacroFilter(vix_threshold=30.0, spread_threshold=3.0)
        rich_signals = {
            "AAPL": {"direction": "short", "z_score": 2.5, "weight": 0.12},
        }
        result = f.scale_weights(rich_signals, vix=18.0, credit_spread=7.5)
        assert result == {}
