"""Tests for data/cleaner.py and data/loader.py.

All tests are fully offline — no Alpaca API calls are made.
"""

import os
import tempfile
from datetime import date
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from data.cleaner import clean_bars
from data.loader import fetch_and_store, load_returns


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_long_df(
    tickers: list[str],
    dates: list[pd.Timestamp],
    base_price: float = 100.0,
) -> pd.DataFrame:
    """Build a synthetic long-format OHLCV DataFrame."""
    rows = []
    for ticker in tickers:
        price = base_price
        for dt in dates:
            price *= 1 + np.random.default_rng(abs(hash(ticker + str(dt)))).normal(0, 0.01)
            rows.append(
                {
                    "date": dt,
                    "ticker": ticker,
                    "open": price * 0.99,
                    "high": price * 1.01,
                    "low": price * 0.98,
                    "close": price,
                    "volume": 1_000_000.0,
                }
            )
    return pd.DataFrame(rows)


def _make_trading_dates(start: str, end: str) -> list[pd.Timestamp]:
    """Return weekday timestamps (simplified trading calendar)."""
    return [
        ts
        for ts in pd.date_range(start, end, freq="B", tz="UTC")
    ]


# ---------------------------------------------------------------------------
# clean_bars tests
# ---------------------------------------------------------------------------


class TestCleanBars:
    def test_returns_all_tickers_when_no_gaps(self) -> None:
        dates = _make_trading_dates("2024-01-02", "2024-03-31")
        df = _make_long_df(["AAPL", "MSFT"], dates)
        result = clean_bars(df)
        assert set(result["ticker"].unique()) == {"AAPL", "MSFT"}

    def test_forward_fill_small_gap(self) -> None:
        # Use a long date range so 2 missing days stay well under the 5% threshold.
        # 2 days missing out of ~65 weekdays ≈ 3% < 5% threshold.
        dates = _make_trading_dates("2024-01-02", "2024-03-31")
        df = _make_long_df(["AAPL"], dates)

        # Remove 2 consecutive dates for AAPL (within max_fill_days=2).
        missing = dates[3:5]
        df_with_gap = df[~df["date"].isin(missing)].copy()

        result = clean_bars(df_with_gap, max_gap_fraction=0.05, max_fill_days=2)
        aapl = result[result["ticker"] == "AAPL"].set_index("date")

        assert "AAPL" in result["ticker"].values, "AAPL was incorrectly excluded"

        # All original dates should be present after fill.
        for dt in dates:
            assert dt in aapl.index, f"{dt} missing after forward-fill"

        # Forward-fill: missing days should have the same close as the day before.
        prev_close = df.set_index("date").loc[dates[2], "close"]
        for dt in missing:
            assert aapl.loc[dt, "close"] == pytest.approx(prev_close)

    def test_excludes_ticker_with_large_gap(self) -> None:
        # 5 missing days out of ~23 ≈ 21.7% > 5% threshold → ticker excluded.
        dates = _make_trading_dates("2024-01-02", "2024-01-31")
        df = _make_long_df(["AAPL"], dates)

        missing = dates[3:8]
        df_with_gap = df[~df["date"].isin(missing)].copy()

        result = clean_bars(df_with_gap, max_gap_fraction=0.05, max_fill_days=2)
        assert "AAPL" not in result["ticker"].values

    def test_excludes_ticker_above_gap_threshold(self) -> None:
        dates = _make_trading_dates("2024-01-02", "2024-03-29")
        df_good = _make_long_df(["MSFT"], dates)

        # AAPL is missing 40% of dates — well above 5%.
        n_missing = int(len(dates) * 0.40)
        kept_dates = dates[n_missing:]
        df_bad = _make_long_df(["AAPL"], kept_dates)

        df = pd.concat([df_good, df_bad], ignore_index=True)
        result = clean_bars(df, max_gap_fraction=0.05)

        assert "MSFT" in result["ticker"].values
        assert "AAPL" not in result["ticker"].values

    def test_returns_empty_df_when_all_tickers_excluded(self) -> None:
        # AAPL missing 8 out of 23 business days (35% > 5% threshold) → excluded.
        dates = _make_trading_dates("2024-01-02", "2024-01-31")
        df = _make_long_df(["AAPL"], dates)
        missing = dates[3:11]  # 8 consecutive days removed
        df_with_gap = df[~df["date"].isin(missing)].copy()
        result = clean_bars(df_with_gap, max_gap_fraction=0.05)
        assert result.empty

    def test_output_schema(self) -> None:
        dates = _make_trading_dates("2024-01-02", "2024-02-29")
        df = _make_long_df(["AAPL", "NVDA"], dates)
        result = clean_bars(df)
        assert list(result.columns) == ["date", "ticker", "open", "high", "low", "close", "volume"]

    def test_empty_input_returns_empty(self) -> None:
        df = pd.DataFrame(columns=["date", "ticker", "open", "high", "low", "close", "volume"])
        result = clean_bars(df)
        assert result.empty


# ---------------------------------------------------------------------------
# load_returns tests
# ---------------------------------------------------------------------------


def _write_parquet(df: pd.DataFrame, path: str) -> None:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], utc=True)
    df.to_parquet(path, index=False)


class TestLoadReturns:
    def test_no_nan_in_output(self) -> None:
        dates = _make_trading_dates("2024-01-02", "2024-06-28")
        df = _make_long_df(["AAPL", "MSFT", "NVDA"], dates)

        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
            path = f.name
        try:
            _write_parquet(df, path)
            result = load_returns(
                tickers=["AAPL", "MSFT", "NVDA"],
                start=date(2024, 1, 2),
                end=date(2024, 6, 28),
                parquet_path=path,
            )
            assert result.isna().sum().sum() == 0
        finally:
            os.unlink(path)

    def test_no_dates_after_end(self) -> None:
        dates = _make_trading_dates("2024-01-02", "2024-12-31")
        df = _make_long_df(["AAPL", "MSFT"], dates)

        end = date(2024, 6, 30)

        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
            path = f.name
        try:
            _write_parquet(df, path)
            result = load_returns(
                tickers=["AAPL", "MSFT"],
                start=date(2024, 1, 2),
                end=end,
                parquet_path=path,
            )
            end_ts = pd.Timestamp(end, tz="UTC")
            assert (result.index <= end_ts).all(), (
                "load_returns returned rows after the requested end date."
            )
        finally:
            os.unlink(path)

    def test_output_is_wide_format(self) -> None:
        dates = _make_trading_dates("2024-01-02", "2024-03-29")
        df = _make_long_df(["AAPL", "MSFT"], dates)

        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
            path = f.name
        try:
            _write_parquet(df, path)
            result = load_returns(
                tickers=["AAPL", "MSFT"],
                start=date(2024, 1, 2),
                end=date(2024, 3, 29),
                parquet_path=path,
            )
            assert "AAPL" in result.columns
            assert "MSFT" in result.columns
            assert result.index.name == "date" or isinstance(result.index, pd.DatetimeIndex)
        finally:
            os.unlink(path)

    def test_excludes_high_gap_ticker_from_returns(self) -> None:
        dates = _make_trading_dates("2024-01-02", "2024-06-28")
        df_good = _make_long_df(["MSFT"], dates)

        # NVDA only has data for the last 30% of the window.
        n = int(len(dates) * 0.70)
        df_bad = _make_long_df(["NVDA"], dates[n:])

        df = pd.concat([df_good, df_bad], ignore_index=True)

        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
            path = f.name
        try:
            _write_parquet(df, path)
            result = load_returns(
                tickers=["MSFT", "NVDA"],
                start=date(2024, 1, 2),
                end=date(2024, 6, 28),
                parquet_path=path,
            )
            assert "MSFT" in result.columns
            assert "NVDA" not in result.columns
        finally:
            os.unlink(path)

    def test_raises_if_parquet_missing(self) -> None:
        with pytest.raises(FileNotFoundError):
            load_returns(
                tickers=["AAPL"],
                start=date(2024, 1, 2),
                end=date(2024, 6, 28),
                parquet_path="/nonexistent/path/prices.parquet",
            )

    def test_returns_are_log_returns(self) -> None:
        """Verify the arithmetic: log(close_t / close_{t-1})."""
        dates = _make_trading_dates("2024-01-02", "2024-01-31")
        prices = [100.0 * (1.01**i) for i in range(len(dates))]
        rows = [
            {
                "date": dt,
                "ticker": "AAPL",
                "open": p,
                "high": p,
                "low": p,
                "close": p,
                "volume": 1e6,
            }
            for dt, p in zip(dates, prices)
        ]
        df = pd.DataFrame(rows)

        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
            path = f.name
        try:
            _write_parquet(df, path)
            result = load_returns(
                tickers=["AAPL"],
                start=date(2024, 1, 2),
                end=date(2024, 1, 31),
                parquet_path=path,
            )
            expected_return = np.log(1.01)
            assert result["AAPL"].iloc[0] == pytest.approx(expected_return, rel=1e-6)
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# fetch_and_store tests (Alpaca mocked)
# ---------------------------------------------------------------------------


class TestFetchAndStore:
    def _make_fetcher_mock(self, df: pd.DataFrame) -> MagicMock:
        mock_fetcher = MagicMock()
        mock_fetcher.fetch_bars.return_value = df
        return mock_fetcher

    def test_creates_parquet_on_first_run(self) -> None:
        dates = _make_trading_dates("2024-01-02", "2024-03-29")
        df = _make_long_df(["AAPL", "MSFT"], dates)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "prices.parquet")
            with patch("data.loader.AlpacaFetcher") as MockFetcher:
                MockFetcher.return_value = self._make_fetcher_mock(df)
                fetch_and_store(
                    tickers=["AAPL", "MSFT"],
                    start=date(2024, 1, 2),
                    end=date(2024, 3, 29),
                    parquet_path=path,
                )
            assert os.path.exists(path)
            stored = pd.read_parquet(path)
            assert set(stored["ticker"].unique()) == {"AAPL", "MSFT"}

    def test_incremental_append_deduplicates(self) -> None:
        dates_a = _make_trading_dates("2024-01-02", "2024-01-31")
        dates_b = _make_trading_dates("2024-01-15", "2024-02-29")  # overlap
        df_a = _make_long_df(["AAPL"], dates_a)
        df_b = _make_long_df(["AAPL"], dates_b)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "prices.parquet")

            with patch("data.loader.AlpacaFetcher") as MockFetcher:
                MockFetcher.return_value = self._make_fetcher_mock(df_a)
                fetch_and_store(["AAPL"], date(2024, 1, 2), date(2024, 1, 31), parquet_path=path)

            with patch("data.loader.AlpacaFetcher") as MockFetcher:
                MockFetcher.return_value = self._make_fetcher_mock(df_b)
                fetch_and_store(["AAPL"], date(2024, 1, 15), date(2024, 2, 29), parquet_path=path)

            stored = pd.read_parquet(path)
            aapl = stored[stored["ticker"] == "AAPL"]
            dupes = aapl.duplicated(subset=["date", "ticker"]).sum()
            assert dupes == 0, f"Duplicates found after incremental append: {dupes}"

    def test_does_nothing_when_fetcher_returns_empty(self) -> None:
        empty_df = pd.DataFrame(
            columns=["date", "ticker", "open", "high", "low", "close", "volume"]
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "prices.parquet")
            with patch("data.loader.AlpacaFetcher") as MockFetcher:
                MockFetcher.return_value = self._make_fetcher_mock(empty_df)
                fetch_and_store(["AAPL"], date(2024, 1, 2), date(2024, 1, 31), parquet_path=path)
            assert not os.path.exists(path)
