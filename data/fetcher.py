"""Alpaca market data fetcher for OHLCV bars."""

import logging
import os
from datetime import date, datetime, timezone

import pandas as pd
from alpaca.data import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


class AlpacaFetcher:
    """Fetches historical OHLCV bars from the Alpaca market data API.

    Credentials are loaded exclusively from environment variables
    (ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_BASE_URL). Never hardcode keys.
    """

    def __init__(self) -> None:
        """Initialise the Alpaca client from environment variables.

        Raises:
            EnvironmentError: If ALPACA_API_KEY or ALPACA_SECRET_KEY are missing.
        """
        api_key = os.environ.get("ALPACA_API_KEY")
        secret_key = os.environ.get("ALPACA_SECRET_KEY")
        if not api_key or not secret_key:
            raise EnvironmentError(
                "ALPACA_API_KEY and ALPACA_SECRET_KEY must be set in the environment "
                "(load from .env, never hardcode)."
            )
        self._client = StockHistoricalDataClient(api_key=api_key, secret_key=secret_key)

    def fetch_bars(
        self,
        tickers: list[str],
        start: date,
        end: date,
        timeframe: str = "1Day",
    ) -> pd.DataFrame:
        """Fetch daily OHLCV bars from Alpaca for the given tickers and date range.

        Args:
            tickers: List of uppercase ticker symbols.
            start: Inclusive start date (calendar date, not datetime).
            end: Inclusive end date (calendar date, not datetime).
            timeframe: Alpaca timeframe string; only "1Day" is supported for now.

        Returns:
            DataFrame in long format with columns:
                date (datetime64[ns, UTC] normalised to midnight UTC),
                ticker (str), open (float), high (float), low (float),
                close (float), volume (float).
            Sorted by (ticker, date). May contain gaps for non-trading days.

        Raises:
            ValueError: If timeframe is not "1Day".
            RuntimeError: If the Alpaca API call fails.
        """
        if timeframe != "1Day":
            raise ValueError(f"Unsupported timeframe: {timeframe!r}. Only '1Day' is supported.")

        tf = TimeFrame(1, TimeFrameUnit.Day)

        start_dt = datetime(start.year, start.month, start.day, tzinfo=timezone.utc)
        end_dt = datetime(end.year, end.month, end.day, 23, 59, 59, tzinfo=timezone.utc)

        request = StockBarsRequest(
            symbol_or_symbols=tickers,
            timeframe=tf,
            start=start_dt,
            end=end_dt,
            adjustment="all",
        )

        logger.info(
            "Fetching bars for %d tickers from %s to %s", len(tickers), start, end
        )

        bars = self._client.get_stock_bars(request)
        raw_df = bars.df

        if raw_df.empty:
            logger.warning("Alpaca returned no data for tickers=%s, start=%s, end=%s", tickers, start, end)
            return pd.DataFrame(columns=["date", "ticker", "open", "high", "low", "close", "volume"])

        df = raw_df.reset_index()

        symbol_col = "symbol" if "symbol" in df.columns else df.columns[1]
        timestamp_col = "timestamp" if "timestamp" in df.columns else df.columns[0]

        df = df.rename(columns={symbol_col: "ticker", timestamp_col: "date"})

        df["date"] = pd.to_datetime(df["date"], utc=True).dt.normalize()

        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = df[col].astype(float)

        df = df[["date", "ticker", "open", "high", "low", "close", "volume"]]
        df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

        logger.info("Fetched %d rows for %d tickers", len(df), df["ticker"].nunique())
        return df
