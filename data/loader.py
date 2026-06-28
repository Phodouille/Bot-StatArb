"""Load and persist OHLCV price data; compute log-returns."""

import logging
import os
from datetime import date

import numpy as np
import pandas as pd

from data.cleaner import clean_bars
from data.fetcher import AlpacaFetcher

logger = logging.getLogger(__name__)

_DEFAULT_PARQUET = "db/prices.parquet"


def load_returns(
    tickers: list[str],
    start: date,
    end: date,
    parquet_path: str = _DEFAULT_PARQUET,
) -> pd.DataFrame:
    """Load adjusted log-returns from persisted storage.

    This is the public data contract consumed by signals/, backtest/, and risk/.
    Its signature and return shape are stable — do not change without team
    alignment and updating CLAUDE.md.

    Args:
        tickers: Ticker symbols to include.
        start: Inclusive start date.
        end: Inclusive end date.  No row dated after this date will be returned
            (no look-ahead guarantee).
        parquet_path: Path to the long-format parquet store.

    Returns:
        Wide DataFrame: index=DatetimeIndex (trading days), columns=ticker,
        values=log-return (float64).  Never contains NaN.  Tickers that fail
        the missing-data threshold inside the requested window are dropped and
        logged.  The DataFrame columns are the *surviving* tickers — callers
        must check which tickers are present.

    Raises:
        FileNotFoundError: If parquet_path does not exist.
        ValueError: If the resulting DataFrame is empty after cleaning.
    """
    if not os.path.exists(parquet_path):
        raise FileNotFoundError(
            f"Price store not found at {parquet_path!r}. "
            "Run fetch_and_store() first to populate it."
        )

    raw = pd.read_parquet(parquet_path)

    raw["date"] = pd.to_datetime(raw["date"], utc=True)

    start_ts = pd.Timestamp(start, tz="UTC")
    end_ts = pd.Timestamp(end, tz="UTC")

    mask = (
        raw["ticker"].isin(tickers)
        & (raw["date"] >= start_ts)
        & (raw["date"] <= end_ts)
    )
    subset = raw.loc[mask].copy()

    if subset.empty:
        logger.warning(
            "No data found for tickers=%s between %s and %s.", tickers, start, end
        )
        return pd.DataFrame()

    cleaned = clean_bars(subset)

    if cleaned.empty:
        raise ValueError(
            f"All tickers were excluded by clean_bars for the window {start} to {end}."
        )

    cleaned["date"] = pd.to_datetime(cleaned["date"], utc=True)

    cleaned = cleaned[cleaned["date"] <= end_ts]

    wide_close = (
        cleaned.pivot(index="date", columns="ticker", values="close")
        .sort_index()
    )

    log_returns = np.log(wide_close / wide_close.shift(1))

    log_returns = log_returns.iloc[1:]

    log_returns = log_returns[log_returns.index <= end_ts]

    cols_before = set(log_returns.columns)
    log_returns = log_returns.dropna(axis=1, how="any")
    cols_after = set(log_returns.columns)
    dropped = cols_before - cols_after
    if dropped:
        logger.warning(
            "Tickers dropped due to remaining NaN after forward-fill: %s", sorted(dropped)
        )

    log_returns = log_returns.dropna(axis=0, how="any")

    assert log_returns.isna().sum().sum() == 0, "BUG: NaN leaked into load_returns output."

    if log_returns.empty:
        raise ValueError(
            f"Empty returns matrix after cleaning for window {start} to {end}."
        )

    logger.info(
        "load_returns: returned %d rows x %d tickers for window %s to %s.",
        len(log_returns),
        len(log_returns.columns),
        start,
        end,
    )
    return log_returns


def fetch_and_store(
    tickers: list[str],
    start: date,
    end: date,
    parquet_path: str = _DEFAULT_PARQUET,
) -> None:
    """Fetch bars from Alpaca, clean, and persist to parquet (incremental).

    If the parquet file already exists, new data is appended and duplicates on
    (date, ticker) are dropped so that re-running the function is idempotent.

    Args:
        tickers: Ticker symbols to fetch.
        start: Inclusive start date for the fetch request.
        end: Inclusive end date for the fetch request.
        parquet_path: Destination parquet file path.
    """
    fetcher = AlpacaFetcher()
    new_data = fetcher.fetch_bars(tickers=tickers, start=start, end=end)

    if new_data.empty:
        logger.warning("fetch_and_store: AlpacaFetcher returned no data; nothing stored.")
        return

    cleaned = clean_bars(new_data)

    if os.path.exists(parquet_path):
        existing = pd.read_parquet(parquet_path)
        combined = pd.concat([existing, cleaned], ignore_index=True)
    else:
        os.makedirs(os.path.dirname(parquet_path) if os.path.dirname(parquet_path) else ".", exist_ok=True)
        combined = cleaned

    combined["date"] = pd.to_datetime(combined["date"], utc=True)
    combined = (
        combined
        .drop_duplicates(subset=["date", "ticker"], keep="last")
        .sort_values(["ticker", "date"])
        .reset_index(drop=True)
    )

    combined.to_parquet(parquet_path, index=False)
    logger.info(
        "fetch_and_store: stored %d rows (%d tickers) to %s.",
        len(combined),
        combined["ticker"].nunique(),
        parquet_path,
    )
