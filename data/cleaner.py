"""Cleaning utilities for raw OHLCV bar DataFrames."""

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def clean_bars(
    df: pd.DataFrame,
    max_gap_fraction: float = 0.05,
    max_fill_days: int = 2,
) -> pd.DataFrame:
    """Clean a long-format OHLCV DataFrame.

    Operates on the union of trading dates observed across all tickers so that
    the cleaning step is calendar-aware without requiring an external calendar
    dependency.

    Args:
        df: Long-format DataFrame with columns: date, ticker, open, high, low,
            close, volume.  ``date`` must be datetime64.
        max_gap_fraction: Maximum tolerated fraction of missing trading dates
            per ticker. Tickers exceeding this threshold are dropped entirely.
        max_fill_days: Maximum number of consecutive missing days to fill via
            forward-fill. Gaps longer than this are left as NaN (and will
            eventually trigger exclusion via the fraction threshold or downstream
            dropna).

    Returns:
        Cleaned long-format DataFrame containing only valid tickers, with small
        gaps forward-filled and the same column schema as the input.
        Sorted by (ticker, date).
    """
    if df.empty:
        return df.copy()

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], utc=True)

    min_date = df["date"].min()
    max_date = df["date"].max()
    all_dates = pd.bdate_range(start=min_date, end=max_date, tz="UTC")
    n_dates = len(all_dates)

    tickers = df["ticker"].unique().tolist()
    cleaned_frames: list[pd.DataFrame] = []

    for ticker in tickers:
        subset = df[df["ticker"] == ticker].set_index("date").sort_index()

        missing_dates = all_dates.difference(subset.index)
        missing_fraction = len(missing_dates) / n_dates if n_dates > 0 else 0.0

        if missing_fraction > max_gap_fraction:
            logger.warning(
                "Excluding ticker %s: %.1f%% missing dates (threshold %.1f%%).",
                ticker,
                missing_fraction * 100,
                max_gap_fraction * 100,
            )
            continue

        if len(missing_dates) > 0:
            empty_rows = pd.DataFrame(
                {col: pd.array([None] * len(missing_dates), dtype=object) for col in subset.columns},
                index=missing_dates,
            )
            empty_rows["ticker"] = ticker
            subset = pd.concat([subset, empty_rows]).sort_index()
            subset["ticker"] = ticker

        subset = _forward_fill_limited(subset, max_fill_days=max_fill_days)

        subset.index.name = "date"
        subset = subset.reset_index()
        subset["date"] = pd.to_datetime(subset["date"], utc=True)

        cols = ["date", "ticker", "open", "high", "low", "close", "volume"]
        cleaned_frames.append(subset[cols])

    if not cleaned_frames:
        logger.warning("clean_bars: all tickers were excluded.")
        return pd.DataFrame(columns=["date", "ticker", "open", "high", "low", "close", "volume"])

    result = pd.concat(cleaned_frames, ignore_index=True)
    result = result.sort_values(["ticker", "date"]).reset_index(drop=True)
    return result


def _forward_fill_limited(df: pd.DataFrame, max_fill_days: int) -> pd.DataFrame:
    """Forward-fill numeric columns up to max_fill_days consecutive NaN rows.

    Args:
        df: DataFrame indexed by date, sorted ascending.
        max_fill_days: Maximum consecutive NaN rows to fill.

    Returns:
        DataFrame with small gaps filled; larger gaps remain NaN.
    """
    numeric_cols = ["open", "high", "low", "close", "volume"]
    existing_numeric = [c for c in numeric_cols if c in df.columns]

    for col in existing_numeric:
        series = df[col].copy()
        filled = series.ffill(limit=max_fill_days)
        df[col] = filled

    return df
