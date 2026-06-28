"""CLI to incrementally refresh the price store.

Usage:
    python -m data.refresh [--start YYYY-MM-DD] [--end YYYY-MM-DD]
                           [--parquet-path PATH]

By default, fetches from the day after the latest stored date up to yesterday
(incremental update). Pass --start and --end to override.
"""

import argparse
import logging
import os
import sys
from datetime import date, timedelta

import pandas as pd

from config.universe import TICKERS
from data.loader import fetch_and_store

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

_DEFAULT_PARQUET = "db/prices.parquet"
_HISTORY_START = date(2020, 1, 2)


def _latest_stored_date(parquet_path: str) -> date | None:
    """Return the most recent date present in the parquet store, or None."""
    if not os.path.exists(parquet_path):
        return None
    df = pd.read_parquet(parquet_path, columns=["date"])
    if df.empty:
        return None
    return pd.to_datetime(df["date"]).max().date()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Refresh the price parquet store incrementally.")
    parser.add_argument("--start", type=date.fromisoformat, default=None,
                        help="Start date (YYYY-MM-DD). Defaults to day after latest stored date.")
    parser.add_argument("--end", type=date.fromisoformat, default=None,
                        help="End date (YYYY-MM-DD). Defaults to yesterday.")
    parser.add_argument("--parquet-path", default=_DEFAULT_PARQUET,
                        help=f"Path to parquet store (default: {_DEFAULT_PARQUET}).")

    args = parser.parse_args(argv)

    end = args.end or (date.today() - timedelta(days=1))

    if args.start:
        start = args.start
    else:
        latest = _latest_stored_date(args.parquet_path)
        if latest is None:
            start = _HISTORY_START
            logger.info("No existing store found — fetching full history from %s.", start)
        else:
            start = latest + timedelta(days=1)
            logger.info("Incremental update: latest stored date is %s, fetching from %s.", latest, start)

    if start > end:
        logger.info("Store is already up to date (start=%s > end=%s). Nothing to fetch.", start, end)
        return 0

    logger.info("Fetching %d tickers from %s to %s into %s.", len(TICKERS), start, end, args.parquet_path)
    fetch_and_store(tickers=TICKERS, start=start, end=end, parquet_path=args.parquet_path)
    logger.info("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
