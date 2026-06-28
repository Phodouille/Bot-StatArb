"""Main entry point for signal generation.

Composes ClassicPCA / AdaptivePCA, z-score computation, and the MacroFilter
according to the benchmark level requested by the caller.

Benchmark levels handled here:
    3 — ClassicPCA + rolling z-score + threshold filter (no macro filter)
    4 — AdaptivePCA + rolling z-score + threshold filter + MacroFilter

Levels 1 and 2 (buy-and-hold, equal-weight sector) are implemented in
backtest/ and do not require signal generation.
"""

import pandas as pd

try:
    from data import load_returns  # noqa: F401 — imported for live use
except ImportError:
    load_returns = None  # data module not yet available; use injected returns

from signals.filters import MacroFilter
from signals.pca import AdaptivePCA, ClassicPCA
from signals.zscore import compute_zscore, get_signals


def generate_signals(
    returns: pd.DataFrame,
    level: int = 4,
    window: int = 60,
    n_components: int = 5,
    zscore_window: int = 21,
    entry_threshold: float = 2.0,
    exit_threshold: float = 0.5,
    vix: float | None = None,
    credit_spread: float | None = None,
    vix_threshold: float = 30.0,
    spread_threshold: float = 3.0,
    vol_lookback: int = 21,
) -> dict[str, dict]:
    """Generate trading signals from a log-return matrix.

    No look-ahead guarantee: all PCA fits, z-score computations, and
    volatility estimates use only data up to and including the last row of
    `returns`. The caller is responsible for passing a slice that ends at
    date J when computing signals for J.

    Args:
        returns: Wide log-return matrix (index=DatetimeIndex, columns=tickers).
            Produced by data.load_returns(); must contain no NaN values.
        level: Benchmark level.
            3 → ClassicPCA, no macro filter.
            4 → AdaptivePCA + MacroFilter (requires vix and credit_spread).
        window: PCA fitting window in trading days (ClassicPCA fixed window or
            AdaptivePCA base_window). Configurable; tested range 30-90.
        n_components: Number of principal components to retain.
        zscore_window: Rolling window for z-score normalization (days).
        entry_threshold: |z| must exceed this to open a new position (default 2.0).
        exit_threshold: |z| below this signals position closure. Currently used
            as a documentation anchor; threshold crossing for existing positions
            is enforced in execution/, not here.
        vix: Current VIX level. Required for level=4; ignored for level=3.
        credit_spread: Current IG/HY credit spread (pp). Required for level=4.
        vix_threshold: Hard suspend threshold for VIX (MacroFilter).
        spread_threshold: Hard suspend threshold for credit spread (MacroFilter).
        vol_lookback: Days for recent-vol estimation in AdaptivePCA.

    Returns:
        Signal contract dict per CONTRAT 1:
        ``{"TICKER": {"direction": "long"|"short", "z_score": float, "weight": float}}``
        Only tickers with a desired position are included. Empty dict if no
        signals exceed the entry threshold or if the macro filter suspends trading.

    Raises:
        ValueError: If `level` is not 3 or 4, or if `returns` is too short to
            fit PCA with the given parameters.
    """
    if level not in (3, 4):
        raise ValueError(f"generate_signals supports level 3 or 4; got {level}.")

    if len(returns) < n_components:
        raise ValueError(
            f"Insufficient history: need at least {n_components} rows, "
            f"got {len(returns)}."
        )

    if level == 3:
        model: ClassicPCA = ClassicPCA(window=window, n_components=n_components)
    else:
        model = AdaptivePCA(
            base_window=window,
            n_components=n_components,
            vol_lookback=vol_lookback,
        )

    model.fit(returns)
    residuals = model.get_residuals(returns)

    zscore = compute_zscore(residuals, window=zscore_window)
    raw_signals = get_signals(zscore, entry=entry_threshold, exit=exit_threshold)

    if level == 4:
        if vix is None or credit_spread is None:
            raise ValueError(
                "level=4 requires `vix` and `credit_spread` to be provided."
            )
        macro = MacroFilter(
            vix_threshold=vix_threshold,
            spread_threshold=spread_threshold,
        )
        return macro.scale_weights(raw_signals, vix=vix, credit_spread=credit_spread)

    return raw_signals
