"""Z-score computation and signal extraction from PCA residuals."""

import pandas as pd


def compute_zscore(residuals: pd.DataFrame, window: int = 21) -> pd.DataFrame:
    """Compute rolling z-scores from PCA residuals.

    For each ticker at each date J:
        z_J = (residual_J - mean(residuals_{J-window+1:J}))
              / std(residuals_{J-window+1:J})

    The first `window - 1` rows will be NaN because a full window is required.
    No look-ahead: pandas rolling with default closed='right' only uses past
    and current observations.

    Args:
        residuals: DataFrame of PCA residuals (index=DatetimeIndex, columns=tickers).
        window: Rolling lookback in trading days (default 21 ~ 1 month).

    Returns:
        DataFrame of z-scores, same shape as `residuals`. First `window - 1`
        rows are NaN.
    """
    roll = residuals.rolling(window=window, min_periods=window)
    mu = roll.mean()
    sigma = roll.std(ddof=1)
    zscore = (residuals - mu) / sigma
    return zscore


def get_signals(
    zscore: pd.DataFrame,
    entry: float = 2.0,
    exit: float = 0.5,
) -> dict[str, dict[str, float | str]]:
    """Extract trading signals from the last row of a z-score DataFrame.

    Signal logic (applied to the most recent date only):
    - |z| > entry  →  position opened
        - z < -entry  →  direction = "long"   (ticker is cheap vs factor)
        - z >  entry  →  direction = "short"  (ticker is expensive vs factor)
    - |z| <= entry (including |z| < exit region) → ticker omitted from output

    Weight allocation:
        weight_i = |z_i| / sum(|z_active|) * 0.5

    Allocating 50 % of portfolio across active names ensures the book stays
    roughly half-invested; position sizing up to the 5-10 % cap per name is
    enforced by the `risk/` module, not here.

    Args:
        zscore: Rolling z-score DataFrame (index=DatetimeIndex, columns=tickers).
        entry: Absolute z-score threshold to open a position (default 2.0).
        exit: Absolute z-score threshold to close — tickers below this are
            excluded from the output. Tickers between `exit` and `entry` that
            are already open are handled by the caller (backtest/execution).

    Returns:
        Signal contract dict containing only tickers with |z| > entry:
        ``{"TICKER": {"direction": "long"|"short", "z_score": float, "weight": float}}``
        Returns an empty dict if no z-scores exceed the entry threshold or if
        the last row is entirely NaN.
    """
    last = zscore.iloc[-1].dropna()

    active = last[last.abs() > entry]
    if active.empty:
        return {}

    total_abs = active.abs().sum()

    signals: dict[str, dict[str, float | str]] = {}
    for ticker, z in active.items():
        direction = "long" if z < 0 else "short"
        weight = float(abs(z) / total_abs * 0.5)
        signals[str(ticker)] = {
            "direction": direction,
            "z_score": float(z),
            "weight": weight,
        }

    return signals
