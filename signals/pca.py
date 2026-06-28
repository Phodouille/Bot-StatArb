"""PCA-based factor models for statistical arbitrage signal generation.

Two variants:
- ClassicPCA: fixed rolling window
- AdaptivePCA: window length adjusts to realized volatility
"""

import logging

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

logger = logging.getLogger(__name__)


class ClassicPCA:
    """Rolling PCA with a fixed lookback window.

    Args:
        window: Number of trading days used to fit PCA. Configurable; tested
            range 30-90 (default 60 for the class, overridden by generator).
        n_components: Number of principal components to retain.
    """

    def __init__(self, window: int = 60, n_components: int = 5) -> None:
        self.window = window
        self.n_components = n_components
        self._pca: PCA | None = None
        self._fitted_mean: pd.Series | None = None
        self._columns: list[str] = []

    def fit(self, returns: pd.DataFrame) -> None:
        """Fit PCA on the most recent `window` rows of returns.

        Uses only returns.iloc[-window:] — no look-ahead. The caller must
        ensure that `returns` ends at date J when computing signals for J.

        Args:
            returns: Wide log-return matrix (index=DatetimeIndex, columns=tickers).

        Raises:
            ValueError: If the DataFrame has fewer rows than n_components.
        """
        if len(returns) < self.n_components:
            raise ValueError(
                f"Need at least {self.n_components} rows to fit PCA; "
                f"got {len(returns)}."
            )

        effective = min(self.window, len(returns))
        self._fit_on_slice(returns.iloc[-effective:])

    def get_residuals(self, returns: pd.DataFrame) -> pd.DataFrame:
        """Compute residuals: actual returns minus PCA reconstruction.

        The PCA loadings are fixed at fit time, so applying them here does not
        introduce look-ahead regardless of the index of `returns`.

        Args:
            returns: Wide log-return matrix, same columns as used in fit().

        Returns:
            DataFrame of residuals with the same shape and index as `returns`.

        Raises:
            RuntimeError: If called before fit().
        """
        if self._pca is None or self._fitted_mean is None:
            raise RuntimeError("Call fit() before get_residuals().")

        centered = (returns - self._fitted_mean).values
        scores = centered @ self._pca.components_.T
        reconstruction = scores @ self._pca.components_
        residuals_values = centered - reconstruction

        return pd.DataFrame(
            residuals_values,
            index=returns.index,
            columns=returns.columns,
        )

    def _fit_on_slice(self, slice_: pd.DataFrame) -> None:
        """Fit PCA on an already-sliced DataFrame (internal, no further windowing).

        Args:
            slice_: Contiguous sub-DataFrame to fit on.
        """
        self._fitted_mean = slice_.mean()
        centered = slice_ - self._fitted_mean
        self._pca = PCA(n_components=self.n_components)
        self._pca.fit(centered.values)
        self._columns = slice_.columns.tolist()


class AdaptivePCA(ClassicPCA):
    """Rolling PCA with a volatility-adaptive window.

    Adaptive rule
    -------------
    Cross-sectional mean absolute return is used as a market-vol proxy.

        recent_vol = annualised std of the last `vol_lookback` daily values
        long_vol   = annualised std over all available history

        ratio  = recent_vol / long_vol
        window = clamp(int(base_window * ratio),
                       base_window // 2,
                       base_window * 2)

    High vol (ratio > 1) → longer window: more samples stabilise noisy
    loadings when individual return variance is elevated.
    Low vol (ratio < 1) → shorter window: faster adaptation to structural
    shifts when the market is calm.

    If long_vol is zero or undefined (< 2 rows), the base_window is used.

    Args:
        base_window: Central window length around which the rule pivots.
        n_components: Number of principal components.
        vol_lookback: Days used to compute recent_vol (default 21 ~ 1 month).
    """

    def __init__(
        self,
        base_window: int = 60,
        n_components: int = 5,
        vol_lookback: int = 21,
    ) -> None:
        super().__init__(window=base_window, n_components=n_components)
        self.base_window = base_window
        self.vol_lookback = vol_lookback
        self._last_window_used: int | None = None

    def _compute_window(self, returns: pd.DataFrame) -> int:
        """Derive the adaptive window from realized cross-sectional volatility.

        All computations use only data present in `returns` — no look-ahead.

        Args:
            returns: Wide log-return matrix up to and including date J.

        Returns:
            Integer window length clamped to [base_window // 2, base_window * 2].
        """
        if len(returns) < 2:
            return self.base_window

        mean_abs = returns.abs().mean(axis=1)

        lb = min(self.vol_lookback, len(mean_abs))
        recent_vol = float(mean_abs.iloc[-lb:].std() * np.sqrt(252))
        long_vol = float(mean_abs.std() * np.sqrt(252))

        if long_vol == 0.0 or np.isnan(long_vol) or np.isnan(recent_vol):
            return self.base_window

        ratio = recent_vol / long_vol
        raw = int(self.base_window * ratio)
        window = int(np.clip(raw, self.base_window // 2, self.base_window * 2))
        return window

    def fit(self, returns: pd.DataFrame) -> None:
        """Fit PCA using a volatility-adaptive window derived from `returns`.

        No look-ahead: both window computation and PCA fitting use only the
        rows present in `returns`.

        Args:
            returns: Wide log-return matrix ending at date J.

        Raises:
            ValueError: If fewer rows than n_components are available.
        """
        if len(returns) < self.n_components:
            raise ValueError(
                f"Need at least {self.n_components} rows to fit PCA; "
                f"got {len(returns)}."
            )

        window = self._compute_window(returns)
        self._last_window_used = window
        effective = min(window, len(returns))
        logger.debug("AdaptivePCA: adaptive window=%d (base=%d)", effective, self.base_window)
        self._fit_on_slice(returns.iloc[-effective:])
