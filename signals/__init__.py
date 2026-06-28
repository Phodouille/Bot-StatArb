"""signals — PCA-based signal generation for the StatArb bot.

Public API
----------
generate_signals : main entry point (levels 3 and 4)
ClassicPCA       : fixed-window PCA model
AdaptivePCA      : volatility-adaptive PCA model
MacroFilter      : VIX / credit-spread macro stress filter
compute_zscore   : rolling z-score from PCA residuals
get_signals      : threshold-based signal extraction
"""

from signals.filters import MacroFilter
from signals.generator import generate_signals
from signals.pca import AdaptivePCA, ClassicPCA
from signals.zscore import compute_zscore, get_signals

__all__ = [
    "generate_signals",
    "ClassicPCA",
    "AdaptivePCA",
    "MacroFilter",
    "compute_zscore",
    "get_signals",
]
