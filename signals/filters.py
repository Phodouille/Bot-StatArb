"""Macro filters that scale or suspend trading signals under market stress.

Rule (documented for calibration record):
  - VIX > vix_threshold  OR  credit_spread > spread_threshold
      → hard suspend: return {} (no trades)
  - 20 <= VIX <= vix_threshold  (and spread is below threshold)
      → linear scale-down of all weights
        multiplier = 1 - (VIX - 20) / (vix_threshold - 20)
        weight_new = weight_old * multiplier
  - VIX < 20 and spread < spread_threshold
      → signals unchanged (multiplier = 1.0)

Default thresholds: VIX=30, credit_spread=3.0 pp.
The linear ramp starts at VIX=20 (long-run median ≈ 19-20).
"""

import copy


_VIX_RAMP_START = 20.0  # VIX level below which no reduction is applied


class MacroFilter:
    """Scales or suspends signals based on VIX and credit spread levels.

    Args:
        vix_threshold: VIX level above which all trading is suspended.
            Below this but above _VIX_RAMP_START, weights are reduced linearly.
        spread_threshold: IG/HY credit spread (percentage points) above which
            all trading is suspended.
    """

    def __init__(
        self,
        vix_threshold: float = 30.0,
        spread_threshold: float = 3.0,
    ) -> None:
        self.vix_threshold = vix_threshold
        self.spread_threshold = spread_threshold

    def should_trade(self, vix: float, credit_spread: float) -> bool:
        """Return False if market stress exceeds either hard threshold.

        Args:
            vix: Current VIX level.
            credit_spread: Current IG/HY credit spread in percentage points.

        Returns:
            True when both indicators are below their respective thresholds.
        """
        return vix <= self.vix_threshold and credit_spread <= self.spread_threshold

    def _multiplier(self, vix: float, credit_spread: float) -> float:
        """Compute the weight multiplier in [0.0, 1.0].

        Returns 0.0 if either threshold is exceeded (hard suspend).
        Returns a value in (0, 1) when VIX is in the ramp zone [20, vix_threshold].
        Returns 1.0 when both indicators are below their soft/hard boundaries.

        Args:
            vix: Current VIX level.
            credit_spread: Current credit spread in percentage points.

        Returns:
            Float multiplier to apply to all signal weights.
        """
        if vix > self.vix_threshold or credit_spread > self.spread_threshold:
            return 0.0

        if vix >= _VIX_RAMP_START:
            ramp_range = self.vix_threshold - _VIX_RAMP_START
            if ramp_range <= 0:
                return 0.0
            return 1.0 - (vix - _VIX_RAMP_START) / ramp_range

        return 1.0

    def scale_weights(
        self,
        signals: dict[str, dict],
        vix: float,
        credit_spread: float,
    ) -> dict[str, dict]:
        """Apply macro filter to a signal dict, reducing or zeroing weights.

        Args:
            signals: Signal contract dict produced by zscore.get_signals().
            vix: Current VIX level.
            credit_spread: Current credit spread in percentage points.

        Returns:
            A new dict (original is not mutated) with weights scaled by the
            macro multiplier. Returns {} when multiplier is 0 (hard suspend or
            weight rounds to zero).
        """
        mult = self._multiplier(vix, credit_spread)
        if mult == 0.0:
            return {}

        if mult == 1.0:
            return signals

        scaled: dict[str, dict] = {}
        for ticker, info in signals.items():
            entry = copy.copy(info)
            entry["weight"] = float(info["weight"]) * mult
            scaled[ticker] = entry
        return scaled
