"""Stateful drawdown kill switch.

Designed to be pure enough for deterministic testing: given an equity value,
``check()`` transitions state and returns a boolean.  The dashboard can read
``is_triggered`` and call ``reset()`` to resume trading after manual review.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class KillSwitch:
    """Tracks peak portfolio value and triggers when drawdown exceeds the configured threshold.

    State machine:
        - Starts untriggered with peak = 0.
        - On the first ``check()`` call the peak is initialised to ``current_value``.
        - Once triggered, stays triggered until ``reset()`` is called explicitly.

    Args:
        params: Full risk params dict produced by ``load_risk_params()``.
            Reads ``params["drawdown"]["kill_switch_threshold"]`` and
            ``params["drawdown"]["warning_threshold"]``.
    """

    def __init__(self, params: dict[str, Any]) -> None:
        self._kill_threshold: float = params["drawdown"]["kill_switch_threshold"]
        self._warn_threshold: float = params["drawdown"]["warning_threshold"]
        self._triggered: bool = False
        self._peak_value: float = 0.0

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def check(self, current_value: float) -> bool:
        """Evaluate current equity against the high-water mark.

        Updates the peak whenever ``current_value`` exceeds it, then computes
        drawdown and transitions state accordingly.

        Args:
            current_value: Current portfolio equity (e.g. Alpaca
                ``portfolio_value``).  Must be positive.

        Returns:
            ``True`` if the kill switch is (or just became) triggered,
            ``False`` otherwise.
        """
        if current_value <= 0:
            logger.error(
                "KillSwitch.check received non-positive value %.2f — treating as triggered",
                current_value,
            )
            self._triggered = True
            return True

        # Once triggered, stay triggered regardless of recovery.
        if self._triggered:
            return True

        if current_value > self._peak_value:
            self._peak_value = current_value

        dd = self.current_drawdown(current_value)

        if abs(dd) >= self._kill_threshold:
            self._triggered = True
            logger.critical(
                "KILL SWITCH TRIGGERED: drawdown %.4f (%.2f%%) >= threshold %.4f (%.2f%%). "
                "Peak was %.2f, current is %.2f. Trading halted until manual reset.",
                dd,
                abs(dd) * 100,
                self._kill_threshold,
                self._kill_threshold * 100,
                self._peak_value,
                current_value,
            )
            return True

        if abs(dd) >= self._warn_threshold:
            logger.warning(
                "Drawdown WARNING: %.4f (%.2f%%) >= warning threshold %.4f (%.2f%%). "
                "Peak %.2f, current %.2f.",
                dd,
                abs(dd) * 100,
                self._warn_threshold,
                self._warn_threshold * 100,
                self._peak_value,
                current_value,
            )

        return False

    @property
    def is_triggered(self) -> bool:
        """Whether the kill switch is currently active."""
        return self._triggered

    def reset(self) -> None:
        """Manually reset the kill switch after human review.

        Resets both the triggered flag and the peak value so that the new
        post-reset equity becomes the reference high-water mark on the next
        ``check()`` call.
        """
        logger.warning(
            "KillSwitch manually reset. Previous peak was %.2f. "
            "Peak will be re-initialised on next check().",
            self._peak_value,
        )
        self._triggered = False
        self._peak_value = 0.0

    def current_drawdown(self, current_value: float) -> float:
        """Compute the drawdown relative to the current peak.

        Args:
            current_value: Current portfolio equity.

        Returns:
            ``(current_value - peak) / peak`` as a signed fraction.
            Negative means a drawdown; zero means at or above peak.
            Returns ``0.0`` if peak has not been initialised yet.
        """
        if self._peak_value <= 0:
            return 0.0
        return (current_value - self._peak_value) / self._peak_value

    @property
    def peak_value(self) -> float:
        """The highest portfolio value observed since the last reset."""
        return self._peak_value
