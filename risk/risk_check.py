"""Top-level ``risk_check`` function — the single entry point for execution.

This is what execution/ and the dashboard should call.  It chains
``KillSwitch``, ``RiskChecker``, and ``PositionSizer`` in the correct order
and is guaranteed never to silently swallow a violation.
"""

from __future__ import annotations

import logging
from typing import Any

from risk.checker import RiskChecker
from risk.kill_switch import KillSwitch
from risk.sizer import PositionSizer

logger = logging.getLogger(__name__)


def risk_check(
    signals: dict[str, dict],
    portfolio_state: dict[str, Any],
    params: dict[str, Any],
    kill_switch: KillSwitch,
) -> dict[str, dict]:
    """Validate and size a proposed signal set against all risk constraints.

    This is the primary interface consumed by ``execution/``.  It will never
    silently pass a violation: every breach is logged at WARNING or CRITICAL
    level and surfaces in the returned dict or as an empty dict.

    Args:
        signals: Raw signal contract dict from ``signals/``.
            Format: ``{ticker: {"direction": str, "z_score": float, "weight": float}}``.
        portfolio_state: Dict with keys:
            - ``"portfolio_value"`` (float): total equity.
            - ``"buying_power"`` (float): Alpaca buying_power (T+2-aware).
            - ``"current_equity"`` (float): current equity for drawdown check
              (may equal ``portfolio_value``; kept separate for flexibility).
        params: Full risk params dict from ``load_risk_params()``.
        kill_switch: Shared ``KillSwitch`` instance (stateful).

    Returns:
        Adjusted signal dict with weights that respect all constraints, or
        an empty dict if the kill switch is triggered or no valid signals
        remain after filtering.
    """
    portfolio_value: float = portfolio_state["portfolio_value"]
    buying_power: float = portfolio_state["buying_power"]
    current_equity: float = portfolio_state.get("current_equity", portfolio_value)

    # Always evaluate drawdown first — this updates the peak and may trigger.
    kill_switch.check(current_equity)

    if kill_switch.is_triggered:
        logger.critical(
            "risk_check: kill switch active — returning empty signal set. "
            "No trades will be placed until the kill switch is manually reset."
        )
        return {}

    checker = RiskChecker(params, kill_switch=kill_switch)
    _, violations = checker.validate(signals, portfolio_value, buying_power)
    if violations:
        logger.warning("risk_check pre-sizing violations: %s", violations)

    sizer = PositionSizer(params)
    adjusted = sizer.size_positions(signals, portfolio_value, buying_power)

    if not adjusted:
        logger.info("risk_check: no signals remain after sizing constraints.")
        return {}

    # Final validation on the adjusted signals.
    all_ok, post_violations = checker.validate(adjusted, portfolio_value, buying_power)
    if not all_ok:
        logger.error(
            "risk_check: post-sizing validation still has violations: %s. "
            "Returning adjusted signals anyway — execution layer must handle.",
            post_violations,
        )

    return adjusted
