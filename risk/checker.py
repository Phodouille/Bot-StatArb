"""Stateless signal validation layer.

``RiskChecker`` is the façade that execution and the dashboard call before
any trade is placed.  It is intentionally stateless apart from the optional
``KillSwitch`` it holds a reference to — all state lives in ``KillSwitch``.
"""

from __future__ import annotations

import logging
from typing import Any

from risk.kill_switch import KillSwitch

logger = logging.getLogger(__name__)


class RiskChecker:
    """Validates a proposed signal set against all configured risk constraints.

    Args:
        params: Full risk params dict produced by ``load_risk_params()``.
        kill_switch: Optional ``KillSwitch`` instance.  When provided,
            ``validate()`` checks it first — a triggered kill switch
            short-circuits all other checks.
    """

    def __init__(
        self,
        params: dict[str, Any],
        kill_switch: KillSwitch | None = None,
    ) -> None:
        self._pos = params["position"]
        self._kill_switch = kill_switch

    # ------------------------------------------------------------------
    # Individual checks (testable in isolation)
    # ------------------------------------------------------------------

    def check_market_neutrality(self, signals: dict[str, dict]) -> tuple[bool, float]:
        """Compute net exposure and test against the neutrality constraint.

        Args:
            signals: Signal contract dict ``{ticker: {direction, z_score, weight}}``.

        Returns:
            Tuple ``(is_neutral, net_exposure)`` where ``net_exposure`` is
            ``sum(long weights) - sum(short weights)`` (signed).  ``is_neutral``
            is ``True`` iff ``|net_exposure| <= max_net_exposure``.
        """
        long_total = sum(
            s["weight"] for s in signals.values() if s.get("direction") == "long"
        )
        short_total = sum(
            s["weight"] for s in signals.values() if s.get("direction") == "short"
        )
        net = long_total - short_total
        is_neutral = abs(net) <= self._pos["max_net_exposure"]

        if not is_neutral:
            logger.warning(
                "RISK VIOLATION: net exposure %.4f exceeds max_net_exposure %.4f.",
                net, self._pos["max_net_exposure"],
            )

        return is_neutral, net

    def check_concentration(self, signals: dict[str, dict]) -> tuple[bool, list[str]]:
        """Identify tickers whose weight exceeds the per-ticker cap.

        Args:
            signals: Signal contract dict.

        Returns:
            Tuple ``(ok, violating_tickers)``.  ``ok`` is ``True`` iff no
            ticker exceeds ``max_weight_per_ticker``.
        """
        max_w = self._pos["max_weight_per_ticker"]
        violations = [
            ticker
            for ticker, sig in signals.items()
            if sig.get("weight", 0.0) > max_w
        ]

        if violations:
            for ticker in violations:
                logger.warning(
                    "RISK VIOLATION: %s weight %.4f > max_weight_per_ticker %.4f.",
                    ticker, signals[ticker]["weight"], max_w,
                )

        return len(violations) == 0, violations

    # ------------------------------------------------------------------
    # Full validation pipeline
    # ------------------------------------------------------------------

    def validate(
        self,
        signals: dict[str, dict],
        portfolio_value: float,
        buying_power: float,
    ) -> tuple[bool, list[str]]:
        """Run all risk checks in priority order and aggregate violations.

        Check order: kill_switch → concentration → neutrality → buying_power.
        Returns on first kill-switch trigger (trading must stop immediately).
        Other checks accumulate into the violations list.

        Args:
            signals: Signal contract dict.
            portfolio_value: Total portfolio equity in dollars.
            buying_power: Available buying power from Alpaca (T+2 aware).
                Never pass ``cash`` here.

        Returns:
            Tuple ``(all_ok, violations)`` where ``violations`` is a list of
            human-readable strings describing every constraint that was
            breached.  ``all_ok`` is ``False`` if *any* violation exists.
        """
        violations: list[str] = []

        # 1. Kill switch — if triggered, nothing else matters.
        if self._kill_switch is not None and self._kill_switch.is_triggered:
            msg = "Kill switch is active — all trading halted until manual reset."
            logger.critical(msg)
            violations.append(msg)
            return False, violations

        # 2. Per-ticker concentration.
        conc_ok, over_cap = self.check_concentration(signals)
        if not conc_ok:
            violations.append(
                f"Concentration violation: {over_cap} exceed max_weight_per_ticker "
                f"{self._pos['max_weight_per_ticker']:.1%}."
            )

        # 3. Market neutrality.
        neutral_ok, net = self.check_market_neutrality(signals)
        if not neutral_ok:
            violations.append(
                f"Neutrality violation: net_exposure={net:.4f} exceeds "
                f"max_net_exposure={self._pos['max_net_exposure']:.1%}."
            )

        # 4. Buying-power check.
        if portfolio_value > 0:
            gross_weight = sum(s.get("weight", 0.0) for s in signals.values())
            required_capital = gross_weight * portfolio_value
            if required_capital > buying_power:
                msg = (
                    f"Insufficient buying_power: need ${required_capital:,.2f} "
                    f"but buying_power=${buying_power:,.2f}. "
                    "(buying_power used, not cash — T+2 settlement.)"
                )
                logger.warning(msg)
                violations.append(msg)

        all_ok = len(violations) == 0
        return all_ok, violations
