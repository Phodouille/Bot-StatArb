"""Position sizing: translates raw signal weights into risk-adjusted allocations.

All constraints are driven by ``config/risk_params.yaml``; no numeric constant
lives in this file.  The module consumes and produces the CONTRAT 1 signal
format.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class PositionSizer:
    """Applies portfolio-level risk constraints to a proposed signal set.

    Constraint application order (mirrors ``size_positions`` steps):
        1. Drop positions below ``min_weight_per_ticker``.
        2. Cap each weight at ``max_weight_per_ticker``.
        3. Trim to ``max_positions`` by keeping the highest ``|z_score|``.
        4. Re-normalise gross exposure to ``max_gross_exposure``.
        5. Enforce ``max_net_exposure`` by trimming the dominant direction.
        6. Scale down proportionally if required capital > ``buying_power``.

    Args:
        params: Full risk params dict produced by ``load_risk_params()``.
    """

    def __init__(self, params: dict[str, Any]) -> None:
        self._pos = params["position"]
        self._siz = params["sizing"]

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def size_positions(
        self,
        signals: dict[str, dict],
        portfolio_value: float,
        buying_power: float,
    ) -> dict[str, dict]:
        """Apply risk constraints to a signal dict and return adjusted signals.

        Uses ``buying_power`` (not cash) to account for Alpaca T+2 cash-account
        settlement delay.

        Args:
            signals: Signal contract dict ``{ticker: {direction, z_score, weight}}``.
            portfolio_value: Total portfolio equity in dollars.
            buying_power: Available buying power from Alpaca (never use ``cash``).

        Returns:
            Adjusted signal dict in the same CONTRAT 1 format with weights
            that satisfy all configured constraints.  Empty dict if no valid
            signals remain after filtering.
        """
        if not signals:
            return {}

        # Working copy — do not mutate the caller's dict.
        adjusted: dict[str, dict] = {
            ticker: dict(sig) for ticker, sig in signals.items()
        }

        max_w = self._pos["max_weight_per_ticker"]
        min_w = self._pos["min_weight_per_ticker"]
        max_gross = self._pos["max_gross_exposure"]
        max_net = self._pos["max_net_exposure"]
        max_pos = self._siz["max_positions"]
        port_frac = self._siz["portfolio_fraction"]

        # Step 1: Drop sub-minimum weights.
        pre_filter_count = len(adjusted)
        adjusted = {
            t: s for t, s in adjusted.items() if s["weight"] >= min_w
        }
        dropped = pre_filter_count - len(adjusted)
        if dropped:
            logger.info("Dropped %d signal(s) below min_weight_per_ticker %.4f", dropped, min_w)

        if not adjusted:
            return {}

        # Step 2: Cap individual weights.
        for ticker, sig in adjusted.items():
            if sig["weight"] > max_w:
                logger.warning(
                    "RISK VIOLATION: %s weight %.4f > max_weight_per_ticker %.4f — capping.",
                    ticker, sig["weight"], max_w,
                )
                sig["weight"] = max_w

        # Step 3: Trim to max_positions by |z_score|.
        if len(adjusted) > max_pos:
            ranked = sorted(
                adjusted.items(),
                key=lambda kv: abs(kv[1].get("z_score", 0.0)),
                reverse=True,
            )
            dropped_tickers = [t for t, _ in ranked[max_pos:]]
            logger.warning(
                "Trimming %d signal(s) exceeding max_positions=%d: %s",
                len(dropped_tickers), max_pos, dropped_tickers,
            )
            adjusted = dict(ranked[:max_pos])

        # Step 4: Re-normalise to max_gross_exposure.
        gross = sum(s["weight"] for s in adjusted.values())
        if gross > max_gross:
            scale = max_gross / gross
            logger.warning(
                "Gross exposure %.4f > max_gross_exposure %.4f — scaling by %.4f.",
                gross, max_gross, scale,
            )
            for sig in adjusted.values():
                sig["weight"] *= scale

        # Step 5: Enforce max_net_exposure.
        adjusted = self._enforce_net_exposure(adjusted, max_net)

        # Step 6: Scale down if required capital > buying_power.
        adjusted = self._enforce_buying_power(
            adjusted, portfolio_value, buying_power, port_frac
        )

        return adjusted

    def compute_dollar_positions(
        self,
        sized_signals: dict[str, dict],
        portfolio_value: float,
    ) -> dict[str, dict]:
        """Convert weight-based signals to dollar allocations.

        Args:
            sized_signals: Output of ``size_positions()``.
            portfolio_value: Total portfolio equity in dollars.

        Returns:
            Dict ``{ticker: {direction, dollars, weight, z_score}}``.
        """
        result: dict[str, dict] = {}
        for ticker, sig in sized_signals.items():
            result[ticker] = {
                "direction": sig["direction"],
                "dollars": round(sig["weight"] * portfolio_value, 2),
                "weight": sig["weight"],
                "z_score": sig.get("z_score", float("nan")),
            }
        return result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _enforce_net_exposure(
        self,
        signals: dict[str, dict],
        max_net: float,
    ) -> dict[str, dict]:
        """Trim the dominant direction until |net| <= max_net_exposure.

        Rather than rejecting positions entirely, weights in the dominant
        direction are scaled uniformly until the neutrality constraint is met.

        Args:
            signals: Working signal dict (weights already capped/normalised).
            max_net: Maximum allowed |net exposure| as a fraction.

        Returns:
            Adjusted signal dict.
        """
        long_total = sum(s["weight"] for s in signals.values() if s["direction"] == "long")
        short_total = sum(s["weight"] for s in signals.values() if s["direction"] == "short")
        net = long_total - short_total

        if abs(net) <= max_net:
            return signals

        logger.warning(
            "Net exposure %.4f exceeds max_net_exposure %.4f — adjusting dominant side.",
            net, max_net,
        )

        if net > max_net:
            # Longs are dominant; reduce them so net = max_net.
            # new_long_total = short_total + max_net
            target_long = short_total + max_net
            if long_total > 0:
                scale = target_long / long_total
                for sig in signals.values():
                    if sig["direction"] == "long":
                        sig["weight"] *= scale
        else:
            # Shorts are dominant; reduce them so net = -max_net.
            target_short = long_total + max_net
            if short_total > 0:
                scale = target_short / short_total
                for sig in signals.values():
                    if sig["direction"] == "short":
                        sig["weight"] *= scale

        return signals

    def _enforce_buying_power(
        self,
        signals: dict[str, dict],
        portfolio_value: float,
        buying_power: float,
        port_frac: float,
    ) -> dict[str, dict]:
        """Scale positions down if the required capital exceeds buying_power.

        Uses ``buying_power`` (not cash) per CLAUDE.md: T+2 settlement means
        uninvested cash from recent sales is not immediately available.

        Args:
            signals: Working signal dict.
            portfolio_value: Total portfolio equity.
            buying_power: Available buying power from Alpaca.
            port_frac: Maximum fraction of portfolio to deploy.

        Returns:
            Scaled signal dict.
        """
        gross_weight = sum(s["weight"] for s in signals.values())
        effective_cap = min(port_frac, buying_power / portfolio_value) if portfolio_value > 0 else 0.0

        if gross_weight > effective_cap and gross_weight > 0:
            scale = effective_cap / gross_weight
            logger.warning(
                "Required capital (gross weight %.4f × portfolio $%.2f) exceeds "
                "buying_power $%.2f — scaling all weights by %.4f.",
                gross_weight, portfolio_value, buying_power, scale,
            )
            for sig in signals.values():
                sig["weight"] *= scale

        return signals
