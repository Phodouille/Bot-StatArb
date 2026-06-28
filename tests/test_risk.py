"""Pytest suite for the risk/ module.

Covers:
- load_risk_params: valid load and missing-field errors.
- PositionSizer.size_positions: per-ticker cap, gross-exposure cap, buying-power cap.
- KillSwitch.check: triggers exactly at threshold, not before; stays triggered; reset.
- RiskChecker.check_market_neutrality: detects net exposure > max_net_exposure.
- RiskChecker.validate: aggregates violations; kill switch blocks everything.
- risk_check (top-level): returns {} when kill switch is active.
"""

from __future__ import annotations

import copy
import os
import tempfile
from pathlib import Path

import pytest
import yaml

from risk.checker import RiskChecker
from risk.config import load_risk_params
from risk.kill_switch import KillSwitch
from risk.risk_check import risk_check
from risk.sizer import PositionSizer

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

YAML_CONTENT = {
    "position": {
        "max_weight_per_ticker": 0.10,
        "min_weight_per_ticker": 0.001,
        "max_gross_exposure": 1.5,
        "max_net_exposure": 0.10,
    },
    "drawdown": {
        "kill_switch_threshold": 0.05,
        "warning_threshold": 0.03,
    },
    "sizing": {
        "portfolio_fraction": 0.50,
        "max_positions": 20,
    },
    "transaction": {
        "cost_bps": 7.5,
    },
}


@pytest.fixture()
def params() -> dict:
    """Return a copy of the canonical test params (no file I/O)."""
    return copy.deepcopy(YAML_CONTENT)


@pytest.fixture()
def yaml_file(tmp_path: Path) -> Path:
    """Write the canonical params to a temporary YAML file."""
    p = tmp_path / "risk_params.yaml"
    p.write_text(yaml.dump(YAML_CONTENT))
    return p


@pytest.fixture()
def kill_switch(params: dict) -> KillSwitch:
    return KillSwitch(params)


@pytest.fixture()
def sizer(params: dict) -> PositionSizer:
    return PositionSizer(params)


@pytest.fixture()
def checker(params: dict) -> RiskChecker:
    return RiskChecker(params)


def _signals(n_long: int = 2, n_short: int = 2, weight: float = 0.05) -> dict:
    """Build a balanced signal dict with ``n_long`` longs and ``n_short`` shorts."""
    sigs: dict = {}
    tickers = [f"T{i:02d}" for i in range(n_long + n_short)]
    for i, ticker in enumerate(tickers[:n_long]):
        sigs[ticker] = {"direction": "long", "z_score": -(i + 2.0), "weight": weight}
    for i, ticker in enumerate(tickers[n_long:]):
        sigs[ticker] = {"direction": "short", "z_score": (i + 2.0), "weight": weight}
    return sigs


# ---------------------------------------------------------------------------
# load_risk_params
# ---------------------------------------------------------------------------


class TestLoadRiskParams:
    def test_loads_valid_yaml(self, yaml_file: Path) -> None:
        params = load_risk_params(str(yaml_file))
        assert params["position"]["max_weight_per_ticker"] == 0.10
        assert params["drawdown"]["kill_switch_threshold"] == 0.05
        assert params["sizing"]["max_positions"] == 20
        assert params["transaction"]["cost_bps"] == 7.5

    def test_raises_on_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_risk_params(str(tmp_path / "nonexistent.yaml"))

    def test_raises_on_missing_section(self, tmp_path: Path) -> None:
        bad = copy.deepcopy(YAML_CONTENT)
        del bad["drawdown"]
        p = tmp_path / "bad.yaml"
        p.write_text(yaml.dump(bad))
        with pytest.raises(ValueError, match="drawdown"):
            load_risk_params(str(p))

    def test_raises_on_missing_key_within_section(self, tmp_path: Path) -> None:
        bad = copy.deepcopy(YAML_CONTENT)
        del bad["position"]["max_weight_per_ticker"]
        p = tmp_path / "bad.yaml"
        p.write_text(yaml.dump(bad))
        with pytest.raises(ValueError, match="max_weight_per_ticker"):
            load_risk_params(str(p))

    def test_raises_on_non_mapping_yaml(self, tmp_path: Path) -> None:
        p = tmp_path / "list.yaml"
        p.write_text("- a\n- b\n")
        with pytest.raises(ValueError):
            load_risk_params(str(p))


# ---------------------------------------------------------------------------
# KillSwitch
# ---------------------------------------------------------------------------


class TestKillSwitch:
    def test_not_triggered_below_threshold(self, kill_switch: KillSwitch) -> None:
        # Initialise peak to 100_000, then drop by 4.9% — must NOT trigger.
        kill_switch.check(100_000.0)
        result = kill_switch.check(95_100.0)  # 4.9% drawdown
        assert result is False
        assert kill_switch.is_triggered is False

    def test_triggers_exactly_at_threshold(self, kill_switch: KillSwitch) -> None:
        # Threshold = 5%.  Drop of exactly 5% must trigger.
        kill_switch.check(100_000.0)
        result = kill_switch.check(95_000.0)  # exactly 5%
        assert result is True
        assert kill_switch.is_triggered is True

    def test_triggers_above_threshold(self, kill_switch: KillSwitch) -> None:
        kill_switch.check(100_000.0)
        result = kill_switch.check(90_000.0)  # 10% drawdown
        assert result is True
        assert kill_switch.is_triggered is True

    def test_stays_triggered_after_recovery(self, kill_switch: KillSwitch) -> None:
        kill_switch.check(100_000.0)
        kill_switch.check(94_000.0)  # triggers
        # Even if value recovers, kill switch must remain active.
        result = kill_switch.check(100_000.0)
        assert result is True
        assert kill_switch.is_triggered is True

    def test_reset_clears_triggered_flag(self, kill_switch: KillSwitch) -> None:
        kill_switch.check(100_000.0)
        kill_switch.check(90_000.0)  # triggers
        assert kill_switch.is_triggered is True

        kill_switch.reset()
        assert kill_switch.is_triggered is False
        assert kill_switch.peak_value == 0.0

    def test_not_triggered_after_reset_below_threshold(self, kill_switch: KillSwitch) -> None:
        kill_switch.check(100_000.0)
        kill_switch.check(90_000.0)
        kill_switch.reset()

        # Re-initialise peak at 80k, drop to 77k (3.75% — below threshold).
        kill_switch.check(80_000.0)
        result = kill_switch.check(77_000.0)
        assert result is False
        assert kill_switch.is_triggered is False

    def test_current_drawdown_before_initialisation(self, kill_switch: KillSwitch) -> None:
        assert kill_switch.current_drawdown(50_000.0) == 0.0

    def test_current_drawdown_value(self, kill_switch: KillSwitch) -> None:
        kill_switch.check(100_000.0)
        dd = kill_switch.current_drawdown(95_000.0)
        assert abs(dd - (-0.05)) < 1e-9

    def test_peak_updates_on_new_high(self, kill_switch: KillSwitch) -> None:
        kill_switch.check(100_000.0)
        kill_switch.check(110_000.0)
        assert kill_switch.peak_value == pytest.approx(110_000.0)


# ---------------------------------------------------------------------------
# PositionSizer
# ---------------------------------------------------------------------------


class TestPositionSizer:
    def test_cap_per_ticker_enforced(self, sizer: PositionSizer) -> None:
        signals = {
            "AAPL": {"direction": "long", "z_score": -3.0, "weight": 0.20},  # over cap
            "MSFT": {"direction": "short", "z_score": 2.5, "weight": 0.08},
        }
        result = sizer.size_positions(signals, portfolio_value=100_000, buying_power=100_000)
        assert result["AAPL"]["weight"] <= 0.10, "Weight must be capped at max_weight_per_ticker"

    def test_gross_exposure_cap(self, sizer: PositionSizer) -> None:
        # 10 longs × 0.12 + 10 shorts × 0.12 = 2.4 gross → above 1.5 cap
        signals = {}
        for i in range(10):
            signals[f"L{i}"] = {"direction": "long", "z_score": -(i + 2.0), "weight": 0.10}
            signals[f"S{i}"] = {"direction": "short", "z_score": (i + 2.0), "weight": 0.10}
        result = sizer.size_positions(signals, portfolio_value=100_000, buying_power=100_000)
        gross = sum(s["weight"] for s in result.values())
        # After cap (each capped at 0.10) gross would be 2.0 → should be scaled to 1.5
        assert gross <= 1.50 + 1e-9, f"Gross exposure {gross:.4f} exceeds 1.5"

    def test_buying_power_scale_down(self, sizer: PositionSizer) -> None:
        # 4 positions × 0.10 = 0.40 gross weight → $40k needed; buying_power = $20k
        signals = {
            "A": {"direction": "long", "z_score": -2.0, "weight": 0.10},
            "B": {"direction": "long", "z_score": -2.1, "weight": 0.10},
            "C": {"direction": "short", "z_score": 2.0, "weight": 0.10},
            "D": {"direction": "short", "z_score": 2.1, "weight": 0.10},
        }
        result = sizer.size_positions(
            signals, portfolio_value=100_000, buying_power=20_000
        )
        gross = sum(s["weight"] for s in result.values())
        # buying_power / portfolio_value = 0.20, so gross must be <= 0.20
        assert gross <= 0.20 + 1e-9, f"Gross weight {gross:.4f} exceeds buying_power ratio"

    def test_empty_signals_returns_empty(self, sizer: PositionSizer) -> None:
        assert sizer.size_positions({}, 100_000, 100_000) == {}

    def test_max_positions_trim(self, sizer: PositionSizer) -> None:
        # Build 25 signals (above max_positions=20); highest |z_score| must survive.
        signals = {}
        for i in range(25):
            signals[f"T{i:02d}"] = {
                "direction": "long",
                "z_score": -(float(i) + 1.0),
                "weight": 0.02,
            }
        result = sizer.size_positions(signals, 100_000, 100_000)
        assert len(result) <= 20
        # The 5 lowest |z_score| (T00..T04) must have been dropped.
        for i in range(5):
            assert f"T{i:02d}" not in result

    def test_compute_dollar_positions(self, sizer: PositionSizer) -> None:
        signals = {"AAPL": {"direction": "long", "z_score": -2.3, "weight": 0.05}}
        dollar_pos = sizer.compute_dollar_positions(signals, portfolio_value=100_000)
        assert dollar_pos["AAPL"]["dollars"] == pytest.approx(5_000.0)
        assert dollar_pos["AAPL"]["direction"] == "long"
        assert dollar_pos["AAPL"]["weight"] == 0.05


# ---------------------------------------------------------------------------
# RiskChecker
# ---------------------------------------------------------------------------


class TestRiskChecker:
    def test_neutral_signals_pass(self, checker: RiskChecker) -> None:
        signals = _signals(n_long=2, n_short=2, weight=0.05)
        is_neutral, net = checker.check_market_neutrality(signals)
        assert is_neutral is True
        assert abs(net) < 1e-9

    def test_non_neutral_detected(self, checker: RiskChecker) -> None:
        # 4 longs × 0.10 = 0.40 long, 0 short → net = 0.40 >> 0.10 threshold.
        signals = {
            "A": {"direction": "long", "z_score": -2.0, "weight": 0.10},
            "B": {"direction": "long", "z_score": -2.1, "weight": 0.10},
            "C": {"direction": "long", "z_score": -2.2, "weight": 0.10},
            "D": {"direction": "long", "z_score": -2.3, "weight": 0.10},
        }
        is_neutral, net = checker.check_market_neutrality(signals)
        assert is_neutral is False
        assert net == pytest.approx(0.40)

    def test_net_exposure_just_at_threshold_is_ok(self, checker: RiskChecker) -> None:
        # net = 0.10 exactly — should be considered neutral (<=, not <).
        signals = {
            "A": {"direction": "long", "z_score": -2.0, "weight": 0.20},
            "B": {"direction": "short", "z_score": 2.0, "weight": 0.10},
        }
        is_neutral, net = checker.check_market_neutrality(signals)
        assert is_neutral is True
        assert net == pytest.approx(0.10)

    def test_concentration_violation_detected(self, checker: RiskChecker) -> None:
        signals = {
            "OVER": {"direction": "long", "z_score": -3.0, "weight": 0.15},
            "OK": {"direction": "short", "z_score": 2.0, "weight": 0.05},
        }
        ok, violations = checker.check_concentration(signals)
        assert ok is False
        assert "OVER" in violations

    def test_concentration_no_violation(self, checker: RiskChecker) -> None:
        signals = _signals(n_long=2, n_short=2, weight=0.05)
        ok, violations = checker.check_concentration(signals)
        assert ok is True
        assert violations == []

    def test_validate_returns_violations_list(self, checker: RiskChecker) -> None:
        # Both concentration and neutrality violated.
        signals = {
            "A": {"direction": "long", "z_score": -3.0, "weight": 0.20},
            "B": {"direction": "long", "z_score": -2.5, "weight": 0.20},
        }
        all_ok, violations = checker.validate(signals, 100_000, 100_000)
        assert all_ok is False
        assert len(violations) >= 2

    def test_validate_kill_switch_blocks_everything(self, params: dict) -> None:
        ks = KillSwitch(params)
        checker_with_ks = RiskChecker(params, kill_switch=ks)

        # Trigger the kill switch.
        ks.check(100_000.0)
        ks.check(90_000.0)
        assert ks.is_triggered is True

        # validate() must immediately return a kill-switch violation.
        all_ok, violations = checker_with_ks.validate(
            _signals(), portfolio_value=100_000, buying_power=100_000
        )
        assert all_ok is False
        assert len(violations) == 1
        assert "kill switch" in violations[0].lower()

    def test_validate_buying_power_violation(self, checker: RiskChecker) -> None:
        # 4 positions × 0.10 = 0.40 gross → $40k required; buying_power = $10k
        signals = {
            "A": {"direction": "long", "z_score": -2.0, "weight": 0.10},
            "B": {"direction": "short", "z_score": 2.0, "weight": 0.10},
            "C": {"direction": "long", "z_score": -2.1, "weight": 0.10},
            "D": {"direction": "short", "z_score": 2.1, "weight": 0.10},
        }
        all_ok, violations = checker.validate(signals, 100_000, buying_power=10_000)
        assert all_ok is False
        assert any("buying_power" in v.lower() for v in violations)


# ---------------------------------------------------------------------------
# risk_check (top-level integration)
# ---------------------------------------------------------------------------


class TestRiskCheck:
    def _portfolio_state(
        self,
        value: float = 100_000.0,
        buying_power: float = 100_000.0,
        equity: float | None = None,
    ) -> dict:
        return {
            "portfolio_value": value,
            "buying_power": buying_power,
            "current_equity": equity if equity is not None else value,
        }

    def test_returns_adjusted_signals_normally(self, params: dict) -> None:
        ks = KillSwitch(params)
        signals = _signals(n_long=2, n_short=2, weight=0.05)
        result = risk_check(signals, self._portfolio_state(), params, ks)
        assert isinstance(result, dict)
        assert len(result) > 0

    def test_returns_empty_when_kill_switch_triggered(self, params: dict) -> None:
        ks = KillSwitch(params)
        # Trigger the kill switch before calling risk_check.
        ks.check(100_000.0)
        ks.check(90_000.0)
        assert ks.is_triggered is True

        signals = _signals(n_long=2, n_short=2, weight=0.05)
        result = risk_check(signals, self._portfolio_state(), params, ks)
        assert result == {}

    def test_kill_switch_triggers_inside_risk_check(self, params: dict) -> None:
        ks = KillSwitch(params)
        # First call sets peak to 100k.
        ks.check(100_000.0)

        signals = _signals(n_long=2, n_short=2, weight=0.05)
        # Pass equity = 90k (10% drawdown) — risk_check must detect and return {}.
        result = risk_check(
            signals,
            self._portfolio_state(equity=90_000.0),
            params,
            ks,
        )
        assert result == {}
        assert ks.is_triggered is True

    def test_all_weights_within_cap_after_risk_check(self, params: dict) -> None:
        ks = KillSwitch(params)
        signals = {
            "AAPL": {"direction": "long", "z_score": -3.0, "weight": 0.20},
            "MSFT": {"direction": "short", "z_score": 3.0, "weight": 0.20},
        }
        result = risk_check(signals, self._portfolio_state(), params, ks)
        for ticker, sig in result.items():
            assert sig["weight"] <= 0.10 + 1e-9, (
                f"{ticker} weight {sig['weight']:.4f} still exceeds cap after risk_check"
            )

    def test_empty_signals_returns_empty(self, params: dict) -> None:
        ks = KillSwitch(params)
        result = risk_check({}, self._portfolio_state(), params, ks)
        assert result == {}
