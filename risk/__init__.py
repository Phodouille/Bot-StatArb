"""Risk management module for the StatArb bot.

Public API
----------
load_risk_params    Load and validate config/risk_params.yaml.
KillSwitch          Stateful drawdown guard; call reset() from the dashboard.
PositionSizer       Applies per-ticker caps, neutrality, and buying-power constraints.
RiskChecker         Validates a signal set; surfaces every violation explicitly.
risk_check          Top-level function: runs kill-switch + checker + sizer in sequence.
"""

from risk.checker import RiskChecker
from risk.config import load_risk_params
from risk.kill_switch import KillSwitch
from risk.risk_check import risk_check
from risk.sizer import PositionSizer

__all__ = [
    "load_risk_params",
    "KillSwitch",
    "PositionSizer",
    "RiskChecker",
    "risk_check",
]
