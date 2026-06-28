"""Loads risk parameters from the canonical YAML config file.

All risk modules must obtain their parameters through this loader.
No risk constant may be hardcoded anywhere else in the codebase.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

_REQUIRED_KEYS: dict[str, list[str]] = {
    "position": [
        "max_weight_per_ticker",
        "min_weight_per_ticker",
        "max_gross_exposure",
        "max_net_exposure",
    ],
    "drawdown": [
        "kill_switch_threshold",
        "warning_threshold",
    ],
    "sizing": [
        "portfolio_fraction",
        "max_positions",
    ],
    "transaction": [
        "cost_bps",
    ],
}


def load_risk_params(path: str = "config/risk_params.yaml") -> dict[str, Any]:
    """Load risk parameters from a YAML file and validate all required fields.

    Args:
        path: Path to the YAML config file. Resolved relative to the
            repository root when not absolute.

    Returns:
        Nested dict mirroring the YAML structure.

    Raises:
        FileNotFoundError: If the YAML file does not exist at ``path``.
        ValueError: If any required top-level section or key is missing.
    """
    resolved = Path(path)
    if not resolved.is_absolute():
        # Walk up from this file's location until we find the repo root
        # (identified by the presence of risk_params.yaml or CLAUDE.md).
        candidate = Path(__file__).parent.parent / path
        if candidate.exists():
            resolved = candidate

    if not resolved.exists():
        raise FileNotFoundError(
            f"Risk params file not found: {resolved}. "
            "Expected at config/risk_params.yaml relative to the repo root."
        )

    with resolved.open("r") as fh:
        params: dict[str, Any] = yaml.safe_load(fh)

    if not isinstance(params, dict):
        raise ValueError(f"risk_params.yaml must be a YAML mapping, got {type(params)}")

    missing: list[str] = []
    for section, keys in _REQUIRED_KEYS.items():
        if section not in params:
            missing.append(f"section '{section}'")
            continue
        for key in keys:
            if key not in params[section]:
                missing.append(f"'{section}.{key}'")

    if missing:
        raise ValueError(
            f"risk_params.yaml is missing required field(s): {', '.join(missing)}"
        )

    logger.info("Risk params loaded from %s", resolved)
    return params
