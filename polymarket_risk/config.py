"""Risk configuration as a plain dataclass.

No environment variable magic. Construct it explicitly, call ``validate()``,
and pass it to ``RiskManager``. Defaults mirror the parent bot's live
settings, which are deliberately conservative.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RiskConfig:
    # --- Capital ---
    # Configured deposit amount. Floors the daily-loss base so a transient
    # dip in reported portfolio value does not shrink the loss limit.
    bankroll_usdc: float = 1000.0
    # Halt new trades when effective cash falls below this floor.
    min_bankroll_usdc: float = 50.0

    # --- Sizing ---
    # Fraction of full Kelly. 0.25 (quarter-Kelly) is a deliberate de-risk:
    # half-Kelly drawdowns exceeded comfort during live volatility spikes.
    kelly_multiplier: float = 0.25
    # Per-position cap as a fraction of portfolio value.
    max_position_pct: float = 0.10
    # Total deployed capital cap as a fraction of portfolio value.
    max_total_exposure_pct: float = 0.60

    # --- Loss controls ---
    # Stop opening positions once strategy P&L for the UTC day breaches
    # this fraction of the effective bankroll.
    daily_loss_limit_pct: float = 0.10
    # Halt all trading when portfolio value falls this far below peak equity.
    drawdown_circuit_breaker_pct: float = 0.25

    # --- Entry gates ---
    # Floor for hold-to-resolution BUY entries. Cheap longshots held to a
    # zero-payout resolution are the largest historical bleed. 0 disables.
    min_hold_entry_price: float = 0.80
    # Max open positions per inferred market category. 0 disables.
    max_per_category: int = 3

    # --- Cooldowns ---
    # Block BUY signals on a token for this many seconds after a stop-loss
    # SELL on that token. Prevents losing-streak re-entries. 0 disables.
    stop_loss_reentry_cooldown_sec: float = 21600.0

    def validate(self) -> list[str]:
        """Return a list of problems; empty means the config is usable."""
        errors: list[str] = []
        if self.bankroll_usdc < self.min_bankroll_usdc:
            errors.append(
                f"bankroll_usdc ({self.bankroll_usdc}) is below "
                f"min_bankroll_usdc ({self.min_bankroll_usdc})"
            )
        for name, val in [
            ("max_position_pct", self.max_position_pct),
            ("max_total_exposure_pct", self.max_total_exposure_pct),
            ("daily_loss_limit_pct", self.daily_loss_limit_pct),
            ("drawdown_circuit_breaker_pct", self.drawdown_circuit_breaker_pct),
        ]:
            if not 0.0 < val <= 1.0:
                errors.append(f"{name} must be between 0 and 1, got {val}")
        if not 0.0 < self.kelly_multiplier <= 2.0:
            errors.append(
                f"kelly_multiplier must be between 0 and 2, got {self.kelly_multiplier}"
            )
        if self.min_bankroll_usdc < 0:
            errors.append(f"min_bankroll_usdc must be >= 0, got {self.min_bankroll_usdc}")
        if not 0.0 <= self.min_hold_entry_price < 1.0:
            errors.append(
                f"min_hold_entry_price must be in [0, 1), got {self.min_hold_entry_price}"
            )
        if self.max_per_category < 0:
            errors.append(f"max_per_category must be >= 0, got {self.max_per_category}")
        if self.stop_loss_reentry_cooldown_sec < 0:
            errors.append(
                "stop_loss_reentry_cooldown_sec must be >= 0, "
                f"got {self.stop_loss_reentry_cooldown_sec}"
            )
        return errors
