"""Tests for RiskConfig validation."""

from polymarket_risk import RiskConfig


def test_defaults_are_valid():
    assert RiskConfig().validate() == []


def test_bankroll_below_floor_rejected():
    errors = RiskConfig(bankroll_usdc=10.0, min_bankroll_usdc=50.0).validate()
    assert any("bankroll_usdc" in e for e in errors)


def test_pct_fields_must_be_in_unit_interval():
    errors = RiskConfig(
        max_position_pct=1.5,
        max_total_exposure_pct=0.0,
        daily_loss_limit_pct=-0.1,
        drawdown_circuit_breaker_pct=2.0,
    ).validate()
    assert len(errors) == 4


def test_kelly_multiplier_bounds():
    assert RiskConfig(kelly_multiplier=0.0).validate()
    assert RiskConfig(kelly_multiplier=2.5).validate()
    assert RiskConfig(kelly_multiplier=1.0).validate() == []


def test_min_hold_entry_price_bounds():
    assert RiskConfig(min_hold_entry_price=1.0).validate()
    assert RiskConfig(min_hold_entry_price=-0.1).validate()
    assert RiskConfig(min_hold_entry_price=0.0).validate() == []


def test_negative_cooldown_and_category_cap_rejected():
    errors = RiskConfig(
        max_per_category=-1,
        stop_loss_reentry_cooldown_sec=-1.0,
    ).validate()
    assert len(errors) == 2
