"""Size a hypothetical signal through the risk manager.

Runs offline. No credentials, no network.
"""

from polymarket_risk import (
    Market,
    RiskConfig,
    RiskManager,
    Side,
    StrategyType,
    TradeSignal,
)


def main() -> None:
    config = RiskConfig(
        bankroll_usdc=1000.0,
        kelly_multiplier=0.25,  # quarter-Kelly
        max_position_pct=0.10,
        max_total_exposure_pct=0.60,
        daily_loss_limit_pct=0.10,
        drawdown_circuit_breaker_pct=0.25,
    )
    problems = config.validate()
    if problems:
        raise SystemExit("bad config:\n" + "\n".join(problems))

    risk = RiskManager(config)
    # In the live bot, positions are reconciled against the venue before the
    # drawdown breaker arms. Nothing to reconcile in this demo.
    risk.confirm_reconciliation()

    market = Market(
        id=1,
        question="Will the Fed cut rates at the next meeting?",
        clob_token_ids=["yes-token", "no-token"],
    )
    signal = TradeSignal(
        strategy=StrategyType.DIRECTIONAL,
        market=market,
        token_id="yes-token",
        side=Side.BUY,
        price=0.62,            # market price of YES
        size=0.0,              # the risk manager decides
        confidence=0.80,
        win_probability=0.70,  # your model's estimate, not your confidence
    )

    sized = risk.evaluate_signal(signal, bankroll=1000.0)
    if sized is None:
        print("rejected by the risk layer")
        return

    print(f"approved: ${sized.size:.2f} at {sized.price:.2f}")
    print(f"kelly multiplier: {config.kelly_multiplier} (quarter-Kelly)")

    # The same signal at a longshot price, held to resolution, gets blocked
    # by the min-hold entry gate.
    longshot = TradeSignal(
        strategy=StrategyType.DIRECTIONAL,
        market=market,
        token_id="no-token",
        side=Side.BUY,
        price=0.08,
        size=0.0,
        confidence=0.80,
        win_probability=0.30,
        holds_to_resolution=True,
    )
    blocked = risk.evaluate_signal(longshot, bankroll=1000.0)
    print(f"longshot at 0.08 held to resolution: {'blocked' if blocked is None else 'approved'}")


if __name__ == "__main__":
    main()
