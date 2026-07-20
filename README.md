# polymarket-risk-engine

This is the risk layer of a real trading bot that runs live on Polymarket mainnet. The bot's lifetime PnL is negative. That is exactly why this code matters: the strategies were sometimes wrong, and the risk layer is what kept a wrong bot solvent and running long enough to learn from it. Plenty of people open-source strategies. Almost nobody open-sources the layer that keeps a live bot alive. The parent project has 987 tests. This package extracts the risk engine and the ghost-fill guard, with the tests that cover them, as a standalone library with zero runtime dependencies.

## What is included

- Kelly position sizing with a configurable multiplier. The live bot runs quarter-Kelly (0.25x), a deliberate de-risk after half-Kelly drawdowns got uncomfortable.
- Correlated-portfolio haircuts. Scalar Kelly over-sizes when positions co-move. Ten political YES bets at full Kelly each is one levered bet on political calm, so Kelly is scaled down (0.7x, then 0.4x) as positions pile up in the same category.
- Category concentration caps, so one news event cannot resolve your whole book at once. Frozen and dust positions are excluded so unsellable residue does not block fresh trades.
- A daily loss limit measured against total portfolio value, not just cash. Agent-initiated P&L is tracked separately from passively resolving legacy positions, so old inventory cannot lock the bot out of trading.
- A drawdown circuit breaker with a peak-equity ratchet (peak only moves up) and automatic recovery reset. It arms only after positions are reconciled against the venue, so phantom equity cannot set the peak.
- Stop-loss re-entry cooldowns. A strategy that stops out cannot re-buy the same market minutes later at the now-cheaper price and stop out again.
- Ghost-fill defense (`GhostFillGuard`). On Polymarket, an order can be matched off-chain by the CLOB and still never settle on-chain. In the incrementNonce exploit, a counterparty matches your order, then calls `incrementNonce()` on the exchange contract before settlement, which makes the settlement transaction revert. The attacker keeps fills that went their way and voids the ones that did not. The guard snapshots your ERC-1155 token balance before the order and polls the chain after the CLOB reports a match. If the balance never increases within the timeout, the fill was a ghost and the position is not registered. It can also snapshot the exchange nonce and flag a mismatch directly. On-chain reads are injected callables, so the package itself needs no web3 dependency and everything degrades gracefully when the RPC is down.

## What is deliberately not included

The strategies. No signal generation, no market scanning, no order placement, no market data clients, no edge of any kind. This package will not make you money. It is the part that decides how much you are allowed to lose.

## Install

```
pip install polymarket-risk-engine
```

Or from a checkout:

```
pip install .
```

Requires Python 3.11 or newer. No runtime dependencies.

## Quickstart

```python
from polymarket_risk import (
    Market, RiskConfig, RiskManager, Side, StrategyType, TradeSignal,
)

config = RiskConfig(
    bankroll_usdc=1000.0,
    kelly_multiplier=0.25,   # quarter-Kelly
    max_position_pct=0.10,
    max_total_exposure_pct=0.60,
    daily_loss_limit_pct=0.10,
    drawdown_circuit_breaker_pct=0.25,
)

risk = RiskManager(config)
risk.confirm_reconciliation()

signal = TradeSignal(
    strategy=StrategyType.DIRECTIONAL,
    market=Market(id=1, question="Will the Fed cut rates at the next meeting?"),
    token_id="yes-token",
    side=Side.BUY,
    price=0.62,
    size=0.0,                # the risk manager decides
    confidence=0.80,
    win_probability=0.70,    # your model's estimate, not your confidence
)

sized = risk.evaluate_signal(signal, bankroll=1000.0)
if sized is None:
    print("rejected by the risk layer")
else:
    print(f"approved: ${sized.size:.2f}")
```

A runnable version with the min-hold gate demo is in `examples/quickstart.py`:

```
python examples/quickstart.py
```

Run the tests:

```
uv run --group dev pytest
```

The full bot and the book that documents it are at BOOK_LINK_PLACEHOLDER.
