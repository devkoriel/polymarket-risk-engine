"""Data models used by the risk layer.

These are trimmed versions of the parent bot's internal models. The full bot
tags every signal with one of many strategy types; the risk layer only cares
about a few behavioral distinctions, so this package ships a coarse enum plus
a per-signal ``holds_to_resolution`` flag instead of the full roster.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum


class Side(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


class OrderType(str, Enum):
    """Polymarket CLOB order types."""

    LIMIT = "LIMIT"
    FOK = "FOK"
    FAK = "FAK"
    POST_ONLY = "POST_ONLY"


class StrategyType(str, Enum):
    """Coarse strategy tags.

    ARBITRAGE: simultaneous offsetting legs, fast exit.
    DIRECTIONAL: a view on the outcome; may hold toward resolution.
    MARKET_MAKING: passive two-sided quoting.
    RECONCILED: positions imported from outside the bot (web UI trades,
        transfers, pre-bot history). Their passive resolutions are excluded
        from the strategy daily loss limit so legacy inventory cannot lock
        the bot out of trading.
    """

    ARBITRAGE = "ARBITRAGE"
    DIRECTIONAL = "DIRECTIONAL"
    MARKET_MAKING = "MARKET_MAKING"
    RECONCILED = "RECONCILED"


def _utcnow() -> datetime:
    return datetime.now(UTC)


@dataclass(kw_only=True)
class Market:
    """Minimal market descriptor. The risk layer reads ``id`` and ``question``."""

    id: int
    question: str
    clob_token_ids: list[str] = field(default_factory=list)


@dataclass(kw_only=True)
class TradeSignal:
    """Instruction from a strategy to place an order.

    ``size`` is advisory on input; ``RiskManager.evaluate_signal`` returns a
    new signal with the size it actually allows.
    """

    strategy: StrategyType
    market: Market
    token_id: str
    side: Side
    price: float
    size: float
    confidence: float = 0.0
    win_probability: float = 0.5
    order_type: OrderType = OrderType.LIMIT
    reason: str = ""
    # True when the strategy plans to hold the position until the market
    # resolves (no fast exit). Gates the min_hold_entry_price check in
    # RiskManager: cheap longshots held to a zero-payout resolution are the
    # single largest historical bleed in the parent bot.
    holds_to_resolution: bool = False
    created_at: datetime = field(default_factory=_utcnow)


@dataclass(kw_only=True)
class Position:
    """Open position in a market outcome."""

    market_id: int
    token_id: str
    question: str = ""
    side: Side
    entry_price: float
    size: float
    current_price: float = 0.0
    unrealized_pnl: float = 0.0
    strategy: StrategyType = StrategyType.DIRECTIONAL
    opened_at: datetime = field(default_factory=_utcnow)

    @property
    def market_value(self) -> float:
        return self.size * self.current_price

    @property
    def cost_basis(self) -> float:
        return self.size * self.entry_price
