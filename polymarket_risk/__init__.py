"""polymarket-risk-engine: the risk layer of a live Polymarket trading bot.

Kelly sizing with correlated-portfolio haircuts, daily loss limits, a
drawdown circuit breaker, stop-loss cooldowns, and on-chain ghost-fill
defense. No strategies included.
"""

from polymarket_risk.config import RiskConfig
from polymarket_risk.ghost_fill_guard import (
    BALANCE_OF_ABI,
    CTF_ADDRESS,
    GhostFillEvent,
    GhostFillGuard,
)
from polymarket_risk.models import (
    Market,
    OrderType,
    Position,
    Side,
    StrategyType,
    TradeSignal,
)
from polymarket_risk.position_sizer import PositionSizer
from polymarket_risk.risk_manager import RiskManager

__version__ = "0.1.0"

__all__ = [
    "BALANCE_OF_ABI",
    "CTF_ADDRESS",
    "GhostFillEvent",
    "GhostFillGuard",
    "Market",
    "OrderType",
    "Position",
    "PositionSizer",
    "RiskConfig",
    "RiskManager",
    "Side",
    "StrategyType",
    "TradeSignal",
]
