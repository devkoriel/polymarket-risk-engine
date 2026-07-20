"""Kelly Criterion position sizing with conservative guardrails."""

from __future__ import annotations

from polymarket_risk._logging import get_logger
from polymarket_risk.config import RiskConfig
from polymarket_risk.models import TradeSignal

log = get_logger(__name__)


class PositionSizer:
    """Size trades using Kelly Criterion with configurable aggression.

    Kelly fraction: f = (bp - q) / b
    where b = odds, p = win probability, q = 1-p

    kelly_multiplier controls aggression: 0.5 = half-Kelly, 1.0 = full Kelly.
    """

    def __init__(self, config: RiskConfig) -> None:
        self._config = config

    def compute_size(
        self,
        signal: TradeSignal,
        bankroll: float,
        current_exposure: float,
        portfolio_value: float = 0.0,
        same_category_positions: int = 0,
    ) -> float:
        """Return USDC amount to risk on this signal, or 0 if no trade.

        Args:
            bankroll: Available cash (CLOB balance).
            current_exposure: Total cost basis of open positions.
            portfolio_value: Cash + market value of all positions.
                Used for exposure cap so deployed capital doesn't
                shrink the cap to zero.  Falls back to bankroll
                when not provided (backward compat).
            same_category_positions: Count of open positions in the signal's
                inferred category. Used to scale Kelly down when concentrated
                (correlated-portfolio Kelly penalty).
        """
        effective_portfolio = portfolio_value if portfolio_value > 0 else bankroll

        # Per-position cap: fraction of portfolio, but can't exceed available cash
        max_position = min(
            effective_portfolio * self._config.max_position_pct,
            bankroll * 0.95,
        )
        # Total exposure cap: fraction of portfolio (not just cash)
        max_total = effective_portfolio * self._config.max_total_exposure_pct
        remaining_capacity = max(0.0, max_total - current_exposure)

        if remaining_capacity <= 0:
            log.debug("sizer_at_capacity", exposure=current_exposure)
            return 0.0

        # Kelly calculation
        kelly = self._kelly_fraction(signal)
        if kelly <= 0:
            log.debug("sizer_no_edge", strategy=signal.strategy.value, price=signal.price)
            return 0.0

        # Minimum Kelly floor — skip trades with negligible edge
        if kelly < 0.01:
            log.debug(
                "sizer_below_kelly_floor",
                kelly=round(kelly, 4),
                strategy=signal.strategy.value,
            )
            return 0.0

        # Apply Kelly multiplier (0.25 = quarter-Kelly for bankroll preservation)
        # Kelly sizes relative to total portfolio, not just cash.
        # Cash constraint is enforced via max_position cap above.
        #
        # Correlated-portfolio penalty: scalar Kelly over-sizes when positions
        # co-move. 14 political YES bets at full Kelly each is effectively a
        # 14x-levered bet on political calm. Scale Kelly down by category cluster:
        #   <2 same-cat positions: no scaling (1.0)
        #   2 same-cat:            0.7x (correlation-aware haircut)
        #   3+ same-cat:           0.4x (approaches quarter-Kelly on the cluster)
        if same_category_positions >= 3:
            cluster_scale = 0.4
        elif same_category_positions == 2:
            cluster_scale = 0.7
        else:
            cluster_scale = 1.0

        scaled_kelly = kelly * self._config.kelly_multiplier * cluster_scale
        kelly_amount = effective_portfolio * scaled_kelly

        # Apply all caps
        size = min(kelly_amount, max_position, remaining_capacity)

        # Floor: CLOB minimum order size is 5 tokens.
        # At high prices (e.g. $0.97), $1 buys only ~1 token → rejected.
        # Use 5.5 tokens worth of USDC as floor (margin for rounding).
        min_usdc = max(1.00, 5.5 * signal.price)
        if size < min_usdc:
            return 0.0

        log.info(
            "sizer_computed",
            kelly=round(kelly, 4),
            scaled_kelly=round(scaled_kelly, 4),
            multiplier=self._config.kelly_multiplier,
            cluster_scale=cluster_scale,
            same_cat=same_category_positions,
            kelly_amt=round(kelly_amount, 2),
            capped=round(size, 2),
            bankroll=round(bankroll, 2),
            portfolio=round(effective_portfolio, 2),
            strategy=signal.strategy.value,
            win_prob=round(signal.win_probability, 3),
            price=round(signal.price, 3),
        )
        return round(size, 2)

    def _kelly_fraction(self, signal: TradeSignal) -> float:
        """Compute Kelly fraction from win_probability and price."""
        price = signal.price
        if price <= 0 or price >= 1:
            return 0.0

        # For binary outcomes: odds = (1-price)/price
        odds = (1.0 - price) / price
        # Clamp win_probability to prevent full-bankroll bets at p=1.0
        p = min(signal.win_probability, 0.99)
        q = 1.0 - p

        if odds <= 0 or p <= 0:
            return 0.0

        f = (odds * p - q) / odds
        # Hard cap: no single trade > 25% of bankroll at full Kelly
        return min(max(0.0, f), 0.25)
