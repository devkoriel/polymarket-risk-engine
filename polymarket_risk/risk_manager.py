"""Central risk manager: validates signals against portfolio limits."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import replace
from datetime import UTC, date, datetime

from polymarket_risk._logging import get_logger
from polymarket_risk.config import RiskConfig
from polymarket_risk.models import Position, Side, StrategyType, TradeSignal
from polymarket_risk.position_sizer import PositionSizer

log = get_logger(__name__)


# Category keyword buckets for diversification cap. Order matters: first match
# wins, so put narrower/higher-signal terms earlier.
_CATEGORY_KEYWORDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("weather", ("temperature", "weather", "rainfall", "snow")),
    ("crypto",  ("bitcoin", "ethereum", " eth ", "btc", "solana", " sol ", "crypto")),
    ("politics", ("election", "president", "nominee", "impeach", "prime minister",
                  "chancellor", "parliament", "senate", "congress")),
    ("sports",  ("nba", "nfl", "mlb", "nhl", "champion", "world cup", "premier league",
                 "la liga", "serie a", "bundesliga", "wimbledon", "ufc", "tennis",
                 "football", "basketball", "hockey", "baseball", "soccer")),
    ("macro",   ("fed", "rate cut", "inflation", "cpi", "gdp", "recession", "jobless")),
)


def _infer_category(question: str) -> str:
    """Map a market question to a coarse category, or '' if unrecognized."""
    q = question.lower()
    for name, needles in _CATEGORY_KEYWORDS:
        if any(needle in q for needle in needles):
            return name
    return ""


class RiskManager:
    """Gate between strategy signals and order execution.

    Enforces:
    - Per-position size limits (max_position_pct)
    - Total portfolio exposure cap (max_total_exposure_pct)
    - Daily loss limit (daily_loss_limit_pct)
    - Drawdown circuit breaker
    - Minimum bankroll floor
    - Category concentration cap
    - Stop-loss re-entry cooldowns
    """

    def __init__(self, config: RiskConfig) -> None:
        self._config = config
        self._sizer = PositionSizer(config)
        self._positions: dict[str, Position] = {}
        self._daily_pnl: float = 0.0
        # Strategy-only daily P&L excludes passive RECONCILED position resolutions.
        # Used for the daily loss limit check so legacy positions resolving don't
        # block the agent from making profitable strategy trades.
        self._strategy_daily_pnl: float = 0.0
        self._peak_equity: float = config.bankroll_usdc
        self._current_day: date = datetime.now(UTC).date()
        self._circuit_broken = False
        self._reconciliation_confirmed: bool = False
        self._last_bankroll_floor_log: float = 0.0
        self._last_daily_limit_log: float = 0.0
        self._on_position_closed: Callable[[Position, float], None] | None = None
        # Pending removals: positions absent from the positions API that need
        # confirmation before recording P&L (prevents phantom losses from API
        # flicker). Maps token_id -> (Position, first_absent_timestamp)
        self._pending_removals: dict[str, tuple[Position, float]] = {}
        # Tokens already finalized as removed in this UTC day. Guards against
        # the positions API re-surfacing a resolved/sold position and the
        # passive sync re-adding it, which would re-trigger pending->finalize
        # and double-count P&L. Cleared on daily reset; cleared per-token if
        # the agent legitimately re-buys via register_position().
        self._removed_tokens: set[str] = set()
        # token_id -> cooldown_until_timestamp. Set by record_stop_loss() so
        # strategies can't immediately re-enter a market that just stopped out.
        self._stop_loss_cooldowns: dict[str, float] = {}
        self._last_stop_loss_block_log: float = 0.0

    @property
    def is_circuit_broken(self) -> bool:
        return self._circuit_broken

    @property
    def should_halt(self) -> bool:
        """Check if trading should stop (for use by passive pipelines)."""
        return self._circuit_broken

    def confirm_reconciliation(self) -> None:
        """Mark positions as verified against the venue.

        Until this is called, the drawdown check is deferred: establishing
        peak equity from unverified positions (e.g. resolved markets still
        showing as open) would arm the circuit breaker on phantom equity.
        """
        self._reconciliation_confirmed = True

    # Positions below this price are "frozen" — too illiquid to sell,
    # so they should not block new trades via the exposure cap.
    _FROZEN_PRICE_THRESHOLD = 0.005
    # CLOB rejects sub-$1 notional orders. Residual positions below this value
    # are dust and should not consume a whole category-concentration slot.
    _DUST_MARKET_VALUE_THRESHOLD = 1.0

    def _is_frozen_position(self, position: Position) -> bool:
        """Return True when a position is too illiquid to reduce."""
        return 0 < position.current_price < self._FROZEN_PRICE_THRESHOLD

    def _counts_for_category_cap(self, position: Position) -> bool:
        """Return True when a position has enough live value to count."""
        if self._is_frozen_position(position):
            return False
        if (
            position.current_price > 0
            and position.market_value < self._DUST_MARKET_VALUE_THRESHOLD
        ):
            return False
        return True

    @property
    def total_exposure(self) -> float:
        """Active capital at risk used for the exposure cap.

        Excludes "frozen" positions (current_price < 0.005) that cannot
        be sold — including them would permanently block new trades
        while the agent has no way to reduce the exposure.

        Uses market_value where available so depreciated positions
        don't penalise the cap at full cost-basis weight.
        """
        total = 0.0
        for p in self._positions.values():
            # Skip frozen positions: near-zero prices can't be sold,
            # profit-taking already skips them, so don't count toward cap.
            if self._is_frozen_position(p):
                continue
            if p.current_price > 0:
                mv = p.market_value
                if mv != mv:  # NaN check
                    continue
                total += mv
            else:
                cb = p.cost_basis
                if cb != cb:  # NaN check
                    log.error("risk_nan_cost_basis", token_id=p.token_id[:12],
                              size=p.size, entry_price=p.entry_price)
                    continue
                total += cb
        return total

    @property
    def total_market_value(self) -> float:
        total = 0.0
        for p in self._positions.values():
            mv = p.market_value
            if mv != mv:  # NaN check
                continue
            total += mv
        return total

    @property
    def total_unrealized_pnl(self) -> float:
        return sum(p.unrealized_pnl for p in self._positions.values())

    @property
    def position_count(self) -> int:
        return len(self._positions)

    def evaluate_signal(
        self,
        signal: TradeSignal,
        bankroll: float,
    ) -> TradeSignal | None:
        """Validate signal and attach proper sizing. Returns None if rejected."""

        # Circuit breaker check
        if self._circuit_broken:
            log.warning("risk_circuit_broken", signal_rejected=signal.market.id)
            return None

        # Bankroll floor: cash alone, but allow trading when positions cover the gap.
        # Effective USDC = cash + 25% of current position market value.
        # Prevents blocking when cash is temporarily low due to deployed capital.
        if bankroll < self._config.min_bankroll_usdc:
            effective_usdc = bankroll + self.total_market_value * 0.25
            if effective_usdc < self._config.min_bankroll_usdc:
                now = time.time()
                if now - self._last_bankroll_floor_log > 60.0:
                    log.warning(
                        "risk_bankroll_floor",
                        bankroll=bankroll,
                        floor=self._config.min_bankroll_usdc,
                        market_value=round(self.total_market_value, 2),
                    )
                    self._last_bankroll_floor_log = now
                return None

        # Daily loss check — use total portfolio value (cash + positions) so the limit
        # reflects actual capital at risk, not just undeployed cash.  When most capital
        # is in positions, CLOB balance can be near-zero while portfolio is healthy.
        # Floor at bankroll_usdc (configured deposit amount) to prevent transient dips.
        self._maybe_reset_daily()
        portfolio_value = bankroll + self.total_market_value
        effective_bankroll = max(
            portfolio_value, self._config.bankroll_usdc, self._config.min_bankroll_usdc
        )
        daily_limit = effective_bankroll * self._config.daily_loss_limit_pct
        if self._strategy_daily_pnl < -daily_limit:
            now = time.time()
            if now - self._last_daily_limit_log > 60.0:
                log.warning(
                    "risk_daily_limit",
                    strategy_pnl=round(self._strategy_daily_pnl, 2),
                    total_pnl=round(self._daily_pnl, 2),
                    limit=round(-daily_limit, 2),
                )
                self._last_daily_limit_log = now
            return None

        # Drawdown circuit breaker — compare portfolio (cash + market value), not cost basis.
        # Use max(bankroll, 0) so a transient API zero-read doesn't produce a negative
        # portfolio_value while market_value is still healthy.
        portfolio_value = max(bankroll, 0.0) + self.total_market_value

        if not self._reconciliation_confirmed:
            # Don't establish peak or check drawdown until positions are verified
            # by the first reconciliation cycle (removes resolved-market positions)
            log.debug("drawdown_check_deferred", portfolio=round(portfolio_value, 2))
        else:
            # Establish peak from first live reading after reconciliation (0 = not yet set)
            if self._peak_equity <= 0:
                self._peak_equity = portfolio_value
                log.info("peak_equity_established", peak=portfolio_value)

            if portfolio_value < self._peak_equity * (1.0 - self._config.drawdown_circuit_breaker_pct):
                self._circuit_broken = True
                log.error(
                    "risk_drawdown_breaker",
                    portfolio=portfolio_value,
                    bankroll=bankroll,
                    peak=self._peak_equity,
                    threshold=self._config.drawdown_circuit_breaker_pct,
                )
                return None

            # Track peak equity (ratchet up only)
            if portfolio_value > self._peak_equity:
                self._peak_equity = portfolio_value

        # Duplicate position check (same token)
        if signal.token_id in self._positions:
            log.debug("risk_duplicate_position", token_id=signal.token_id[:12])
            return None

        # Stop-loss re-entry cooldown. A stop-loss SELL on a token blocks
        # fresh BUY signals on the same token for a configured window.
        # Prevents losing-streak re-entries where the strategy re-fires
        # minutes after a stop, producing compounding losses.
        if signal.side == Side.BUY:
            cooldown_until = self._stop_loss_cooldowns.get(signal.token_id)
            if cooldown_until is not None:
                now_ts = time.time()
                if now_ts < cooldown_until:
                    if now_ts - self._last_stop_loss_block_log > 60.0:
                        log.info(
                            "risk_stop_loss_cooldown",
                            strategy=signal.strategy.value,
                            token_id=signal.token_id[:12],
                            remaining_sec=int(cooldown_until - now_ts),
                        )
                        self._last_stop_loss_block_log = now_ts
                    return None
                # Expired — clean up
                self._stop_loss_cooldowns.pop(signal.token_id, None)

        # Min-hold-entry-price gate. Longshot YES at 0.01-0.25 held to
        # resolution is the largest historical bleed. Applies only to
        # signals that hold toward resolution; everything else is expected
        # to carry an explicit fast exit.
        min_hold = self._config.min_hold_entry_price
        if (
            min_hold > 0
            and signal.side == Side.BUY
            and signal.holds_to_resolution
            and signal.price < min_hold
        ):
            log.info(
                "risk_min_hold_price",
                strategy=signal.strategy.value,
                price=signal.price,
                min=min_hold,
            )
            return None

        # Category concurrency cap. Prevents 8 simultaneous political YES
        # positions from all resolving NO on the same news event.
        # Also used below as the Kelly cluster-scale input.
        cat = _infer_category(signal.market.question)
        open_in_cat = 0
        if cat:
            open_in_cat = sum(
                1 for p in self._positions.values()
                if self._counts_for_category_cap(p) and _infer_category(p.question) == cat
            )
        cat_cap = self._config.max_per_category
        if cat_cap > 0 and cat and open_in_cat >= cat_cap:
            log.info(
                "risk_category_cap",
                category=cat,
                open=open_in_cat,
                cap=cat_cap,
            )
            return None

        # Size the trade — use portfolio value (cash + positions) for the
        # exposure cap so deployed capital doesn't shrink max_total to zero.
        # Pass same-category count to enable Kelly cluster-scale penalty.
        portfolio_value = max(bankroll, 0.0) + self.total_market_value
        size = self._sizer.compute_size(
            signal, bankroll, self.total_exposure,
            portfolio_value=portfolio_value,
            same_category_positions=open_in_cat,
        )
        if size <= 0:
            return None

        # Return a new signal with proper sizing (input signal is not mutated)
        return replace(signal, size=size)

    def register_position(self, position: Position) -> None:
        self._positions[position.token_id] = position
        # Legitimate re-entry by the agent clears any prior removal guard.
        self._removed_tokens.discard(position.token_id)

    def close_position(self, token_id: str, pnl: float) -> None:
        self._positions.pop(token_id, None)
        self._daily_pnl += pnl
        # Agent-initiated sells always count toward the strategy limit
        self._strategy_daily_pnl += pnl

    def reduce_position(self, token_id: str, sold_size: float, pnl: float) -> None:
        """Reduce position size after a partial sell.

        If remaining size drops below 0.5 tokens, close entirely.
        """
        pos = self._positions.get(token_id)
        if pos is None:
            return
        remaining = pos.size - sold_size
        if remaining < 0.5:
            self._positions.pop(token_id, None)
        else:
            pos.size = remaining
        self._daily_pnl += pnl
        # Agent-initiated sells always count toward the strategy limit
        self._strategy_daily_pnl += pnl

    def get_positions(self) -> list[Position]:
        return list(self._positions.values())

    def get_position(self, token_id: str) -> Position | None:
        return self._positions.get(token_id)

    def get_position_ids(self) -> set[str]:
        return set(self._positions.keys())

    def record_stop_loss(self, token_id: str, now: float | None = None) -> None:
        """Mark ``token_id`` as on stop-loss re-entry cooldown.

        Called by the order manager after a stop-loss SELL executes, so the
        next few hours of BUY signals on the same token are rejected. The
        cooldown window is ``config.stop_loss_reentry_cooldown_sec``; pass
        0 (or set the config field to 0) to disable.
        """
        cooldown = self._config.stop_loss_reentry_cooldown_sec
        if cooldown <= 0:
            return
        ts = now if now is not None else time.time()
        self._stop_loss_cooldowns[token_id] = ts + cooldown
        log.info(
            "stop_loss_cooldown_recorded",
            token_id=token_id[:12],
            cooldown_sec=int(cooldown),
        )

    def reset_circuit_breaker(self) -> None:
        self._circuit_broken = False
        log.info("risk_circuit_breaker_reset")

    def check_circuit_recovery(self, bankroll: float) -> bool:
        """Auto-reset circuit breaker if portfolio recovers above threshold.

        Returns True if breaker was reset (trading can resume).
        """
        if not self._circuit_broken:
            return False
        portfolio = bankroll + self.total_market_value
        threshold = self._peak_equity * (1.0 - self._config.drawdown_circuit_breaker_pct)
        if portfolio >= threshold:
            self._circuit_broken = False
            log.info(
                "circuit_breaker_auto_reset",
                portfolio=round(portfolio, 2),
                peak=round(self._peak_equity, 2),
                threshold=round(threshold, 2),
            )
            return True
        return False

    def reconcile_positions(
        self,
        actual: dict[str, Position],
        grace_seconds: float = 0.0,
    ) -> tuple[int, int, int]:
        """Reconcile internal positions with the venue's actual state.

        Compares the risk manager's positions with the actual net positions
        computed from venue trade history. Handles:
        - User sold on web → position removed
        - User bought on web → position added
        - Partial sell → position size updated
        - Market resolved → position removed (not in actual)

        Args:
            actual: dict of token_id → Position from venue trade aggregation
            grace_seconds: skip removal of positions opened within this window,
                preventing race between trade execution and reconciliation

        Returns (added, removed, updated) counts.
        """
        added = removed = updated = 0
        now = datetime.now(UTC)

        # Remove positions that no longer exist on-chain.
        # Uses a two-phase removal to prevent phantom P&L from positions-API flicker:
        # Phase 1: position disappears → moved to pending_removals
        # Phase 2: still absent after REMOVAL_CONFIRM_SECONDS → finalized
        removal_confirm_secs = 300.0  # 5 min confirmation window
        now_ts = time.time()
        stale_ids = []
        for tid in self._positions:
            if tid not in actual:
                pos = self._positions[tid]
                try:
                    age = (now - pos.opened_at).total_seconds()
                except TypeError:
                    age = grace_seconds + 1  # Naive datetime — treat as old
                if age > grace_seconds:
                    stale_ids.append(tid)
                else:
                    log.debug(
                        "reconcile_skipped_recent",
                        token_id=tid[:12],
                        age_sec=round(age, 0),
                        grace=grace_seconds,
                    )

        for tid in stale_ids:
            if tid not in self._pending_removals:
                # Phase 1: first time absent — move to pending, don't finalize yet
                self._pending_removals[tid] = (self._positions[tid], now_ts)
                log.info(
                    "position_removal_pending",
                    token_id=tid[:12],
                    size=self._positions[tid].size,
                    entry=round(self._positions[tid].entry_price, 4),
                )
                # Keep in _positions for now — will be finalized next cycle
                continue

            # Phase 2: check if confirmation window has elapsed
            _, first_absent_ts = self._pending_removals[tid]
            if now_ts - first_absent_ts < removal_confirm_secs:
                # Still within confirmation window — skip
                continue

            # Confirmed absent — finalize removal
            del self._pending_removals[tid]
            old = self._positions.pop(tid)
            removed += 1
            # Guard against the positions API re-surfacing this token later in
            # the day (e.g. delayed event indexing). Without this, the passive
            # sync below would re-add the position and the next pending→finalize
            # cycle would double-count P&L. Cleared on UTC reset or on a
            # legitimate re-buy via register_position().
            self._removed_tokens.add(tid)

            # Estimate exit price: resolved YES tokens exit at 1.0
            # Others: use current_price if available, else entry_price (sold/expired)
            exit_price = old.current_price if old.current_price > 0 else old.entry_price
            pnl = (exit_price - old.entry_price) * old.size

            # Record realized P&L
            self._daily_pnl += pnl
            # Only count strategy positions toward the daily loss limit.
            # RECONCILED positions are legacy (web UI / pre-strategy) — their
            # passive resolution should not block profitable strategy trades.
            if old.strategy != StrategyType.RECONCILED:
                self._strategy_daily_pnl += pnl

            if self._on_position_closed is not None:
                try:
                    self._on_position_closed(old, exit_price)
                except Exception:
                    log.warning("on_position_closed_error", token_id=tid[:12], exc_info=True)

            log.info(
                "position_reconciled_removed",
                token_id=tid[:12],
                size=old.size,
                entry=round(old.entry_price, 4),
                exit=round(exit_price, 4),
                pnl=round(pnl, 4),
                cost_basis=round(old.cost_basis, 2),
            )

        # Clear pending removals for positions that reappeared in the positions API
        reappeared = [tid for tid in self._pending_removals if tid in actual]
        for tid in reappeared:
            log.info(
                "position_removal_cancelled",
                token_id=tid[:12],
                reason="reappeared_in_positions_api",
            )
            del self._pending_removals[tid]

        # Add new or update changed positions
        for tid, pos in actual.items():
            if tid in self._removed_tokens:
                # Confirmed-removed earlier today; ignore API re-surfacing
                # to prevent the pending→finalize cycle from double-counting.
                log.debug(
                    "position_reconciled_add_skipped",
                    token_id=tid[:12],
                    reason="previously_removed_today",
                )
                continue
            existing = self._positions.get(tid)
            if existing is None:
                self._positions[tid] = pos
                added += 1
                log.info(
                    "position_reconciled_added",
                    token_id=tid[:12],
                    size=pos.size,
                    entry=round(pos.entry_price, 4),
                )
            elif abs(existing.size - pos.size) > 0.5:
                # Size changed (partial sell/buy on web) — preserve metadata
                self._positions[tid] = Position(
                    market_id=existing.market_id,
                    token_id=tid,
                    question=existing.question,
                    side=existing.side,
                    entry_price=pos.entry_price,
                    size=pos.size,
                    current_price=pos.current_price,
                    unrealized_pnl=pos.unrealized_pnl,
                    strategy=existing.strategy,
                    opened_at=existing.opened_at,
                )
                updated += 1
                log.info(
                    "position_reconciled_updated",
                    token_id=tid[:12],
                    old_size=existing.size,
                    new_size=pos.size,
                )
            else:
                # Size unchanged — still refresh price and unrealized P&L
                if pos.current_price > 0:
                    existing.current_price = pos.current_price
                    existing.unrealized_pnl = pos.unrealized_pnl

        return added, removed, updated

    def _maybe_reset_daily(self) -> None:
        today = datetime.now(UTC).date()
        if today > self._current_day:
            self._daily_pnl = 0.0
            self._strategy_daily_pnl = 0.0
            self._current_day = today
            self._removed_tokens.clear()
            log.info("risk_daily_reset")
