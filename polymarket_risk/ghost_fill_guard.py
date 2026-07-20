"""Ghost fill protection: verify on-chain settlement after a CLOB match.

Defends against the incrementNonce exploit where an attacker:
1. Matches an order off-chain (CLOB reports MATCHED)
2. Calls incrementNonce() on the CTF Exchange before on-chain settlement
3. Settlement tx reverts — attacker keeps winning trades, invalidates losing ones

Defense: after the CLOB reports MATCHED, poll the ERC-1155 token balance
on-chain to verify the wallet actually received the tokens. If the balance
doesn't increase within the timeout, the fill was a ghost — don't register it.

Also detects the USDC drain variant where the counterparty empties their wallet
before settlement, causing forced removal of our orders.

This standalone version takes injected callables for the on-chain reads, so
the package itself needs no web3 or RPC dependency. Wire it up like:

    from web3 import Web3
    from polymarket_risk.ghost_fill_guard import CTF_ADDRESS, BALANCE_OF_ABI

    w3 = Web3(Web3.HTTPProvider(rpc_url))
    ctf = w3.eth.contract(address=Web3.to_checksum_address(CTF_ADDRESS), abi=BALANCE_OF_ABI)

    def read_balance(token_id: str) -> float:
        raw = ctf.functions.balanceOf(wallet_address, int(token_id)).call()
        return raw / 1e6  # CTF outcome tokens use 6 decimals

    guard = GhostFillGuard(balance_reader=read_balance)

The optional ``nonce_reader`` should return ``nonces(wallet)`` from the
exchange contract that fills your orders. Raise (or let web3 raise) on RPC
errors; the guard maps rate-limit errors to a cooldown and degrades
gracefully on everything else.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime

from polymarket_risk._logging import get_logger

log = get_logger(__name__)

# Polymarket's ConditionalTokens (ERC-1155) contract on Polygon mainnet.
# Public, well-known address; used for balanceOf reads.
CTF_ADDRESS = "0x4D97DCd97eC945f40cF65F87097ACe5EA0476045"

# Minimal ABI for the balance read.
BALANCE_OF_ABI = [
    {
        "inputs": [
            {"name": "account", "type": "address"},
            {"name": "id", "type": "uint256"},
        ],
        "name": "balanceOf",
        "outputs": [{"name": "", "type": "uint256"}],
        "stateMutability": "view",
        "type": "function",
    },
]

# token_id -> balance in tokens (raw / 1e6), or None if unavailable.
BalanceReader = Callable[[str], float | None]
# () -> current exchange nonce for the wallet, or None if unavailable.
NonceReader = Callable[[], int | None]

# RPC cooldown after rate-limit failure
_RPC_COOLDOWN_SEC = 600  # 10 minutes

# Minimum balance delta tolerance for fill verification.
# The effective tolerance is max(_MIN_DELTA, expected_delta * _MIN_DELTA_PCT).
# _MIN_DELTA is the absolute dust floor; _MIN_DELTA_PCT handles proportional
# shortfalls from partial fills, taker fees, and on-chain rounding on crypto
# markets. Observed live: 2.5% shortfall (expected 15.45, actual 15.0653)
# falsely flagged as ghost fills → retries → over-allocation.
_MIN_DELTA = 0.001
_MIN_DELTA_PCT = 0.05


@dataclass
class GhostFillEvent:
    """Record of a detected ghost fill."""

    order_id: str
    token_id: str
    expected_delta: float
    balance_before: float
    balance_after: float
    detected_at: datetime = field(default_factory=lambda: datetime.now(UTC))


class GhostFillGuard:
    """On-chain settlement verifier for CLOB fills.

    Usage in an order manager:
        balance_before = await guard.snapshot_balance(token_id)
        # ... place order, wait for CLOB MATCHED ...
        verified = await guard.verify_settlement(token_id, expected_delta, balance_before)
        if not verified:
            # Ghost fill — don't register position
    """

    def __init__(
        self,
        balance_reader: BalanceReader,
        nonce_reader: NonceReader | None = None,
        *,
        verify_timeout: float = 30.0,
        verify_interval: float = 3.0,
    ) -> None:
        self._balance_reader = balance_reader
        self._nonce_reader = nonce_reader
        self._verify_timeout = verify_timeout
        self._verify_interval = verify_interval
        self._ghost_fills: list[GhostFillEvent] = []
        self._last_rpc_failure: float = 0.0
        self._verified_count: int = 0
        self._nonce_mismatch_count: int = 0

    @property
    def ghost_fill_count(self) -> int:
        return len(self._ghost_fills)

    @property
    def verified_count(self) -> int:
        return self._verified_count

    @property
    def recent_ghost_fills(self) -> list[GhostFillEvent]:
        """Last 50 ghost fill events."""
        return self._ghost_fills[-50:]

    def _rpc_available(self) -> bool:
        if self._last_rpc_failure == 0.0:
            return True
        return time.time() - self._last_rpc_failure >= _RPC_COOLDOWN_SEC

    def _note_read_failure(self, exc: Exception, event_prefix: str) -> None:
        err = str(exc)
        if "rate limit" in err.lower() or "-32090" in err:
            self._last_rpc_failure = time.time()
            log.warning(f"{event_prefix}_rpc_rate_limited", cooldown=_RPC_COOLDOWN_SEC)
        else:
            log.warning(f"{event_prefix}_read_failed", error=err[:200])

    def _read_balance_sync(self, token_id: str) -> float | None:
        """Synchronous on-chain balance read via the injected reader.

        Returns tokens (raw balance divided by 1e6) or None.
        """
        if not self._rpc_available():
            return None
        try:
            return self._balance_reader(token_id)
        except Exception as exc:
            self._note_read_failure(exc, "ghost_guard_balance")
            return None

    async def snapshot_balance(self, token_id: str) -> float | None:
        """Get current on-chain ERC-1155 balance before placing an order.

        Returns balance in tokens (divided by 1e6) or None if RPC unavailable.
        None means the guard will skip verification for this trade (graceful degradation).
        """
        loop = asyncio.get_running_loop()
        try:
            return await asyncio.wait_for(
                loop.run_in_executor(None, self._read_balance_sync, token_id),
                timeout=20.0,
            )
        except asyncio.TimeoutError:
            log.warning("ghost_guard_snapshot_timeout", token_id=token_id[:16])
            return None

    def _read_nonce_sync(self) -> int | None:
        """Synchronous on-chain nonce read for incrementNonce exploit detection."""
        if self._nonce_reader is None:
            return None
        if not self._rpc_available():
            return None
        try:
            return self._nonce_reader()
        except Exception as exc:
            self._note_read_failure(exc, "ghost_guard_nonce")
            return None

    async def snapshot_nonce(self) -> int | None:
        """Get current exchange nonce before placing an order."""
        loop = asyncio.get_running_loop()
        try:
            return await asyncio.wait_for(
                loop.run_in_executor(None, self._read_nonce_sync),
                timeout=20.0,
            )
        except asyncio.TimeoutError:
            log.warning("ghost_guard_nonce_timeout")
            return None

    async def verify_settlement(
        self,
        token_id: str,
        expected_delta: float,
        balance_before: float | None,
        order_id: str = "",
        nonce_before: int | None = None,
    ) -> bool:
        """Poll on-chain balance until it reflects the fill.

        Args:
            token_id: ERC-1155 token ID on the CTF contract.
            expected_delta: Expected balance increase (trade size).
            balance_before: Balance snapshot from before the order (None = skip).
            order_id: For logging.
            nonce_before: Exchange nonce snapshot (None = skip nonce check).

        Returns:
            True if settlement confirmed or verification unavailable (graceful).
            False if ghost fill detected (balance didn't increase or nonce changed).
        """
        if balance_before is None:
            log.debug("ghost_guard_skip_no_snapshot", order_id=order_id)
            return True  # Can't verify without baseline — proceed anyway

        if not self._rpc_available():
            log.debug("ghost_guard_skip_rpc_cooldown", order_id=order_id)
            return True  # RPC in cooldown — degrade gracefully

        # Early nonce check — if nonce changed, incrementNonce exploit likely
        if nonce_before is not None:
            current_nonce = await self.snapshot_nonce()
            if current_nonce is not None and current_nonce != nonce_before:
                self._nonce_mismatch_count += 1
                event = GhostFillEvent(
                    order_id=order_id,
                    token_id=token_id,
                    expected_delta=expected_delta,
                    balance_before=balance_before,
                    balance_after=balance_before,  # No settlement expected
                )
                self._ghost_fills.append(event)
                if len(self._ghost_fills) > 200:
                    self._ghost_fills = self._ghost_fills[-200:]
                log.critical(
                    "ghost_fill_nonce_mismatch",
                    order_id=order_id,
                    nonce_before=nonce_before,
                    nonce_after=current_nonce,
                    token_id=token_id[:16],
                )
                return False

        loop = asyncio.get_running_loop()
        start = time.time()
        last_balance = balance_before
        polls = 0

        while time.time() - start < self._verify_timeout:
            await asyncio.sleep(self._verify_interval)
            polls += 1

            try:
                current = await asyncio.wait_for(
                    loop.run_in_executor(None, self._read_balance_sync, token_id),
                    timeout=20.0,
                )
            except asyncio.TimeoutError:
                log.warning("ghost_guard_verify_timeout", order_id=order_id, polls=polls)
                return True  # RPC timed out — trust the CLOB result
            if current is None:
                # RPC failed mid-verification — trust the CLOB result
                log.warning("ghost_guard_rpc_failed_mid_verify", order_id=order_id, polls=polls)
                return True

            last_balance = current
            delta = current - balance_before
            tolerance = max(_MIN_DELTA, expected_delta * _MIN_DELTA_PCT)
            if delta >= expected_delta - tolerance:
                self._verified_count += 1
                log.info(
                    "ghost_guard_verified",
                    order_id=order_id,
                    delta=round(delta, 4),
                    expected=round(expected_delta, 4),
                    tolerance=round(tolerance, 4),
                    polls=polls,
                    elapsed_sec=round(time.time() - start, 1),
                )
                return True

        # Timeout — ghost fill detected
        actual_delta = last_balance - balance_before
        event = GhostFillEvent(
            order_id=order_id,
            token_id=token_id,
            expected_delta=expected_delta,
            balance_before=balance_before,
            balance_after=last_balance,
        )
        self._ghost_fills.append(event)
        if len(self._ghost_fills) > 200:
            self._ghost_fills = self._ghost_fills[-200:]

        log.critical(
            "ghost_fill_detected",
            order_id=order_id,
            token_id=token_id[:16],
            expected_delta=round(expected_delta, 4),
            actual_delta=round(actual_delta, 4),
            balance_before=round(balance_before, 4),
            balance_after=round(last_balance, 4),
            polls=polls,
            timeout=self._verify_timeout,
        )
        return False
