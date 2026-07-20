"""Tests for ghost_fill_guard.py — on-chain settlement verification.

Ported from the parent bot's test suite. The guard now takes injected
callables for balance and nonce reads instead of building its own web3
client, so everything here runs offline.
"""

import time
from unittest.mock import patch

import pytest

from polymarket_risk.ghost_fill_guard import (
    GhostFillEvent,
    GhostFillGuard,
    _MIN_DELTA,
    _MIN_DELTA_PCT,
    _RPC_COOLDOWN_SEC,
)


def _make_guard(balance_reader=None, nonce_reader=None) -> GhostFillGuard:
    return GhostFillGuard(
        balance_reader=balance_reader or (lambda token_id: None),
        nonce_reader=nonce_reader,
        verify_timeout=2.0,  # Short timeout for tests
        verify_interval=0.1,
    )


@pytest.fixture
def guard():
    return _make_guard()


class TestGhostFillGuardInit:
    def test_initial_state(self, guard):
        assert guard.ghost_fill_count == 0
        assert guard.verified_count == 0
        assert guard.recent_ghost_fills == []

    def test_rpc_available_initially(self, guard):
        assert guard._rpc_available() is True


class TestRPCAvailability:
    def test_rpc_cooldown(self, guard):
        guard._last_rpc_failure = time.time()
        assert guard._rpc_available() is False

    def test_rpc_available_after_cooldown(self, guard):
        guard._last_rpc_failure = time.time() - _RPC_COOLDOWN_SEC - 1
        assert guard._rpc_available() is True


class TestSnapshotBalance:
    @pytest.mark.asyncio
    async def test_returns_none_when_rpc_in_cooldown(self, guard):
        guard._last_rpc_failure = time.time()
        result = await guard.snapshot_balance("12345")
        assert result is None

    @pytest.mark.asyncio
    async def test_returns_balance(self, guard):
        with patch.object(guard, "_read_balance_sync", return_value=100.5):
            result = await guard.snapshot_balance("12345")
        assert result == 100.5

    @pytest.mark.asyncio
    async def test_returns_none_on_rpc_failure(self, guard):
        with patch.object(guard, "_read_balance_sync", return_value=None):
            result = await guard.snapshot_balance("12345")
        assert result is None

    @pytest.mark.asyncio
    async def test_injected_reader_is_used(self):
        """The injected callable is the actual read path (no patching)."""
        seen: list[str] = []

        def reader(token_id: str) -> float:
            seen.append(token_id)
            return 42.5

        guard = _make_guard(balance_reader=reader)
        result = await guard.snapshot_balance("98765")
        assert result == 42.5
        assert seen == ["98765"]


class TestVerifySettlement:
    @pytest.mark.asyncio
    async def test_skip_when_no_snapshot(self, guard):
        """If balance_before is None, verification is skipped (graceful degradation)."""
        result = await guard.verify_settlement(
            token_id="12345",
            expected_delta=10.0,
            balance_before=None,
            order_id="order-1",
        )
        assert result is True

    @pytest.mark.asyncio
    async def test_skip_when_rpc_in_cooldown(self, guard):
        guard._last_rpc_failure = time.time()
        result = await guard.verify_settlement(
            token_id="12345",
            expected_delta=10.0,
            balance_before=100.0,
            order_id="order-1",
        )
        assert result is True

    @pytest.mark.asyncio
    async def test_verified_when_balance_increases(self, guard):
        """Balance increased by expected amount → verified."""
        call_count = 0

        def mock_balance(token_id):
            nonlocal call_count
            call_count += 1
            # First poll: balance increased
            return 110.0

        with patch.object(guard, "_read_balance_sync", side_effect=mock_balance):
            result = await guard.verify_settlement(
                token_id="12345",
                expected_delta=10.0,
                balance_before=100.0,
                order_id="order-1",
            )
        assert result is True
        assert guard.verified_count == 1

    @pytest.mark.asyncio
    async def test_ghost_fill_when_balance_unchanged(self, guard):
        """Balance doesn't increase within timeout → ghost fill."""
        with patch.object(guard, "_read_balance_sync", return_value=100.0):
            result = await guard.verify_settlement(
                token_id="12345",
                expected_delta=10.0,
                balance_before=100.0,
                order_id="order-ghost",
            )
        assert result is False
        assert guard.ghost_fill_count == 1
        assert guard.recent_ghost_fills[0].order_id == "order-ghost"
        assert guard.recent_ghost_fills[0].expected_delta == 10.0

    @pytest.mark.asyncio
    async def test_graceful_on_rpc_failure_mid_verify(self, guard):
        """If RPC fails mid-verification, trust the CLOB result."""
        with patch.object(guard, "_read_balance_sync", return_value=None):
            result = await guard.verify_settlement(
                token_id="12345",
                expected_delta=10.0,
                balance_before=100.0,
                order_id="order-1",
            )
        assert result is True

    @pytest.mark.asyncio
    async def test_partial_fill_tolerance(self, guard):
        """Balance increased by slightly less than expected (within MIN_DELTA tolerance)."""
        with patch.object(guard, "_read_balance_sync", return_value=109.9995):
            result = await guard.verify_settlement(
                token_id="12345",
                expected_delta=10.0,
                balance_before=100.0,
                order_id="order-1",
            )
        assert result is True

    @pytest.mark.asyncio
    async def test_accepts_realistic_partial_fill_2_5pct(self, guard):
        """Live scenario: expected 15.45, actual 15.0653 (2.5% shortfall).

        Previously falsely rejected as ghost fill with _MIN_DELTA=0.001 absolute
        tolerance, causing retries and over-allocation. Percentage tolerance
        (5%) must accept this.
        """
        with patch.object(guard, "_read_balance_sync", return_value=115.0653):
            result = await guard.verify_settlement(
                token_id="12345",
                expected_delta=15.45,
                balance_before=100.0,
                order_id="order-psg",
            )
        assert result is True
        assert guard.verified_count == 1

    @pytest.mark.asyncio
    async def test_accepts_near_max_tolerance_shortfall(self, guard):
        """4.9% shortfall — just under the 5% threshold — must accept."""
        with patch.object(guard, "_read_balance_sync", return_value=100.0 + 9.51):
            result = await guard.verify_settlement(
                token_id="12345",
                expected_delta=10.0,
                balance_before=100.0,
                order_id="order-boundary",
            )
        assert result is True

    @pytest.mark.asyncio
    async def test_rejects_large_shortfall(self, guard):
        """10% shortfall is clearly abnormal — still treated as ghost fill."""
        with patch.object(guard, "_read_balance_sync", return_value=109.0):
            result = await guard.verify_settlement(
                token_id="12345",
                expected_delta=10.0,
                balance_before=100.0,
                order_id="order-ghost-large",
            )
        assert result is False
        assert guard.ghost_fill_count == 1

    @pytest.mark.asyncio
    async def test_small_order_respects_absolute_floor(self, guard):
        """For tiny orders, absolute _MIN_DELTA floor dominates over percentage.

        expected=0.01, 5% is 0.0005, but _MIN_DELTA=0.001 is the floor.
        Shortfall of 0.0008 (8%) is within 0.001 floor → accept.
        """
        with patch.object(guard, "_read_balance_sync", return_value=100.0 + 0.0092):
            result = await guard.verify_settlement(
                token_id="12345",
                expected_delta=0.01,
                balance_before=100.0,
                order_id="order-dust",
            )
        assert result is True

    @pytest.mark.asyncio
    async def test_delayed_settlement(self, guard):
        """Balance increases on second poll → still verified."""
        call_count = 0

        def mock_balance(token_id):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return 100.0  # Not yet
            return 110.0  # Settled

        with patch.object(guard, "_read_balance_sync", side_effect=mock_balance):
            result = await guard.verify_settlement(
                token_id="12345",
                expected_delta=10.0,
                balance_before=100.0,
                order_id="order-1",
            )
        assert result is True
        assert call_count >= 2


class TestGhostFillEventTracking:
    @pytest.mark.asyncio
    async def test_event_recorded(self, guard):
        with patch.object(guard, "_read_balance_sync", return_value=100.0):
            await guard.verify_settlement(
                token_id="token-abc",
                expected_delta=5.0,
                balance_before=100.0,
                order_id="order-x",
            )
        events = guard.recent_ghost_fills
        assert len(events) == 1
        assert events[0].token_id == "token-abc"
        assert events[0].expected_delta == 5.0
        assert events[0].balance_before == 100.0
        assert events[0].balance_after == 100.0

    @pytest.mark.asyncio
    async def test_event_list_bounded(self, guard):
        """Ghost fill list should be bounded at 200."""
        guard._ghost_fills = [
            GhostFillEvent(
                order_id=f"order-{i}",
                token_id="t",
                expected_delta=1.0,
                balance_before=0.0,
                balance_after=0.0,
            )
            for i in range(250)
        ]
        with patch.object(guard, "_read_balance_sync", return_value=0.0):
            await guard.verify_settlement(
                token_id="t",
                expected_delta=1.0,
                balance_before=0.0,
                order_id="overflow",
            )
        assert len(guard._ghost_fills) <= 201  # 200 + 1 new, then trimmed


class TestNonceMonitoring:
    @pytest.mark.asyncio
    async def test_nonce_snapshot(self, guard):
        with patch.object(guard, "_read_nonce_sync", return_value=42):
            result = await guard.snapshot_nonce()
        assert result == 42

    @pytest.mark.asyncio
    async def test_nonce_snapshot_returns_none_on_failure(self, guard):
        with patch.object(guard, "_read_nonce_sync", return_value=None):
            result = await guard.snapshot_nonce()
        assert result is None

    @pytest.mark.asyncio
    async def test_nonce_snapshot_none_without_reader(self, guard):
        """No injected nonce_reader → snapshot returns None (check skipped)."""
        result = await guard.snapshot_nonce()
        assert result is None

    @pytest.mark.asyncio
    async def test_nonce_mismatch_detects_ghost_fill(self, guard):
        """If nonce changed between order and verification, it's an exploit."""
        # Balance stays same (settlement will fail), nonce changed
        with patch.object(guard, "_read_nonce_sync", return_value=43), \
             patch.object(guard, "_read_balance_sync", return_value=100.0):
            result = await guard.verify_settlement(
                token_id="12345",
                expected_delta=10.0,
                balance_before=100.0,
                order_id="order-nonce",
                nonce_before=42,
            )
        assert result is False
        assert guard._nonce_mismatch_count == 1
        assert guard.ghost_fill_count == 1

    @pytest.mark.asyncio
    async def test_nonce_unchanged_proceeds_normally(self, guard):
        """Same nonce → proceed to normal balance verification."""
        with patch.object(guard, "_read_nonce_sync", return_value=42), \
             patch.object(guard, "_read_balance_sync", return_value=110.0):
            result = await guard.verify_settlement(
                token_id="12345",
                expected_delta=10.0,
                balance_before=100.0,
                order_id="order-ok",
                nonce_before=42,
            )
        assert result is True
        assert guard._nonce_mismatch_count == 0

    @pytest.mark.asyncio
    async def test_nonce_none_skips_check(self, guard):
        """If nonce_before is None, skip nonce check entirely."""
        with patch.object(guard, "_read_balance_sync", return_value=110.0):
            result = await guard.verify_settlement(
                token_id="12345",
                expected_delta=10.0,
                balance_before=100.0,
                order_id="order-no-nonce",
                nonce_before=None,
            )
        assert result is True

    def test_read_nonce_rpc_cooldown(self):
        guard = _make_guard(nonce_reader=lambda: 7)
        guard._last_rpc_failure = time.time()
        result = guard._read_nonce_sync()
        assert result is None


class TestReadBalanceSync:
    def test_rpc_rate_limit_sets_cooldown(self):
        """A rate-limit error from the injected reader should trigger cooldown."""

        def reader(token_id):
            raise Exception("rate limit exceeded")

        guard = _make_guard(balance_reader=reader)
        result = guard._read_balance_sync("12345")
        assert result is None
        assert guard._last_rpc_failure > 0

    def test_generic_reader_error_returns_none_without_cooldown(self):
        """Non-rate-limit reader failures degrade gracefully, no cooldown."""

        def reader(token_id):
            raise Exception("connection reset by peer")

        guard = _make_guard(balance_reader=reader)
        result = guard._read_balance_sync("12345")
        assert result is None
        assert guard._last_rpc_failure == 0.0

    def test_rpc_not_available_returns_none(self, guard):
        guard._last_rpc_failure = time.time()
        result = guard._read_balance_sync("12345")
        assert result is None
