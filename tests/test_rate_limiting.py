#!/usr/bin/env python3
"""
Comprehensive tests for the rate limiting system.

Tests all components including token bucket, circuit breaker, queue management,
cost tracking, and API service integration.
"""

import asyncio
import pytest
import time
import threading
from decimal import Decimal
from unittest.mock import Mock, patch, MagicMock

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from src.core.rate_limiter import (
    TokenBucket,
    CircuitBreaker,
    RateLimiter,
    RateLimitConfig,
    Priority,
    CircuitState,
    rate_limited,
    rate_limit_manager,
)
from src.core.api_wrappers import (
    DeepgramWrapper,
    OpenAIWrapper,
    AnthropicWrapper,
    GeminiWrapper,
    graceful_api_call,
    check_service_health,
)
from src.core.rate_limit_config import (
    EnvironmentConfig,
    Environment,
    get_service_config,
)
from src.core.rate_limit_monitor import RateLimitMonitor, RateLimitAlert


class TestTokenBucket:
    """Test token bucket rate limiting"""

    def test_token_bucket_creation(self):
        """Test token bucket initialization"""
        bucket = TokenBucket(max_tokens=5, refill_rate=1.0, burst_capacity=2)

        assert bucket.max_tokens == 5
        assert bucket.refill_rate == 1.0
        assert bucket.burst_capacity == 2
        assert bucket.available_tokens == 5.0  # Starts full

    def test_token_consumption(self):
        """Test token consumption"""
        bucket = TokenBucket(max_tokens=5, refill_rate=1.0)

        # Should be able to consume tokens
        assert bucket.consume(3) == True
        assert bucket.available_tokens == 2.0

        # Should fail when not enough tokens
        assert bucket.consume(3) == False
        assert bucket.available_tokens == 2.0  # Unchanged

        # Should succeed with remaining tokens
        assert bucket.consume(2) == True
        assert bucket.available_tokens == 0.0

    def test_token_refill(self):
        """Test token refill over time"""
        bucket = TokenBucket(max_tokens=5, refill_rate=2.0)  # 2 tokens per second

        # Consume all tokens
        bucket.consume(5)
        assert bucket.available_tokens == 0.0

        # Wait for refill
        time.sleep(1.1)  # Should add ~2 tokens

        # Check refill happened
        assert bucket.available_tokens >= 1.8  # Account for timing variance
        assert bucket.consume(2) == True

    def test_burst_capacity(self):
        """Test burst capacity functionality"""
        bucket = TokenBucket(max_tokens=5, refill_rate=1.0, burst_capacity=3)

        # Should start with max + burst capacity
        assert bucket.available_tokens == 5.0

        # Let it accumulate tokens
        time.sleep(0.1)
        bucket._refill()  # Force refill

        # Should be able to consume up to max + burst
        assert bucket.consume(8) == True  # 5 + 3
        assert bucket.consume(1) == False  # Should fail

    def test_time_until_available(self):
        """Test time calculation for token availability"""
        bucket = TokenBucket(max_tokens=5, refill_rate=1.0)

        # Consume all tokens
        bucket.consume(5)

        # Should need 3 seconds for 3 tokens
        time_needed = bucket.time_until_available(3)
        assert time_needed == 3.0


class TestCircuitBreaker:
    """Test circuit breaker functionality"""

    def test_circuit_breaker_creation(self):
        """Test circuit breaker initialization"""
        config = RateLimitConfig(
            failure_threshold=3, recovery_timeout=30, success_threshold=2
        )
        breaker = CircuitBreaker(config)

        assert breaker.state == CircuitState.CLOSED
        assert breaker.failure_count == 0
        assert breaker.success_count == 0

    def test_circuit_breaker_failure_tracking(self):
        """Test failure tracking and state changes"""
        config = RateLimitConfig(failure_threshold=3, recovery_timeout=30)
        breaker = CircuitBreaker(config)

        # Should allow execution initially
        assert breaker.can_execute() == True

        # Record failures
        for i in range(2):
            breaker.record_failure()
            assert breaker.state == CircuitState.CLOSED  # Still closed
            assert breaker.can_execute() == True

        # Third failure should open circuit
        breaker.record_failure()
        assert breaker.state == CircuitState.OPEN
        assert breaker.can_execute() == False

    def test_circuit_breaker_recovery(self):
        """Test circuit breaker recovery process"""
        config = RateLimitConfig(
            failure_threshold=2, recovery_timeout=1, success_threshold=2
        )
        breaker = CircuitBreaker(config)

        # Open the circuit
        breaker.record_failure()
        breaker.record_failure()
        assert breaker.state == CircuitState.OPEN

        # Wait for recovery timeout
        time.sleep(1.1)

        # Should move to half-open
        assert breaker.can_execute() == True  # This triggers state change
        assert breaker.state == CircuitState.HALF_OPEN

        # Record successes to close circuit
        breaker.record_success()
        assert breaker.state == CircuitState.HALF_OPEN  # Still half-open

        breaker.record_success()
        assert breaker.state == CircuitState.CLOSED  # Now closed

    def test_circuit_breaker_half_open_failure(self):
        """Test circuit breaker failure in half-open state"""
        config = RateLimitConfig(failure_threshold=1, recovery_timeout=1)
        breaker = CircuitBreaker(config)

        # Open the circuit
        breaker.record_failure()
        assert breaker.state == CircuitState.OPEN

        # Wait for recovery
        time.sleep(1.1)
        breaker.can_execute()  # Trigger half-open
        assert breaker.state == CircuitState.HALF_OPEN

        # Failure should reopen circuit
        breaker.record_failure()
        assert breaker.state == CircuitState.OPEN


class TestRateLimiter:
    """Test comprehensive rate limiter functionality"""

    @pytest.fixture
    def config(self):
        """Test configuration"""
        return RateLimitConfig(
            max_tokens=3,
            refill_rate=1.0,
            burst_capacity=1,
            max_cost_per_minute=Decimal("1.00"),
            cost_per_request=Decimal("0.10"),
            failure_threshold=2,
            recovery_timeout=1,
            max_queue_size=10,
            request_timeout=5,
        )

    @pytest.fixture
    def limiter(self, config):
        """Test rate limiter instance"""
        limiter = RateLimiter("test_service", config)
        yield limiter
        limiter.shutdown()

    def test_limiter_creation(self, limiter):
        """Test rate limiter creation"""
        assert limiter.service_name == "test_service"
        assert limiter.token_bucket is not None
        assert limiter.circuit_breaker is not None
        assert limiter.request_queue is not None

    def test_successful_execution(self, limiter):
        """Test successful request execution"""

        def test_func(value):
            return value * 2

        result = limiter.execute_sync(test_func, 5)
        assert result == 10

        # Check metrics
        assert limiter.metrics.total_requests == 1
        assert limiter.metrics.successful_requests == 1
        assert limiter.metrics.failed_requests == 0

    def test_failed_execution(self, limiter):
        """Test failed request handling"""

        def failing_func():
            raise ValueError("Test error")

        with pytest.raises(ValueError):
            limiter.execute_sync(failing_func)

        # Check metrics
        assert limiter.metrics.total_requests == 1
        assert limiter.metrics.successful_requests == 0
        assert limiter.metrics.failed_requests == 1

    def test_cost_tracking(self, limiter):
        """Test cost tracking functionality"""

        def test_func():
            return "success"

        # Execute with cost
        result = limiter.execute_sync(test_func, cost_estimate=Decimal("0.05"))

        assert result == "success"

        # Check cost tracking
        cost_rate = limiter._get_current_cost_rate()
        assert cost_rate == Decimal("0.05")

    @pytest.mark.asyncio
    async def test_async_execution(self, limiter):
        """Test async request execution"""

        async def async_func(value):
            await asyncio.sleep(0.01)
            return value * 3

        result = await limiter.execute_async(async_func, 7)
        assert result == 21

    def test_queue_overflow(self, limiter):
        """Test queue overflow handling"""

        def slow_func():
            time.sleep(0.5)
            return "done"

        # Fill the queue beyond capacity
        with pytest.raises(RuntimeError, match="Rate limit queue is full"):
            for i in range(15):  # More than max_queue_size
                limiter.execute_sync(slow_func)

    def test_circuit_breaker_integration(self, limiter):
        """Test circuit breaker integration"""

        def failing_func():
            raise RuntimeError("API error")

        # Cause failures to open circuit
        for i in range(2):
            with pytest.raises(RuntimeError):
                limiter.execute_sync(failing_func)

        # Circuit should be open now
        assert limiter.circuit_breaker.is_open == True

        # New requests should fail fast
        with pytest.raises(RuntimeError, match="Circuit breaker is OPEN"):
            limiter.execute_sync(lambda: "test")

    def test_adaptive_interval(self, limiter):
        """Test adaptive interval adjustment"""
        initial_interval = limiter.adaptive_interval

        # Successful requests should decrease interval
        limiter.execute_sync(lambda: "success")
        assert limiter.adaptive_interval <= initial_interval

        # Failed requests should increase interval
        try:
            limiter.execute_sync(lambda: 1 / 0)
        except:
            pass

        assert limiter.adaptive_interval > initial_interval

    def test_status_reporting(self, limiter):
        """Test status reporting functionality"""
        status = limiter.get_status()

        assert isinstance(status, dict)
        assert "service_name" in status
        assert "available_tokens" in status
        assert "circuit_breaker_state" in status
        assert "queue_size" in status
        assert "success_rate" in status


class TestDecorators:
    """Test rate limiting decorators"""

    def test_rate_limited_decorator_sync(self):
        """Test rate_limited decorator on sync function"""

        @rate_limited("test_service", Priority.NORMAL, Decimal("0.01"))
        def test_function(x, y):
            return x + y

        result = test_function(3, 4)
        assert result == 7

    @pytest.mark.asyncio
    async def test_rate_limited_decorator_async(self):
        """Test rate_limited decorator on async function"""

        @rate_limited("test_service_async", Priority.HIGH, Decimal("0.02"))
        async def async_function(x):
            await asyncio.sleep(0.01)
            return x * 2

        result = await async_function(5)
        assert result == 10


class TestAPIWrappers:
    """Test API service wrappers"""

    def test_deepgram_wrapper_creation(self):
        """Test Deepgram wrapper initialization"""
        wrapper = DeepgramWrapper()
        assert wrapper.service_name == "deepgram"
        assert wrapper.limiter is not None

    def test_openai_wrapper_creation(self):
        """Test OpenAI wrapper initialization"""
        wrapper = OpenAIWrapper()
        assert wrapper.service_name == "openai"
        assert wrapper.limiter is not None

    def test_anthropic_wrapper_creation(self):
        """Test Anthropic wrapper initialization"""
        wrapper = AnthropicWrapper()
        assert wrapper.service_name == "anthropic"
        assert wrapper.limiter is not None

    def test_gemini_wrapper_creation(self):
        """Test Gemini wrapper initialization"""
        wrapper = GeminiWrapper()
        assert wrapper.service_name == "gemini"
        assert wrapper.limiter is not None

    def test_graceful_api_call_success(self):
        """Test graceful API call with success"""

        def successful_service():
            return "success"

        result = graceful_api_call(service_func=successful_service, service_name="test")
        assert result == "success"

    def test_graceful_api_call_with_fallback(self):
        """Test graceful API call with fallback"""

        def failing_service():
            raise RuntimeError("Service unavailable")

        def fallback_service():
            return "fallback"

        result = graceful_api_call(
            service_func=failing_service,
            fallback_func=fallback_service,
            service_name="test",
        )
        assert result == "fallback"

    def test_service_health_check(self):
        """Test service health checking"""
        health_status = check_service_health()

        assert isinstance(health_status, dict)
        assert "deepgram" in health_status
        assert "openai" in health_status
        assert "anthropic" in health_status
        assert "gemini" in health_status

        for service, status in health_status.items():
            assert "healthy" in status
            assert "circuit_state" in status
            assert "success_rate" in status


class TestEnvironmentConfig:
    """Test environment-specific configurations"""

    def test_environment_detection(self):
        """Test environment detection"""
        with patch.dict("os.environ", {"MONTAGE_ENV": "production"}):
            env_config = EnvironmentConfig()
            assert env_config.environment == Environment.PRODUCTION

    def test_development_config(self):
        """Test development environment configuration"""
        env_config = EnvironmentConfig(Environment.DEVELOPMENT)
        deepgram_config = env_config.get_deepgram_config()

        # Development should be more permissive
        assert deepgram_config.max_tokens >= 5
        assert deepgram_config.refill_rate >= 1.0
        assert deepgram_config.max_cost_per_minute >= Decimal("2.00")

    def test_production_config(self):
        """Test production environment configuration"""
        env_config = EnvironmentConfig(Environment.PRODUCTION)
        deepgram_config = env_config.get_deepgram_config()

        # Production should be more restrictive
        assert deepgram_config.max_tokens <= 3
        assert deepgram_config.refill_rate <= 1.0
        assert deepgram_config.failure_threshold <= 3

    def test_service_config_retrieval(self):
        """Test service configuration retrieval"""
        config = get_service_config("deepgram")
        assert isinstance(config, RateLimitConfig)
        assert config.max_tokens > 0
        assert config.refill_rate > 0


class TestRateLimitMonitor:
    """Test rate limiting monitoring"""

    @pytest.fixture
    def monitor(self):
        """Test monitor instance"""
        monitor = RateLimitMonitor()
        yield monitor
        monitor.stop_monitoring()

    def test_monitor_creation(self, monitor):
        """Test monitor initialization"""
        assert monitor.alert_callback is None
        assert monitor.is_monitoring == False
        assert len(monitor.alerts_history) == 0

    def test_alert_creation(self, monitor):
        """Test alert creation and handling"""
        alerts_received = []

        def test_callback(alert):
            alerts_received.append(alert)

        monitor.alert_callback = test_callback

        # Create a test alert
        monitor._maybe_alert(
            "test_alert",
            "test_service",
            "Test alert message",
            "high",
            {"test": "data"},
            time.time(),
        )

        assert len(alerts_received) == 1
        assert alerts_received[0].service == "test_service"
        assert alerts_received[0].severity == "high"

    def test_health_summary(self, monitor):
        """Test health summary generation"""
        summary = monitor.get_system_health_summary()

        assert isinstance(summary, dict)
        assert "overall_status" in summary
        assert "health_score" in summary
        assert "total_services" in summary
        assert "environment" in summary

    def test_metrics_export(self, monitor):
        """Test metrics export functionality"""
        exported_data = monitor.export_metrics("json")

        # Should be valid JSON
        import json

        data = json.loads(exported_data)

        assert "timestamp" in data
        assert "environment" in data
        assert "system_health" in data
        assert "service_metrics" in data


class TestIntegration:
    """Integration tests for the complete system"""

    def test_end_to_end_rate_limiting(self):
        """Test complete rate limiting flow"""
        # Create a test function
        call_count = 0

        @rate_limited("integration_test", Priority.NORMAL, Decimal("0.01"))
        def test_api_call():
            nonlocal call_count
            call_count += 1
            return f"call_{call_count}"

        # Make several calls
        results = []
        for i in range(3):
            result = test_api_call()
            results.append(result)

        assert len(results) == 3
        assert call_count == 3
        assert results == ["call_1", "call_2", "call_3"]

    def test_concurrent_rate_limiting(self):
        """Test rate limiting under concurrent load"""
        results = []
        errors = []

        @rate_limited("concurrent_test", Priority.NORMAL, Decimal("0.01"))
        def concurrent_api_call(thread_id):
            time.sleep(0.1)  # Simulate API delay
            return f"result_{thread_id}"

        def worker(thread_id):
            try:
                result = concurrent_api_call(thread_id)
                results.append(result)
            except Exception as e:
                errors.append(e)

        # Create multiple threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=worker, args=(i,))
            threads.append(thread)
            thread.start()

        # Wait for completion
        for thread in threads:
            thread.join()

        # Should have some results and potentially some rate limiting
        assert len(results) + len(errors) == 5
        assert len(results) > 0  # At least some should succeed

    def test_budget_enforcement(self):
        """Test budget enforcement"""
        # This test would need to be run with a very low budget
        # to actually trigger budget exceeded errors
        pass  # Placeholder for budget testing

    def test_circuit_breaker_integration(self):
        """Test circuit breaker in full system"""
        failure_count = 0

        @rate_limited("circuit_test", Priority.NORMAL, Decimal("0.01"))
        def unreliable_api_call():
            nonlocal failure_count
            failure_count += 1
            if failure_count <= 3:
                raise RuntimeError("API temporarily unavailable")
            return "success"

        # First few calls should fail and open circuit
        for i in range(3):
            with pytest.raises(RuntimeError):
                unreliable_api_call()

        # Circuit should be open, subsequent calls should fail fast
        # (This might need adjustment based on actual circuit breaker config)
        pass  # This test needs refinement based on actual behavior


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
