#!/usr/bin/env python3
"""
Rate limiting monitoring and management utilities.

Provides real-time monitoring, alerting, and management capabilities
for the comprehensive rate limiting system.
"""

import json
import logging
import threading
import time
from collections import defaultdict, deque
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional

from .api_wrappers import get_all_service_status
from .rate_limit_config import env_config
from .rate_limiter import rate_limit_manager

logger = logging.getLogger(__name__)


@dataclass
class RateLimitAlert:
    """Rate limiting alert data"""

    service: str
    alert_type: str
    message: str
    severity: str  # "low", "medium", "high", "critical"
    timestamp: datetime
    details: Dict[str, Any]


@dataclass
class ServiceMetrics:
    """Service-level metrics for monitoring"""

    service_name: str
    requests_per_minute: float
    success_rate: float
    average_wait_time: float
    queue_size: int
    circuit_state: str
    cost_per_minute: float
    available_tokens: float
    consecutive_failures: int
    last_success_time: Optional[datetime]
    last_failure_time: Optional[datetime]


class RateLimitMonitor:
    """Monitor and manage rate limiting across all services"""

    def __init__(
        self, alert_callback: Optional[Callable[[RateLimitAlert], None]] = None
    ):
        self.alert_callback = alert_callback
        self.is_monitoring = False
        self.monitoring_thread = None
        self.alerts_history = deque(maxlen=1000)  # Keep last 1000 alerts
        self.metrics_history = defaultdict(
            lambda: deque(maxlen=300)
        )  # 5 minutes at 1s intervals

        # Thresholds for alerting
        self.alert_thresholds = {
            "circuit_breaker_open": {"severity": "critical", "cooldown": 300},
            "high_failure_rate": {"threshold": 0.5, "severity": "high", "cooldown": 60},
            "queue_backup": {"threshold": 0.8, "severity": "medium", "cooldown": 30},
            "cost_spike": {"threshold": 1.5, "severity": "high", "cooldown": 120},
            "low_tokens": {"threshold": 2, "severity": "low", "cooldown": 60},
        }

        self.last_alert_times = defaultdict(float)  # Track cooldowns

        logger.info("Rate limit monitor initialized")

    def start_monitoring(self, interval: float = 1.0):
        """Start background monitoring"""
        if self.is_monitoring:
            logger.warning("Monitor already running")
            return

        self.is_monitoring = True
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop, args=(interval,), daemon=True
        )
        self.monitoring_thread.start()
        logger.info(f"Rate limit monitoring started with {interval}s interval")

    def stop_monitoring(self):
        """Stop background monitoring"""
        self.is_monitoring = False
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=5)
        logger.info("Rate limit monitoring stopped")

    def _monitoring_loop(self, interval: float):
        """Main monitoring loop"""
        while self.is_monitoring:
            try:
                # Collect metrics from all services
                service_statuses = get_all_service_status()

                for service_name, status in service_statuses.items():
                    # Convert to ServiceMetrics
                    metrics = self._convert_to_metrics(service_name, status)

                    # Store in history
                    self.metrics_history[service_name].append(metrics)

                    # Check for alerts
                    self._check_alerts(metrics)

                # Sleep for interval
                time.sleep(interval)

            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(interval)

    def _convert_to_metrics(
        self, service_name: str, status: Dict[str, Any]
    ) -> ServiceMetrics:
        """Convert rate limiter status to ServiceMetrics"""

        # Calculate requests per minute from recent history
        recent_metrics = list(self.metrics_history[service_name])[
            -60:
        ]  # Last 60 seconds
        if len(recent_metrics) >= 2:
            first_metric = recent_metrics[0]
            # This is a simplified calculation - in reality you'd track actual request counts
            requests_per_minute = len(recent_metrics)  # Approximate
        else:
            requests_per_minute = 0.0

        return ServiceMetrics(
            service_name=service_name,
            requests_per_minute=requests_per_minute,
            success_rate=status.get("success_rate", 0.0),
            average_wait_time=0.0,  # Would need to track this in rate limiter
            queue_size=status.get("queue_size", 0),
            circuit_state=status.get("circuit_breaker_state", "UNKNOWN"),
            cost_per_minute=status.get("current_cost_rate", 0.0),
            available_tokens=status.get("available_tokens", 0.0),
            consecutive_failures=status.get("consecutive_failures", 0),
            last_success_time=None,  # Would need timestamps from rate limiter
            last_failure_time=None,
        )

    def _check_alerts(self, metrics: ServiceMetrics):
        """Check metrics against alert thresholds"""
        current_time = time.time()

        # Circuit breaker open
        if metrics.circuit_state == "OPEN":
            self._maybe_alert(
                "circuit_breaker_open",
                metrics.service_name,
                f"Circuit breaker is OPEN for {metrics.service_name}",
                "critical",
                {"consecutive_failures": metrics.consecutive_failures},
                current_time,
            )

        # High failure rate
        if (
            metrics.success_rate
            < self.alert_thresholds["high_failure_rate"]["threshold"]
        ):
            self._maybe_alert(
                "high_failure_rate",
                metrics.service_name,
                f"High failure rate for {metrics.service_name}: {metrics.success_rate:.1%}",
                "high",
                {"success_rate": metrics.success_rate},
                current_time,
            )

        # Queue backup
        limiter = rate_limit_manager.get_limiter(metrics.service_name)
        queue_capacity = limiter.config.max_queue_size
        if queue_capacity > 0:
            queue_utilization = metrics.queue_size / queue_capacity
            if queue_utilization > self.alert_thresholds["queue_backup"]["threshold"]:
                self._maybe_alert(
                    "queue_backup",
                    metrics.service_name,
                    f"Queue backup for {metrics.service_name}: {queue_utilization:.1%} full",
                    "medium",
                    {"queue_size": metrics.queue_size, "capacity": queue_capacity},
                    current_time,
                )

        # Cost spike
        config = env_config.get_service_config(metrics.service_name)
        cost_threshold = (
            float(config.max_cost_per_minute)
            * self.alert_thresholds["cost_spike"]["threshold"]
        )
        if metrics.cost_per_minute > cost_threshold:
            self._maybe_alert(
                "cost_spike",
                metrics.service_name,
                f"Cost spike for {metrics.service_name}: ${metrics.cost_per_minute:.3f}/min",
                "high",
                {"current_cost": metrics.cost_per_minute, "threshold": cost_threshold},
                current_time,
            )

        # Low tokens
        if metrics.available_tokens < self.alert_thresholds["low_tokens"]["threshold"]:
            self._maybe_alert(
                "low_tokens",
                metrics.service_name,
                f"Low tokens for {metrics.service_name}: {metrics.available_tokens:.1f}",
                "low",
                {"available_tokens": metrics.available_tokens},
                current_time,
            )

    def _maybe_alert(
        self,
        alert_type: str,
        service: str,
        message: str,
        severity: str,
        details: Dict[str, Any],
        current_time: float,
    ):
        """Send alert if cooldown period has passed"""

        alert_key = f"{service}:{alert_type}"
        cooldown = self.alert_thresholds.get(alert_type, {}).get("cooldown", 60)

        if current_time - self.last_alert_times[alert_key] >= cooldown:
            alert = RateLimitAlert(
                service=service,
                alert_type=alert_type,
                message=message,
                severity=severity,
                timestamp=datetime.now(),
                details=details,
            )

            self.alerts_history.append(alert)
            self.last_alert_times[alert_key] = current_time

            logger.warning(f"Rate limit alert: {alert.message}")

            if self.alert_callback:
                try:
                    self.alert_callback(alert)
                except Exception as e:
                    logger.error(f"Alert callback failed: {e}")

    def get_current_metrics(self) -> Dict[str, ServiceMetrics]:
        """Get current metrics for all services"""
        current_metrics = {}
        service_statuses = get_all_service_status()

        for service_name, status in service_statuses.items():
            current_metrics[service_name] = self._convert_to_metrics(
                service_name, status
            )

        return current_metrics

    def get_historical_metrics(
        self, service: str, minutes: int = 5
    ) -> List[ServiceMetrics]:
        """Get historical metrics for a service"""

        if service not in self.metrics_history:
            return []

        # Get last N minutes of data
        target_count = minutes * 60  # Assuming 1s intervals
        history = list(self.metrics_history[service])
        return history[-target_count:] if len(history) > target_count else history

    def get_recent_alerts(self, minutes: int = 60) -> List[RateLimitAlert]:
        """Get alerts from the last N minutes"""
        cutoff_time = datetime.now() - timedelta(minutes=minutes)
        return [
            alert for alert in self.alerts_history if alert.timestamp >= cutoff_time
        ]

    def get_system_health_summary(self) -> Dict[str, Any]:
        """Get overall system health summary"""
        current_metrics = self.get_current_metrics()
        recent_alerts = self.get_recent_alerts(15)  # Last 15 minutes

        # Calculate overall health score
        total_services = len(current_metrics)
        healthy_services = sum(
            1
            for metrics in current_metrics.values()
            if metrics.circuit_state == "CLOSED"
            and metrics.success_rate > 0.8
            and metrics.consecutive_failures < 3
        )

        health_score = (
            (healthy_services / total_services) * 100 if total_services > 0 else 100
        )

        # Categorize alerts by severity
        critical_alerts = [a for a in recent_alerts if a.severity == "critical"]
        high_alerts = [a for a in recent_alerts if a.severity == "high"]

        # Determine overall status
        if critical_alerts:
            overall_status = "critical"
        elif high_alerts:
            overall_status = "degraded"
        elif health_score < 80:
            overall_status = "warning"
        else:
            overall_status = "healthy"

        return {
            "overall_status": overall_status,
            "health_score": health_score,
            "total_services": total_services,
            "healthy_services": healthy_services,
            "recent_alerts_count": len(recent_alerts),
            "critical_alerts_count": len(critical_alerts),
            "high_alerts_count": len(high_alerts),
            "environment": env_config.environment.value,
            "timestamp": datetime.now().isoformat(),
        }

    def export_metrics(self, format: str = "json") -> str:
        """Export current metrics in specified format"""

        data = {
            "timestamp": datetime.now().isoformat(),
            "environment": env_config.environment.value,
            "system_health": self.get_system_health_summary(),
            "service_metrics": {
                name: asdict(metrics)
                for name, metrics in self.get_current_metrics().items()
            },
            "recent_alerts": [asdict(alert) for alert in self.get_recent_alerts(60)],
        }

        if format.lower() == "json":
            return json.dumps(data, indent=2, default=str)
        else:
            raise ValueError(f"Unsupported format: {format}")


class RateLimitManager:
    """High-level manager for rate limiting operations"""

    def __init__(self):
        self.monitor = RateLimitMonitor()
        self.emergency_mode = False

    def start_monitoring(self, alert_callback: Optional[Callable] = None):
        """Start comprehensive monitoring"""
        if alert_callback:
            self.monitor.alert_callback = alert_callback
        self.monitor.start_monitoring()

    def stop_monitoring(self):
        """Stop monitoring"""
        self.monitor.stop_monitoring()

    def enable_emergency_mode(self):
        """Enable emergency mode with minimal rate limits"""
        if self.emergency_mode:
            return

        logger.warning("Enabling emergency mode - reducing all rate limits")

        from .rate_limit_config import ScenarioConfig

        # Apply emergency configs to all services
        for service_name in ["deepgram", "openai", "anthropic", "gemini"]:
            emergency_config = ScenarioConfig.get_emergency_config(service_name)
            limiter = rate_limit_manager.get_limiter(service_name, emergency_config)

        self.emergency_mode = True

    def disable_emergency_mode(self):
        """Disable emergency mode and restore normal limits"""
        if not self.emergency_mode:
            return

        logger.info("Disabling emergency mode - restoring normal rate limits")

        # Shutdown current limiters to force reload with normal configs
        rate_limit_manager.shutdown_all()

        self.emergency_mode = False

    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get comprehensive data for monitoring dashboard"""
        return {
            "health_summary": self.monitor.get_system_health_summary(),
            "current_metrics": {
                name: asdict(metrics)
                for name, metrics in self.monitor.get_current_metrics().items()
            },
            "recent_alerts": [
                asdict(alert) for alert in self.monitor.get_recent_alerts(30)
            ],
            "emergency_mode": self.emergency_mode,
            "environment": env_config.environment.value,
        }


# Global monitor instance
global_monitor = RateLimitMonitor()


# Convenience functions
def start_rate_limit_monitoring(alert_callback: Optional[Callable] = None):
    """Start global rate limit monitoring"""
    if alert_callback:
        global_monitor.alert_callback = alert_callback
    global_monitor.start_monitoring()


def stop_rate_limit_monitoring():
    """Stop global rate limit monitoring"""
    global_monitor.stop_monitoring()


def get_rate_limit_health() -> Dict[str, Any]:
    """Get current rate limiting health status"""
    return global_monitor.get_system_health_summary()


def export_rate_limit_metrics() -> str:
    """Export current rate limiting metrics as JSON"""
    return global_monitor.export_metrics("json")


# Example alert handler
def log_alert_handler(alert: RateLimitAlert):
    """Simple alert handler that logs to standard logging"""
    log_level = {
        "low": logging.INFO,
        "medium": logging.WARNING,
        "high": logging.ERROR,
        "critical": logging.CRITICAL,
    }.get(alert.severity, logging.WARNING)

    logger.log(
        log_level,
        f"RATE_LIMIT_ALERT [{alert.severity.upper()}] {alert.service}: {alert.message}",
        extra={"alert_details": alert.details},
    )
