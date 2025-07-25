"""
P1-02: Celery Resource Watchdog and Memory Management

This module provides comprehensive resource monitoring and limits for Celery workers:
- Memory usage monitoring and automatic cleanup
- CPU usage tracking and throttling
- Task timeout and resource cleanup
- System health monitoring and alerts
- Automatic worker restart on resource exhaustion
"""

import gc
import os
import signal
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional

import psutil
from celery import current_task
from celery.signals import (
    task_failure,
    task_postrun,
    task_prerun,
    task_success,
    worker_ready,
    worker_shutting_down,
)

from ..utils.logging_config import get_logger

logger = get_logger(__name__)

class ResourceAlertLevel(str, Enum):
    """Resource alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"

@dataclass
class ResourceLimits:
    """P1-02: Configurable resource limits for Celery workers"""

    # Memory limits (MB)
    MAX_MEMORY_MB: int = int(os.getenv("CELERY_MAX_MEMORY_MB", "2048"))  # 2GB default
    MEMORY_WARNING_MB: int = int(os.getenv("CELERY_MEMORY_WARNING_MB", "1536"))  # 1.5GB warning

    # CPU limits (percentage)
    MAX_CPU_PERCENT: float = float(os.getenv("CELERY_MAX_CPU_PERCENT", "85.0"))
    CPU_WARNING_PERCENT: float = float(os.getenv("CELERY_CPU_WARNING_PERCENT", "70.0"))

    # Task limits
    MAX_TASK_RUNTIME_SEC: int = int(os.getenv("CELERY_MAX_TASK_TIME_SEC", "1800"))  # 30 minutes
    SOFT_TASK_LIMIT_SEC: int = int(os.getenv("CELERY_SOFT_TASK_LIMIT_SEC", "1500"))  # 25 minutes

    # File system limits
    MAX_TEMP_FILES: int = int(os.getenv("MAX_TEMP_FILES", "100"))
    MAX_TEMP_SIZE_MB: int = int(os.getenv("MAX_TEMP_SIZE_MB", "5000"))  # 5GB

    # Process limits
    MAX_OPEN_FILES: int = int(os.getenv("MAX_OPEN_FILES", "1000"))
    MAX_PROCESSES: int = int(os.getenv("MAX_PROCESSES", "50"))

    # Monitoring intervals
    MONITORING_INTERVAL_SEC: int = int(os.getenv("RESOURCE_MONITOR_INTERVAL", "30"))
    CLEANUP_INTERVAL_SEC: int = int(os.getenv("RESOURCE_CLEANUP_INTERVAL", "300"))  # 5 minutes

class ResourceWatchdog:
    """
    P1-02: Real-time resource monitoring and enforcement for Celery workers
    
    Features:
    - Memory usage monitoring with automatic cleanup
    - CPU throttling and process limits
    - File handle and temporary file tracking
    - Automatic worker shutdown on critical resource exhaustion
    - Resource usage metrics and alerts
    - Emergency cleanup procedures
    """

    def __init__(self):
        self.limits = ResourceLimits()
        self.process = psutil.Process()
        self.start_time = datetime.utcnow()
        self.monitoring_active = False
        self.monitor_thread: Optional[threading.Thread] = None
        self.current_task_start: Optional[datetime] = None
        self.resource_alerts: List[Dict] = []
        self.temp_file_registry: List[Path] = []

        # Statistics tracking
        self.stats = {
            "tasks_processed": 0,
            "memory_cleanups": 0,
            "cpu_throttles": 0,
            "emergency_shutdowns": 0,
            "max_memory_used_mb": 0,
            "max_cpu_used_percent": 0.0
        }

        logger.info(f"Resource watchdog initialized with {self.limits.MAX_MEMORY_MB}MB memory limit")

    def start_monitoring(self):
        """Start background resource monitoring thread"""
        if self.monitoring_active:
            return

        self.monitoring_active = True
        self.monitor_thread = threading.Thread(
            target=self._monitor_resources,
            daemon=True,
            name="ResourceMonitor"
        )
        self.monitor_thread.start()
        logger.info("Resource monitoring started")

    def stop_monitoring(self):
        """Stop background resource monitoring"""
        self.monitoring_active = False
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=5)
        logger.info("Resource monitoring stopped")

    def _monitor_resources(self):
        """Background thread for continuous resource monitoring"""
        last_cleanup = datetime.utcnow()

        while self.monitoring_active:
            try:
                # Get current resource usage
                memory_info = self.process.memory_info()
                memory_mb = memory_info.rss / 1024 / 1024
                cpu_percent = self.process.cpu_percent(interval=1.0)

                # Update statistics
                self.stats["max_memory_used_mb"] = max(self.stats["max_memory_used_mb"], memory_mb)
                self.stats["max_cpu_used_percent"] = max(self.stats["max_cpu_used_percent"], cpu_percent)

                # Check memory limits
                if memory_mb > self.limits.MAX_MEMORY_MB:
                    self._handle_memory_critical(memory_mb)
                elif memory_mb > self.limits.MEMORY_WARNING_MB:
                    self._handle_memory_warning(memory_mb)

                # Check CPU limits
                if cpu_percent > self.limits.MAX_CPU_PERCENT:
                    self._handle_cpu_critical(cpu_percent)
                elif cpu_percent > self.limits.CPU_WARNING_PERCENT:
                    self._handle_cpu_warning(cpu_percent)

                # Check task timeout
                if self.current_task_start:
                    task_duration = (datetime.utcnow() - self.current_task_start).total_seconds()
                    if task_duration > self.limits.MAX_TASK_RUNTIME_SEC:
                        self._handle_task_timeout(task_duration)

                # Periodic cleanup
                now = datetime.utcnow()
                if (now - last_cleanup).total_seconds() > self.limits.CLEANUP_INTERVAL_SEC:
                    self._periodic_cleanup()
                    last_cleanup = now

                # Check temporary files
                self._check_temp_files()

                # Check system resources
                self._check_system_resources()

                time.sleep(self.limits.MONITORING_INTERVAL_SEC)

            except Exception as e:
                logger.error(f"Error in resource monitoring: {e}")
                time.sleep(self.limits.MONITORING_INTERVAL_SEC)

    def _handle_memory_critical(self, memory_mb: float):
        """Handle critical memory usage"""
        logger.critical(f"CRITICAL: Memory usage {memory_mb:.1f}MB exceeds limit {self.limits.MAX_MEMORY_MB}MB")

        self._record_alert(ResourceAlertLevel.CRITICAL, "memory", f"{memory_mb:.1f}MB")

        # Emergency cleanup
        self._emergency_cleanup()

        # Check if still over limit after cleanup
        new_memory_mb = self.process.memory_info().rss / 1024 / 1024
        if new_memory_mb > self.limits.MAX_MEMORY_MB:
            logger.emergency(f"Memory still critical after cleanup: {new_memory_mb:.1f}MB")
            self._emergency_shutdown("Critical memory exhaustion")
        else:
            logger.info(f"Memory cleanup successful: {new_memory_mb:.1f}MB")

    def _handle_memory_warning(self, memory_mb: float):
        """Handle memory warning threshold"""
        logger.warning(f"Memory usage {memory_mb:.1f}MB exceeds warning threshold {self.limits.MEMORY_WARNING_MB}MB")
        self._record_alert(ResourceAlertLevel.WARNING, "memory", f"{memory_mb:.1f}MB")

        # Trigger gentle cleanup
        self._gentle_cleanup()

    def _handle_cpu_critical(self, cpu_percent: float):
        """Handle critical CPU usage"""
        logger.critical(f"CRITICAL: CPU usage {cpu_percent:.1f}% exceeds limit {self.limits.MAX_CPU_PERCENT}%")
        self._record_alert(ResourceAlertLevel.CRITICAL, "cpu", f"{cpu_percent:.1f}%")

        # Throttle current task
        self._throttle_cpu()
        self.stats["cpu_throttles"] += 1

    def _handle_cpu_warning(self, cpu_percent: float):
        """Handle CPU warning threshold"""
        logger.warning(f"CPU usage {cpu_percent:.1f}% exceeds warning threshold {self.limits.CPU_WARNING_PERCENT}%")
        self._record_alert(ResourceAlertLevel.WARNING, "cpu", f"{cpu_percent:.1f}%")

    def _handle_task_timeout(self, duration_sec: float):
        """Handle task timeout"""
        logger.critical(f"CRITICAL: Task timeout after {duration_sec:.1f}s (limit: {self.limits.MAX_TASK_RUNTIME_SEC}s)")
        self._record_alert(ResourceAlertLevel.CRITICAL, "task_timeout", f"{duration_sec:.1f}s")

        # Kill current task
        if current_task:
            logger.error(f"Terminating timed-out task: {current_task.request.id}")
            # Signal the task to abort
            os.kill(os.getpid(), signal.SIGTERM)

    def _emergency_cleanup(self):
        """Aggressive cleanup to free memory immediately"""
        logger.warning("Starting emergency memory cleanup")

        try:
            # Force garbage collection multiple times
            for _ in range(3):
                collected = gc.collect()
                logger.debug(f"Garbage collection freed {collected} objects")

            # Clear temporary file registry
            self._cleanup_temp_files(force=True)

            # Clear caches if available
            self._clear_caches()

            self.stats["memory_cleanups"] += 1
            logger.info("Emergency cleanup completed")

        except Exception as e:
            logger.error(f"Error during emergency cleanup: {e}")

    def _gentle_cleanup(self):
        """Gentle cleanup for warning conditions"""
        try:
            # Single garbage collection
            collected = gc.collect()
            logger.debug(f"Gentle cleanup: garbage collection freed {collected} objects")

            # Clean old temporary files
            self._cleanup_temp_files(force=False)

        except Exception as e:
            logger.error(f"Error during gentle cleanup: {e}")

    def _periodic_cleanup(self):
        """Periodic maintenance cleanup"""
        try:
            # Regular garbage collection
            gc.collect()

            # Clean temporary files
            self._cleanup_temp_files()

            # Rotate alerts log
            if len(self.resource_alerts) > 1000:
                self.resource_alerts = self.resource_alerts[-500:]

            logger.debug("Periodic cleanup completed")

        except Exception as e:
            logger.error(f"Error during periodic cleanup: {e}")

    def _throttle_cpu(self):
        """Throttle CPU usage by introducing delays"""
        try:
            # Reduce process priority
            self.process.nice(10)  # Lower priority

            # Short sleep to reduce CPU load
            time.sleep(0.1)

            logger.info("CPU throttling applied")

        except Exception as e:
            logger.error(f"Error applying CPU throttling: {e}")

    def _cleanup_temp_files(self, force: bool = False):
        """Clean up temporary files"""
        try:
            temp_dirs = [Path("/tmp/montage"), Path("./uploads"), Path("./temp")]

            for temp_dir in temp_dirs:
                if not temp_dir.exists():
                    continue

                cutoff_time = datetime.utcnow() - timedelta(hours=1 if force else 24)

                for temp_file in temp_dir.iterdir():
                    if temp_file.is_file():
                        file_time = datetime.fromtimestamp(temp_file.stat().st_mtime)

                        if file_time < cutoff_time:
                            try:
                                temp_file.unlink()
                                logger.debug(f"Cleaned up temporary file: {temp_file}")
                            except Exception as e:
                                logger.error(f"Failed to clean {temp_file}: {e}")

        except Exception as e:
            logger.error(f"Error cleaning temporary files: {e}")

    def _clear_caches(self):
        """Clear application caches if available"""
        try:
            # Clear any global caches here
            # Example: clear video analysis caches, model caches, etc.

            # Clear any imported module caches
            import sys
            for module_name in list(sys.modules.keys()):
                if hasattr(sys.modules[module_name], '_cache'):
                    if hasattr(sys.modules[module_name]._cache, 'clear'):
                        sys.modules[module_name]._cache.clear()

        except Exception as e:
            logger.error(f"Error clearing caches: {e}")

    def _check_temp_files(self):
        """Monitor temporary file usage"""
        try:
            temp_dirs = [Path("/tmp/montage"), Path("./uploads"), Path("./temp")]
            total_files = 0
            total_size_mb = 0

            for temp_dir in temp_dirs:
                if not temp_dir.exists():
                    continue

                for temp_file in temp_dir.iterdir():
                    if temp_file.is_file():
                        total_files += 1
                        total_size_mb += temp_file.stat().st_size / 1024 / 1024

            if total_files > self.limits.MAX_TEMP_FILES:
                logger.warning(f"Too many temporary files: {total_files} > {self.limits.MAX_TEMP_FILES}")
                self._cleanup_temp_files(force=True)

            if total_size_mb > self.limits.MAX_TEMP_SIZE_MB:
                logger.warning(f"Temporary files too large: {total_size_mb:.1f}MB > {self.limits.MAX_TEMP_SIZE_MB}MB")
                self._cleanup_temp_files(force=True)

        except Exception as e:
            logger.error(f"Error checking temporary files: {e}")

    def _check_system_resources(self):
        """Check system-wide resource limits"""
        try:
            # Check file descriptors
            open_files = len(self.process.open_files())
            if open_files > self.limits.MAX_OPEN_FILES:
                logger.warning(f"Too many open files: {open_files} > {self.limits.MAX_OPEN_FILES}")

            # Check child processes
            children = len(self.process.children())
            if children > self.limits.MAX_PROCESSES:
                logger.warning(f"Too many child processes: {children} > {self.limits.MAX_PROCESSES}")

        except Exception as e:
            logger.error(f"Error checking system resources: {e}")

    def _record_alert(self, level: ResourceAlertLevel, resource_type: str, value: str):
        """Record resource alert"""
        alert = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": level.value,
            "resource": resource_type,
            "value": value,
            "task_id": current_task.request.id if current_task else None
        }

        self.resource_alerts.append(alert)

        # Keep only recent alerts
        if len(self.resource_alerts) > 1000:
            self.resource_alerts = self.resource_alerts[-500:]

    def _emergency_shutdown(self, reason: str):
        """Emergency worker shutdown due to critical resource exhaustion"""
        logger.emergency(f"EMERGENCY SHUTDOWN: {reason}")
        self.stats["emergency_shutdowns"] += 1

        try:
            # Clean up as much as possible
            self._emergency_cleanup()

            # Send SIGTERM to self to trigger graceful shutdown
            os.kill(os.getpid(), signal.SIGTERM)

        except Exception as e:
            logger.error(f"Error during emergency shutdown: {e}")
            # Force exit as last resort
            os._exit(1)

    def register_temp_file(self, file_path: Path):
        """Register a temporary file for tracking"""
        self.temp_file_registry.append(file_path)

    def unregister_temp_file(self, file_path: Path):
        """Remove temporary file from tracking"""
        try:
            self.temp_file_registry.remove(file_path)
        except ValueError:
            pass  # File not in registry

    def get_resource_stats(self) -> Dict:
        """Get current resource usage statistics"""
        memory_info = self.process.memory_info()
        memory_mb = memory_info.rss / 1024 / 1024
        cpu_percent = self.process.cpu_percent()

        uptime_sec = (datetime.utcnow() - self.start_time).total_seconds()

        return {
            "current": {
                "memory_mb": round(memory_mb, 1),
                "cpu_percent": round(cpu_percent, 1),
                "open_files": len(self.process.open_files()),
                "child_processes": len(self.process.children())
            },
            "limits": {
                "max_memory_mb": self.limits.MAX_MEMORY_MB,
                "max_cpu_percent": self.limits.MAX_CPU_PERCENT,
                "max_task_time_sec": self.limits.MAX_TASK_RUNTIME_SEC
            },
            "statistics": self.stats,
            "uptime_hours": round(uptime_sec / 3600, 1),
            "recent_alerts": self.resource_alerts[-10:]  # Last 10 alerts
        }

# Global watchdog instance
resource_watchdog = ResourceWatchdog()

# Signal handlers for Celery integration
@worker_ready.connect
def worker_ready_handler(sender=None, **kwargs):
    """Start resource monitoring when worker is ready"""
    logger.info("Celery worker ready, starting resource monitoring")
    resource_watchdog.start_monitoring()

@worker_shutting_down.connect
def worker_shutdown_handler(sender=None, **kwargs):
    """Stop resource monitoring when worker shuts down"""
    logger.info("Celery worker shutting down, stopping resource monitoring")
    resource_watchdog.stop_monitoring()

@task_prerun.connect
def task_prerun_resource_handler(sender=None, task_id=None, task=None, **kwargs):
    """Record task start time for timeout monitoring"""
    resource_watchdog.current_task_start = datetime.utcnow()
    resource_watchdog.stats["tasks_processed"] += 1
    logger.debug(f"Task {task_id} started resource monitoring")

@task_postrun.connect
def task_postrun_resource_handler(sender=None, task_id=None, task=None, **kwargs):
    """Clean up after task completion"""
    resource_watchdog.current_task_start = None

    # Trigger cleanup after task
    resource_watchdog._gentle_cleanup()

    logger.debug(f"Task {task_id} completed, cleanup triggered")

@task_success.connect
def task_success_resource_handler(sender=None, **kwargs):
    """Handle successful task completion"""
    pass  # Could add success-specific cleanup here

@task_failure.connect
def task_failure_resource_handler(sender=None, task_id=None, **kwargs):
    """Handle task failure cleanup"""
    # More aggressive cleanup on failure
    resource_watchdog._emergency_cleanup()
    logger.debug(f"Task {task_id} failed, emergency cleanup triggered")
