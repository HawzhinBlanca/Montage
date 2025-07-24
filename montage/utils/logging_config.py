#!/usr/bin/env python3
"""
Centralized logging configuration for Montage video processing pipeline.
Provides structured, production-ready logging with JSON output and correlation IDs.
"""

import json
import logging
import logging.handlers
import os
import sys
import threading
import time
import uuid
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Optional

# Import secure logging components
try:
    from .secure_logging import (
        SensitiveDataFilter,
        SecureJSONFormatter,
        configure_secure_logging,
    )
    SECURE_LOGGING_AVAILABLE = True
except ImportError:
    SECURE_LOGGING_AVAILABLE = False

# Thread-local storage for correlation IDs
_context_storage = threading.local()


class CorrelationFilter(logging.Filter):
    """Filter to add correlation ID to log records"""

    def filter(self, record):
        # Add correlation ID if available
        correlation_id = getattr(_context_storage, "correlation_id", None)
        job_id = getattr(_context_storage, "job_id", None)

        record.correlation_id = correlation_id or "no-correlation"
        record.job_id = job_id or "no-job"
        return True


class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging"""

    def __init__(self, include_extra_fields: bool = True):
        super().__init__()
        self.include_extra_fields = include_extra_fields

    def format(self, record: logging.LogRecord) -> str:
        # Base log entry
        log_entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
            "correlation_id": getattr(record, "correlation_id", "no-correlation"),
            "job_id": getattr(record, "job_id", "no-job"),
            "process_id": os.getpid(),
            "thread_id": threading.current_thread().ident,
        }

        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = {
                "type": record.exc_info[0].__name__,
                "message": str(record.exc_info[1]),
                "traceback": self.formatException(record.exc_info),
            }

        # Add extra fields if enabled
        if self.include_extra_fields:
            # Add any extra attributes from the log record
            extra_fields = {}
            for key, value in record.__dict__.items():
                if key not in {
                    "name",
                    "msg",
                    "args",
                    "levelname",
                    "levelno",
                    "pathname",
                    "filename",
                    "module",
                    "lineno",
                    "funcName",
                    "created",
                    "msecs",
                    "relativeCreated",
                    "thread",
                    "threadName",
                    "processName",
                    "process",
                    "message",
                    "exc_info",
                    "exc_text",
                    "stack_info",
                    "correlation_id",
                    "job_id",
                }:
                    # Only include serializable values
                    try:
                        json.dumps(value)
                        extra_fields[key] = value
                    except (TypeError, ValueError):
                        extra_fields[key] = str(value)

            if extra_fields:
                log_entry["extra"] = extra_fields

        return json.dumps(log_entry, default=str)


class ColoredConsoleFormatter(logging.Formatter):
    """Colored formatter for console output in development"""

    # Color codes
    COLORS = {
        "DEBUG": "\033[36m",  # Cyan
        "INFO": "\033[32m",  # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",  # Red
        "CRITICAL": "\033[35m",  # Magenta
        "RESET": "\033[0m",  # Reset
    }

    def format(self, record: logging.LogRecord) -> str:
        # Get correlation info
        correlation_id = getattr(record, "correlation_id", "no-correlation")
        job_id = getattr(record, "job_id", "no-job")

        # Format timestamp
        timestamp = datetime.fromtimestamp(record.created).strftime("%H:%M:%S.%f")[:-3]

        # Build colored log line
        color = self.COLORS.get(record.levelname, "")
        reset = self.COLORS["RESET"]

        # Short correlation ID for readability
        short_correlation = (
            correlation_id[:8] if correlation_id != "no-correlation" else "system"
        )
        short_job = job_id[:8] if job_id != "no-job" else "main"

        log_line = (
            f"{color}[{timestamp}] {record.levelname:8}{reset} "
            f"{record.name:20} | {short_correlation}:{short_job} | {record.getMessage()}"
        )

        # Add exception info if present
        if record.exc_info:
            log_line += f"\n{self.formatException(record.exc_info)}"

        return log_line


class LoggingConfig:
    """Centralized logging configuration manager"""

    def __init__(self):
        self._configured = False
        self._log_level = None
        self._log_dir = None
        self._enable_json = None

    def configure(
        self,
        log_level: str = "INFO",
        log_dir: Optional[str] = None,
        enable_json: bool = None,
        enable_file_logging: bool = True,
        max_file_size: int = 50 * 1024 * 1024,  # 50MB
        backup_count: int = 5,
        console_format: str = "colored",  # "colored", "json", or "simple"
    ):
        """
        Configure logging for the application.

        Args:
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            log_dir: Directory for log files (None for temp dir)
            enable_json: Force JSON output (None for auto-detect based on environment)
            enable_file_logging: Whether to enable file logging
            max_file_size: Maximum size for log files before rotation
            backup_count: Number of backup files to keep
            console_format: Console output format
        """
        if self._configured:
            return

        # Determine environment and JSON usage
        is_production = os.getenv("ENVIRONMENT", "development").lower() == "production"
        is_container = os.path.exists("/.dockerenv") or os.getenv("CONTAINER") == "true"

        if enable_json is None:
            enable_json = is_production or is_container

        # Set up log directory
        if log_dir is None:
            if is_production:
                log_dir = "/var/log/montage"
            else:
                log_dir = Path.home() / ".cache" / "montage" / "logs"

        log_dir = Path(log_dir)
        if enable_file_logging:
            log_dir.mkdir(parents=True, exist_ok=True)

        # Store configuration
        self._log_level = log_level.upper()
        self._log_dir = log_dir
        self._enable_json = enable_json

        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(getattr(logging, self._log_level))

        # Clear existing handlers
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)

        # Add correlation filter to all handlers
        correlation_filter = CorrelationFilter()

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.addFilter(correlation_filter)

        if console_format == "json" or (enable_json and console_format != "colored"):
            console_handler.setFormatter(JSONFormatter())
        elif console_format == "colored" and not enable_json:
            console_handler.setFormatter(ColoredConsoleFormatter())
        else:
            console_handler.setFormatter(
                logging.Formatter(
                    "%(asctime)s - %(name)s - %(levelname)s - [%(correlation_id)s:%(job_id)s] - %(message)s"
                )
            )

        root_logger.addHandler(console_handler)

        # File handlers (if enabled)
        if enable_file_logging:
            # Main log file (all levels)
            main_log_file = log_dir / "montage.log"
            main_handler = logging.handlers.RotatingFileHandler(
                main_log_file,
                maxBytes=max_file_size,
                backupCount=backup_count,
                encoding="utf-8",
            )
            main_handler.addFilter(correlation_filter)
            main_handler.setFormatter(JSONFormatter())
            root_logger.addHandler(main_handler)

            # Error log file (WARNING and above)
            error_log_file = log_dir / "montage-error.log"
            error_handler = logging.handlers.RotatingFileHandler(
                error_log_file,
                maxBytes=max_file_size,
                backupCount=backup_count,
                encoding="utf-8",
            )
            error_handler.setLevel(logging.WARNING)
            error_handler.addFilter(correlation_filter)
            error_handler.setFormatter(JSONFormatter())
            root_logger.addHandler(error_handler)

        # Set specific logger levels for noisy libraries
        self._configure_third_party_loggers()

        self._configured = True

        # Log configuration complete
        logger = logging.getLogger(__name__)
        logger.info(
            "Logging configured",
            extra={
                "log_level": self._log_level,
                "log_dir": str(self._log_dir) if enable_file_logging else None,
                "json_enabled": enable_json,
                "file_logging_enabled": enable_file_logging,
                "environment": os.getenv("ENVIRONMENT", "development"),
            },
        )

    def _configure_third_party_loggers(self):
        """Configure logging levels for third-party libraries"""
        # Reduce verbosity of noisy libraries
        noisy_loggers = {
            "urllib3": logging.WARNING,
            "requests": logging.WARNING,
            "boto3": logging.WARNING,
            "botocore": logging.WARNING,
            "PIL": logging.WARNING,
            "matplotlib": logging.WARNING,
            "transformers": logging.WARNING,
            "huggingface_hub": logging.WARNING,
            "faster_whisper": logging.WARNING,
        }

        for logger_name, level in noisy_loggers.items():
            logging.getLogger(logger_name).setLevel(level)


def get_logger(name: str = None) -> logging.Logger:
    """
    Get a logger instance with proper configuration.

    Args:
        name: Logger name (defaults to caller's module name)

    Returns:
        Configured logger instance
    """
    if name is None:
        # Try to determine caller's module name
        import inspect

        frame = inspect.currentframe()
        try:
            caller_frame = frame.f_back
            name = caller_frame.f_globals.get("__name__", "__main__")
        finally:
            del frame

    return logging.getLogger(name)


@contextmanager
def correlation_context(correlation_id: str = None, job_id: str = None):
    """
    Context manager for adding correlation IDs to logs.

    Args:
        correlation_id: Unique correlation ID for tracing requests
        job_id: Job ID for tracking processing jobs
    """
    # Generate correlation ID if not provided
    if correlation_id is None:
        correlation_id = str(uuid.uuid4())

    # Store previous values
    old_correlation_id = getattr(_context_storage, "correlation_id", None)
    old_job_id = getattr(_context_storage, "job_id", None)

    # Set new values
    _context_storage.correlation_id = correlation_id
    if job_id is not None:
        _context_storage.job_id = job_id

    try:
        yield correlation_id
    finally:
        # Restore previous values
        _context_storage.correlation_id = old_correlation_id
        _context_storage.job_id = old_job_id


def log_function_call(
    logger: logging.Logger = None, log_args: bool = False, log_return: bool = False
):
    """
    Decorator to log function calls with timing information.

    Args:
        logger: Logger to use (defaults to function's module logger)
        log_args: Whether to log function arguments
        log_return: Whether to log return value
    """

    def decorator(func):
        def wrapper(*args, **kwargs):
            func_logger = logger or get_logger(func.__module__)

            start_time = time.time()

            # Log function entry
            log_data = {"function": func.__name__}
            if log_args:
                log_data["args"] = str(args)[:200]  # Truncate long args
                log_data["kwargs"] = {k: str(v)[:200] for k, v in kwargs.items()}

            func_logger.debug(f"Entering {func.__name__}", extra=log_data)

            try:
                result = func(*args, **kwargs)

                # Log successful completion
                elapsed = time.time() - start_time
                completion_data = {
                    "function": func.__name__,
                    "elapsed_seconds": round(elapsed, 3),
                    "status": "success",
                }

                if log_return:
                    completion_data["return_value"] = str(result)[
                        :200
                    ]  # Truncate long returns

                func_logger.debug(f"Completed {func.__name__}", extra=completion_data)
                return result

            except Exception as e:
                # Log error
                elapsed = time.time() - start_time
                error_data = {
                    "function": func.__name__,
                    "elapsed_seconds": round(elapsed, 3),
                    "status": "error",
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                }

                func_logger.error(
                    f"Failed {func.__name__}", extra=error_data, exc_info=True
                )
                raise

        return wrapper

    return decorator


def log_performance(operation_name: str, logger: logging.Logger = None):
    """
    Context manager for logging performance metrics.

    Args:
        operation_name: Name of the operation being measured
        logger: Logger to use (defaults to caller's module logger)
    """

    @contextmanager
    def performance_context():
        start_time = time.time()
        start_memory = _get_memory_usage()

        perf_logger = logger or get_logger()

        perf_logger.info(
            f"Starting {operation_name}",
            extra={"operation": operation_name, "start_memory_mb": start_memory},
        )

        try:
            yield

            # Log success
            elapsed = time.time() - start_time
            end_memory = _get_memory_usage()
            memory_delta = end_memory - start_memory

            perf_logger.info(
                f"Completed {operation_name}",
                extra={
                    "operation": operation_name,
                    "elapsed_seconds": round(elapsed, 3),
                    "memory_delta_mb": round(memory_delta, 2),
                    "end_memory_mb": end_memory,
                    "status": "success",
                },
            )

        except Exception as e:
            # Log failure
            elapsed = time.time() - start_time
            end_memory = _get_memory_usage()

            perf_logger.error(
                f"Failed {operation_name}",
                extra={
                    "operation": operation_name,
                    "elapsed_seconds": round(elapsed, 3),
                    "end_memory_mb": end_memory,
                    "status": "error",
                    "error_type": type(e).__name__,
                },
                exc_info=True,
            )
            raise

    return performance_context()


def _get_memory_usage() -> float:
    """Get current memory usage in MB"""
    try:
        import psutil

        return psutil.Process().memory_info().rss / 1024 / 1024
    except ImportError:
        return 0.0


# Global logging configuration instance
_config = LoggingConfig()


# Convenience functions
def configure_logging(*args, **kwargs):
    """Configure logging - see LoggingConfig.configure for parameters"""
    return _config.configure(*args, **kwargs)


def is_configured() -> bool:
    """Check if logging has been configured"""
    return _config._configured


# Auto-configure if not done
def _auto_configure():
    """Auto-configure logging with sensible defaults"""
    if not is_configured():
        configure_logging()


# Initialize with defaults on import
_auto_configure()
