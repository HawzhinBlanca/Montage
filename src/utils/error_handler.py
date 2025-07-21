#!/usr/bin/env python3
"""
Error handling utilities and decorators for the Montage pipeline.

Provides:
- Decorators for automatic error conversion and context injection
- Retry logic with configurable strategies
- Error recovery helpers
- Integration with logging and monitoring
"""

import functools
import logging
import time
from typing import Callable, Any, Optional, Type, Union, Tuple, Dict
from contextlib import contextmanager

from ..core.exceptions import (
    MontageError,
    ErrorContext,
    RetryStrategy,
    ErrorSeverity,
    handle_error,
    create_error_context,
)

logger = logging.getLogger(__name__)


def with_error_context(
    component: str, operation: str, severity: ErrorSeverity = ErrorSeverity.MEDIUM
) -> Callable:
    """
    Decorator to add error context to functions.

    Usage:
        @with_error_context("VideoProcessor", "extract_segments")
        def extract_segments(video_path: str, segments: List[Segment]):
            # function implementation
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Try to extract common context from arguments
            job_id = kwargs.get("job_id")
            video_path = kwargs.get("video_path")

            # Look in positional args if not in kwargs
            if not video_path and args:
                for arg in args:
                    if isinstance(arg, str) and (
                        arg.endswith(".mp4") or arg.endswith(".mov")
                    ):
                        video_path = arg
                        break

            try:
                return func(*args, **kwargs)
            except Exception as e:
                # Create context for the error
                context = create_error_context(
                    job_id=job_id,
                    video_path=video_path,
                    component=component,
                    operation=operation,
                    function=func.__name__,
                    args_count=len(args),
                    kwargs_keys=list(kwargs.keys()),
                )

                # Convert to MontageError if needed
                montage_error = handle_error(e, context)

                # Override severity if specified
                if (
                    hasattr(montage_error, "severity")
                    and severity != ErrorSeverity.MEDIUM
                ):
                    montage_error.severity = severity

                raise montage_error from e

        return wrapper

    return decorator


def with_retry(
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    retry_on: Optional[Tuple[Type[Exception], ...]] = None,
    ignore_retry_strategy: bool = False,
) -> Callable:
    """
    Decorator to add retry logic to functions.

    Args:
        max_attempts: Maximum number of retry attempts
        base_delay: Base delay between retries in seconds
        max_delay: Maximum delay between retries
        exponential_base: Base for exponential backoff
        retry_on: Tuple of exception types to retry on (None = all MontageErrors)
        ignore_retry_strategy: If True, ignore the error's retry strategy

    Usage:
        @with_retry(max_attempts=5, base_delay=2.0)
        def call_api():
            # function that might fail
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_error = None

            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_error = e

                    # Check if we should retry this error
                    if retry_on and not isinstance(e, retry_on):
                        raise

                    # Convert to MontageError to check retry strategy
                    if isinstance(e, MontageError):
                        montage_error = e
                    else:
                        montage_error = handle_error(e)

                    # Check retry strategy unless ignored
                    if not ignore_retry_strategy:
                        if montage_error.retry_strategy == RetryStrategy.NO_RETRY:
                            logger.debug(
                                f"Error marked as no-retry: {montage_error.message}"
                            )
                            raise

                    # Don't retry on the last attempt
                    if attempt >= max_attempts - 1:
                        raise

                    # Calculate delay based on retry strategy
                    if (
                        montage_error.retry_strategy
                        == RetryStrategy.EXPONENTIAL_BACKOFF
                    ):
                        delay = min(base_delay * (exponential_base**attempt), max_delay)
                    elif montage_error.retry_strategy == RetryStrategy.LINEAR_BACKOFF:
                        delay = min(base_delay * (attempt + 1), max_delay)
                    else:
                        delay = base_delay

                    logger.warning(
                        f"Retry attempt {attempt + 1}/{max_attempts} for {func.__name__} "
                        f"after {delay:.1f}s delay. Error: {str(e)}"
                    )

                    time.sleep(delay)

            # Should never reach here, but just in case
            if last_error:
                raise last_error
            else:
                raise RuntimeError(f"Unexpected retry loop exit in {func.__name__}")

        return wrapper

    return decorator


def safe_execute(
    func: Callable,
    default: Any = None,
    log_errors: bool = True,
    raise_on_critical: bool = True,
) -> Any:
    """
    Execute a function safely, returning a default value on error.

    Args:
        func: Function to execute
        default: Default value to return on error
        log_errors: Whether to log errors
        raise_on_critical: Whether to re-raise critical errors

    Returns:
        Function result or default value
    """
    try:
        return func()
    except Exception as e:
        montage_error = handle_error(e)

        if log_errors:
            logger.warning(f"Safe execution caught error: {montage_error.message}")

        # Re-raise critical errors if requested
        if raise_on_critical and montage_error.severity == ErrorSeverity.CRITICAL:
            raise

        return default


@contextmanager
def error_recovery_context(
    operation: str,
    cleanup: Optional[Callable] = None,
    fallback: Optional[Callable] = None,
):
    """
    Context manager for error recovery with cleanup and fallback.

    Args:
        operation: Description of the operation
        cleanup: Function to call for cleanup on error
        fallback: Function to call as fallback on error

    Usage:
        with error_recovery_context("process_video", cleanup=cleanup_temp_files):
            # risky operation
    """
    try:
        yield
    except Exception as e:
        logger.error(f"Error in {operation}: {str(e)}")

        # Run cleanup if provided
        if cleanup:
            try:
                logger.info(f"Running cleanup for {operation}")
                cleanup()
            except Exception as cleanup_error:
                logger.error(f"Cleanup failed: {str(cleanup_error)}")

        # Run fallback if provided
        if fallback:
            try:
                logger.info(f"Running fallback for {operation}")
                result = fallback()
                if result is not None:
                    return result
            except Exception as fallback_error:
                logger.error(f"Fallback failed: {str(fallback_error)}")

        # Re-raise the original error
        raise


def aggregate_errors(errors: list[Exception]) -> MontageError:
    """
    Aggregate multiple errors into a single MontageError.

    Args:
        errors: List of exceptions to aggregate

    Returns:
        Single MontageError with aggregated information
    """
    if not errors:
        raise ValueError("No errors to aggregate")

    if len(errors) == 1:
        return handle_error(errors[0])

    # Find the most severe error
    montage_errors = [handle_error(e) for e in errors]
    most_severe = max(
        montage_errors, key=lambda e: list(ErrorSeverity).index(e.severity)
    )

    # Create aggregated error
    error_messages = [str(e) for e in errors]
    aggregated = MontageError(
        f"Multiple errors occurred ({len(errors)} total): "
        + "; ".join(error_messages[:3]),
        context=most_severe.context,
        cause=most_severe.cause,
    )

    # Copy severity and retry strategy from most severe
    aggregated.severity = most_severe.severity
    aggregated.retry_strategy = most_severe.retry_strategy

    # Add all error details to metadata
    aggregated.context.metadata["all_errors"] = [
        {
            "type": type(e).__name__,
            "message": str(e),
            "severity": (
                getattr(e, "severity", ErrorSeverity.MEDIUM).value
                if isinstance(e, MontageError)
                else "unknown"
            ),
        }
        for e in errors
    ]

    return aggregated


class ErrorCollector:
    """
    Collect errors during batch operations without stopping execution.

    Usage:
        collector = ErrorCollector()

        for item in items:
            with collector.collect(item_id=item.id):
                process_item(item)

        if collector.has_errors():
            handle_batch_errors(collector.get_errors())
    """

    def __init__(self):
        self.errors: list[Tuple[Dict[str, Any], Exception]] = []

    @contextmanager
    def collect(self, **context):
        """Context manager to collect errors with context"""
        try:
            yield
        except Exception as e:
            self.errors.append((context, e))
            logger.warning(f"Collected error: {str(e)} (context: {context})")

    def has_errors(self) -> bool:
        """Check if any errors were collected"""
        return len(self.errors) > 0

    def get_errors(self) -> list[Tuple[Dict[str, Any], Exception]]:
        """Get all collected errors with their context"""
        return self.errors

    def raise_if_errors(self, message: str = "Batch operation failed"):
        """Raise an aggregated error if any errors were collected"""
        if self.has_errors():
            errors = [error for _, error in self.errors]
            aggregated = aggregate_errors(errors)
            aggregated.message = f"{message}: {aggregated.message}"

            # Add batch context
            aggregated.context.metadata["batch_errors"] = [
                {"context": ctx, "error": str(err)} for ctx, err in self.errors
            ]

            raise aggregated

    def clear(self):
        """Clear collected errors"""
        self.errors.clear()


# Specialized error handlers for common scenarios


def handle_ffmpeg_error(
    command: list[str],
    return_code: int,
    stderr: str,
    operation: str = "FFmpeg operation",
) -> MontageError:
    """
    Create appropriate error for FFmpeg failures.

    Args:
        command: FFmpeg command that failed
        return_code: Process return code
        stderr: Standard error output
        operation: Description of the operation

    Returns:
        Appropriate MontageError subclass
    """
    from ..core.exceptions import FFmpegError, DiskSpaceError, VideoDecodeError

    # Check for specific error patterns
    if "No space left" in stderr:
        return DiskSpaceError(
            f"{operation} failed: No disk space",
            required_gb=0,  # Unknown from stderr
            available_gb=0,
        )
    elif "Invalid data found" in stderr or "Invalid data" in stderr:
        video_path = None
        for i, arg in enumerate(command):
            if arg == "-i" and i + 1 < len(command):
                video_path = command[i + 1]
                break
        return VideoDecodeError(
            f"{operation} failed: Invalid video data",
            video_path=video_path or "unknown",
        )
    else:
        return FFmpegError(
            f"{operation} failed with code {return_code}",
            command=command,
            stderr=stderr,
        )


def handle_api_error(
    service: str,
    error: Exception,
    status_code: Optional[int] = None,
    response_body: Optional[str] = None,
) -> MontageError:
    """
    Create appropriate error for API failures.

    Args:
        service: Name of the API service
        error: Original exception
        status_code: HTTP status code if available
        response_body: Response body if available

    Returns:
        Appropriate MontageError subclass
    """
    from ..core.exceptions import APIError, RateLimitError, AuthenticationError

    # Check status code patterns
    if status_code == 429:
        # Try to extract retry-after header
        retry_after = None
        if hasattr(error, "response") and hasattr(error.response, "headers"):
            retry_after = error.response.headers.get("Retry-After")
            if retry_after:
                try:
                    retry_after = int(retry_after)
                except ValueError:
                    pass

        return RateLimitError(
            f"{service} rate limit exceeded", service=service, retry_after=retry_after
        )
    elif status_code in [401, 403]:
        return AuthenticationError(f"{service} authentication failed", service=service)
    else:
        api_error = APIError(
            f"{service} API error: {str(error)}",
            context=create_error_context(
                metadata={
                    "service": service,
                    "status_code": status_code,
                    "response_body": response_body[:500] if response_body else None,
                }
            ),
        )

        # Set retry strategy based on status code
        if status_code and 500 <= status_code < 600:
            api_error.retry_strategy = RetryStrategy.EXPONENTIAL_BACKOFF

        return api_error
