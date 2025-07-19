"""Retry utilities with exponential backoff"""

import time
import logging
import functools
from typing import Type, Tuple, Callable, Any, Optional
import random
import os
import subprocess
import tempfile
import shutil

logger = logging.getLogger(__name__)


def retry_with_backoff(
    exceptions: Tuple[Type[Exception], ...] = (Exception,),
    max_retries: int = 3,
    initial_delay: float = 1.0,
    backoff_factor: float = 2.0,
    max_delay: float = 60.0,
    jitter: bool = True,
):
    """
    Decorator for retrying functions with exponential backoff

    Args:
        exceptions: Tuple of exception types to retry on
        max_retries: Maximum number of retry attempts
        initial_delay: Initial delay in seconds
        backoff_factor: Multiplier for delay after each retry
        max_delay: Maximum delay between retries
        jitter: Add random jitter to delays to prevent thundering herd
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            delay = initial_delay
            last_exception = None

            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)

                except exceptions as e:
                    last_exception = e

                    if attempt < max_retries:
                        # Add jitter to prevent thundering herd
                        actual_delay = delay
                        if jitter:
                            actual_delay = delay * (0.5 + random.random())

                        logger.warning(
                            f"{func.__name__} failed (attempt {attempt + 1}/{max_retries + 1}): {e}. "
                            f"Retrying in {actual_delay:.1f}s..."
                        )
                        time.sleep(actual_delay)

                        # Calculate next delay with backoff
                        delay = min(delay * backoff_factor, max_delay)
                    else:
                        logger.error(
                            f"{func.__name__} failed after {max_retries + 1} attempts: {e}"
                        )

            raise last_exception

        return wrapper

    return decorator


def retry_on_error(
    max_retries: int = 3,
    delay: float = 1.0,
    exceptions: Tuple[Type[Exception], ...] = (Exception,),
):
    """Simple retry decorator with fixed delay"""

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    if attempt < max_retries - 1:
                        logger.warning(f"{func.__name__} failed: {e}. Retrying...")
                        time.sleep(delay)
                    else:
                        raise

        return wrapper

    return decorator


class RetryableOperation:
    """Class for operations that need retry logic"""

    def __init__(
        self,
        operation: Callable,
        max_retries: int = 3,
        initial_delay: float = 1.0,
        backoff_factor: float = 2.0,
        exceptions: Tuple[Type[Exception], ...] = (Exception,),
    ):
        self.operation = operation
        self.max_retries = max_retries
        self.initial_delay = initial_delay
        self.backoff_factor = backoff_factor
        self.exceptions = exceptions

    def execute(self, *args, **kwargs) -> Any:
        """Execute the operation with retry logic"""
        delay = self.initial_delay
        last_exception = None

        for attempt in range(self.max_retries + 1):
            try:
                return self.operation(*args, **kwargs)

            except self.exceptions as e:
                last_exception = e

                if attempt < self.max_retries:
                    logger.warning(
                        f"Operation failed (attempt {attempt + 1}/{self.max_retries + 1}): {e}"
                    )
                    time.sleep(delay)
                    delay *= self.backoff_factor

        raise last_exception


# Specific retry decorators for common use cases

# For API calls
api_retry = retry_with_backoff(
    exceptions=(ConnectionError, TimeoutError, OSError),
    max_retries=3,
    initial_delay=1.0,
    backoff_factor=2.0,
    max_delay=30.0,
)

# For database operations
try:
    import psycopg2

    db_retry = retry_with_backoff(
        exceptions=(psycopg2.OperationalError, psycopg2.InterfaceError),
        max_retries=3,
        initial_delay=0.5,
        backoff_factor=2.0,
        max_delay=10.0,
    )
except ImportError:
    # If psycopg2 not available, use generic retry
    db_retry = retry_with_backoff(
        exceptions=(Exception,),
        max_retries=3,
        initial_delay=0.5,
        backoff_factor=2.0,
        max_delay=10.0,
    )

# For file operations
file_retry = retry_with_backoff(
    exceptions=(IOError, OSError, PermissionError),
    max_retries=3,
    initial_delay=0.1,
    backoff_factor=2.0,
    max_delay=5.0,
)

# For FFmpeg operations
ffmpeg_retry = retry_with_backoff(
    exceptions=(subprocess.CalledProcessError,),
    max_retries=2,
    initial_delay=2.0,
    backoff_factor=2.0,
    max_delay=10.0,
)


# Example usage functions


@api_retry
def call_openai_api(prompt: str, api_key: str) -> dict:
    """Call OpenAI API with automatic retry"""
    import openai

    client = openai.OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model="gpt-4", messages=[{"role": "user", "content": prompt}], max_tokens=500
    )

    return response.dict()


@db_retry
def execute_db_query(query: str, params: Optional[tuple] = None):
    """Execute database query with automatic retry"""
    from db_secure import db_pool

    with db_pool.get_cursor() as cursor:
        cursor.execute(query, params)
        return cursor.fetchall()


@file_retry
def safe_file_write(path: str, content: str):
    """Write file with automatic retry on failure"""

    # Write to temp file first
    temp_fd, temp_path = tempfile.mkstemp(dir=os.path.dirname(path))
    try:
        with os.fdopen(temp_fd, "w") as f:
            f.write(content)

        # Atomic move
        shutil.move(temp_path, path)
    except Exception:
        if os.path.exists(temp_path):
            os.unlink(temp_path)
        raise


@ffmpeg_retry
def run_ffmpeg_command(cmd: list) -> subprocess.CompletedProcess:
    """Run FFmpeg command with automatic retry"""
    import subprocess

    result = subprocess.run(cmd, check=True, capture_output=True, text=True)

    return result
