"""
Async utilities to replace blocking operations
"""

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Callable

from .logging_config import get_logger

logger = get_logger(__name__)


class NonBlockingDelay:
    """Non-blocking delay utilities"""

    @staticmethod
    async def async_sleep(seconds: float, reason: str = "delay"):
        """Non-blocking async sleep with logging"""
        logger.debug(f"Async sleep for {seconds}s: {reason}")
        await asyncio.sleep(seconds)

    @staticmethod
    def adaptive_delay(base_delay: float, attempt: int, max_delay: float = 30.0) -> float:
        """Calculate exponential backoff delay"""
        delay = min(base_delay * (2 ** attempt), max_delay)
        return delay

    @staticmethod
    async def retry_with_backoff(
        func: Callable,
        max_attempts: int = 3,
        base_delay: float = 1.0,
        exceptions: tuple = (Exception,),
        *args,
        **kwargs
    ):
        """Retry function with exponential backoff"""

        for attempt in range(max_attempts):
            try:
                if asyncio.iscoroutinefunction(func):
                    return await func(*args, **kwargs)
                else:
                    # Run sync function in thread pool
                    loop = asyncio.get_event_loop()
                    with ThreadPoolExecutor(max_workers=1) as executor:
                        return await loop.run_in_executor(executor, lambda: func(*args, **kwargs))

            except exceptions as e:
                if attempt == max_attempts - 1:
                    logger.error(f"Function {func.__name__} failed after {max_attempts} attempts: {e}")
                    raise

                delay = NonBlockingDelay.adaptive_delay(base_delay, attempt)
                logger.warning(f"Attempt {attempt + 1} failed for {func.__name__}: {e}. Retrying in {delay}s")
                await NonBlockingDelay.async_sleep(delay, f"retry attempt {attempt + 1}")


class PerformanceMonitor:
    """Monitor performance without blocking"""

    def __init__(self, name: str):
        self.name = name
        self.start_time = None
        self.checkpoints = []

    def start(self):
        """Start timing"""
        self.start_time = time.time()
        logger.debug(f"Performance monitor started: {self.name}")

    def checkpoint(self, label: str):
        """Add checkpoint"""
        if self.start_time is None:
            self.start()

        elapsed = time.time() - self.start_time
        self.checkpoints.append((label, elapsed))
        logger.debug(f"{self.name} checkpoint '{label}': {elapsed:.2f}s")

    def finish(self):
        """Finish and log results"""
        if self.start_time is None:
            return

        total_time = time.time() - self.start_time

        # Log summary
        logger.info(f"{self.name} completed in {total_time:.2f}s")
        for label, elapsed in self.checkpoints:
            percentage = (elapsed / total_time) * 100 if total_time > 0 else 0
            logger.info(f"  {label}: {elapsed:.2f}s ({percentage:.1f}%)")


def make_non_blocking(sync_func: Callable) -> Callable:
    """Decorator to make synchronous functions non-blocking"""

    async def async_wrapper(*args, **kwargs):
        """Async wrapper for sync function"""
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor(max_workers=1) as executor:
            return await loop.run_in_executor(executor, lambda: sync_func(*args, **kwargs))

    return async_wrapper


# Common patterns
async def wait_for_condition(
    condition_func: Callable[[], bool],
    timeout: float = 30.0,
    poll_interval: float = 0.1,
    condition_name: str = "condition"
):
    """Wait for condition without blocking"""

    start_time = time.time()

    while time.time() - start_time < timeout:
        if condition_func():
            return True

        await NonBlockingDelay.async_sleep(poll_interval, f"polling {condition_name}")

    logger.warning(f"Condition '{condition_name}' not met after {timeout}s timeout")
    return False
