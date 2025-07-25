#!/usr/bin/env python3
"""
FFmpeg process management with zombie reaper functionality
"""

import asyncio
import os
import signal
import time
from typing import Dict, List, Optional, Set

import psutil

from ..utils.logging_config import get_logger

logger = get_logger(__name__)


class FFmpegProcessManager:
    """Manages FFmpeg processes and prevents zombies"""

    def __init__(self):
        self._tracked_pids: Set[int] = set()
        self._zombies_reaped = 0
        self._last_reap_time = time.time()

    def track_ffmpeg_process(self, pid: int):
        """Track an FFmpeg process"""
        self._tracked_pids.add(pid)
        logger.debug(f"Tracking FFmpeg process: PID {pid}")

    def untrack_ffmpeg_process(self, pid: int):
        """Stop tracking an FFmpeg process"""
        self._tracked_pids.discard(pid)
        logger.debug(f"Untracked FFmpeg process: PID {pid}")

    def find_zombie_processes(self) -> List[psutil.Process]:
        """Find zombie processes on the system"""
        zombies = []
        try:
            for proc in psutil.process_iter(['pid', 'name', 'status', 'ppid']):
                try:
                    # Check if process is a zombie
                    if proc.info['status'] == psutil.STATUS_ZOMBIE:
                        zombies.append(proc)
                        logger.warning(f"Found zombie process: PID {proc.info['pid']} ({proc.info['name']})")
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
        except Exception as e:
            logger.error(f"Error finding zombie processes: {e}")

        return zombies

    def reap_zombies(self) -> int:
        """Reap zombie processes"""
        reaped_count = 0

        try:
            # First try to reap child zombies with os.waitpid
            while True:
                try:
                    pid, status = os.waitpid(-1, os.WNOHANG)
                    if pid == 0:
                        # No more zombie children
                        break
                    logger.info(f"Reaped zombie child process: PID {pid}")
                    reaped_count += 1
                    self._tracked_pids.discard(pid)
                except ChildProcessError:
                    # No child processes
                    break
                except Exception as e:
                    logger.debug(f"waitpid error: {e}")
                    break

            # Check for any remaining zombies
            zombies = self.find_zombie_processes()

            # Try to clean up zombies by sending SIGCHLD to their parent
            for zombie in zombies:
                try:
                    parent_pid = zombie.info['ppid']
                    if parent_pid and parent_pid != os.getpid():
                        os.kill(parent_pid, signal.SIGCHLD)
                        logger.info(f"Sent SIGCHLD to parent {parent_pid} of zombie {zombie.pid}")
                except Exception as e:
                    logger.debug(f"Failed to signal parent of zombie {zombie.pid}: {e}")

            if reaped_count > 0:
                self._zombies_reaped += reaped_count
                # metrics.zombies_reaped.inc(reaped_count)  # TODO: Add this metric
                logger.info(f"Reaped {reaped_count} zombie processes (total: {self._zombies_reaped})")

        except Exception as e:
            logger.error(f"Error reaping zombies: {e}")

        self._last_reap_time = time.time()
        return reaped_count

    def get_ffmpeg_process_count(self) -> int:
        """Get count of active FFmpeg processes"""
        count = 0
        try:
            for proc in psutil.process_iter(['name']):
                try:
                    if proc.info['name'] and 'ffmpeg' in proc.info['name'].lower():
                        count += 1
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
        except Exception as e:
            logger.error(f"Error counting FFmpeg processes: {e}")

        return count

    def get_stats(self) -> Dict:
        """Get process management statistics"""
        return {
            "tracked_pids": len(self._tracked_pids),
            "zombies_reaped_total": self._zombies_reaped,
            "active_ffmpeg_count": self.get_ffmpeg_process_count(),
            "last_reap_time": self._last_reap_time,
        }


# Global instance
_process_manager: Optional[FFmpegProcessManager] = None


def get_process_manager() -> FFmpegProcessManager:
    """Get or create global process manager"""
    global _process_manager
    if _process_manager is None:
        _process_manager = FFmpegProcessManager()
    return _process_manager


async def zombie_reaper_loop(interval: float = 60.0):
    """Async loop that periodically reaps zombie processes"""
    manager = get_process_manager()
    logger.info(f"Starting zombie reaper loop (interval: {interval}s)")

    try:
        while True:
            # Reap any zombies
            manager.reap_zombies()

            # Log stats periodically
            stats = manager.get_stats()
            if stats["zombies_reaped_total"] > 0 or stats["active_ffmpeg_count"] > 0:
                logger.info(
                    f"Process stats - Active FFmpeg: {stats['active_ffmpeg_count']}, "
                    f"Zombies reaped: {stats['zombies_reaped_total']}"
                )

            # Wait for next cycle
            await asyncio.sleep(interval)

    except asyncio.CancelledError:
        logger.info("Zombie reaper loop cancelled")
        raise
    except Exception as e:
        logger.error(f"Zombie reaper loop error: {e}")
        raise


if __name__ == "__main__":
    # Test zombie detection
    manager = get_process_manager()

    print("Checking for zombie processes...")
    zombies = manager.find_zombie_processes()
    print(f"Found {len(zombies)} zombie processes")

    if zombies:
        print("\nReaping zombies...")
        reaped = manager.reap_zombies()
        print(f"Reaped {reaped} zombie processes")

    print("\nProcess stats:")
    stats = manager.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
