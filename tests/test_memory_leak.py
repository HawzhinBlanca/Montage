#!/usr/bin/env python3
"""
Unit tests for memory leak prevention and OOM guard functionality
"""

import asyncio
import pytest
from unittest.mock import MagicMock, patch

from montage.utils.memory_manager import enforce_oom_guard, kill_oldest_ffmpeg, get_available_mb


def test_oom_guard(monkeypatch):
    """Test OOM guard triggers when memory is below threshold"""
    from montage.utils.memory_manager import enforce_oom_guard

    # Mock low memory condition
    monkeypatch.setattr("montage.utils.memory_manager.get_available_mb", lambda: 50)

    # Track if kill function was called
    flag = {"killed": False}
    monkeypatch.setattr("montage.utils.memory_manager.kill_oldest_ffmpeg",
                        lambda: flag.__setitem__("killed", True) or True)

    # Run OOM guard with 100MB threshold
    result = enforce_oom_guard(threshold_mb=100)

    # Assert kill was triggered
    assert flag["killed"] is True
    assert result is True


def test_oom_guard_no_trigger(monkeypatch):
    """Test OOM guard doesn't trigger when memory is sufficient"""
    # Mock high memory condition
    monkeypatch.setattr("montage.utils.memory_manager.get_available_mb", lambda: 2000)

    # Track if kill function was called
    flag = {"killed": False}
    monkeypatch.setattr("montage.utils.memory_manager.kill_oldest_ffmpeg",
                        lambda: flag.__setitem__("killed", True) or True)

    # Run OOM guard with 100MB threshold
    result = enforce_oom_guard(threshold_mb=100)

    # Assert kill was NOT triggered
    assert flag["killed"] is False
    assert result is False


def test_kill_oldest_ffmpeg_no_processes(monkeypatch):
    """Test kill_oldest_ffmpeg when no FFmpeg processes exist"""
    # Mock psutil to return no FFmpeg processes
    mock_proc = MagicMock()
    mock_proc.info = {'name': 'python', 'create_time': 123456}

    monkeypatch.setattr("psutil.process_iter", lambda fields: [mock_proc])

    result = kill_oldest_ffmpeg()
    assert result is False


def test_kill_oldest_ffmpeg_success(monkeypatch):
    """Test kill_oldest_ffmpeg successfully kills oldest process"""
    # Create mock FFmpeg processes
    mock_proc1 = MagicMock()
    mock_proc1.info = {'name': 'ffmpeg', 'create_time': 100}
    mock_proc1.pid = 1234
    mock_proc1.terminate = MagicMock()
    mock_proc1.wait = MagicMock()

    mock_proc2 = MagicMock()
    mock_proc2.info = {'name': 'ffmpeg', 'create_time': 200}  # Newer
    mock_proc2.pid = 5678

    # Mock psutil
    monkeypatch.setattr("psutil.process_iter", lambda fields: [mock_proc1, mock_proc2])

    result = kill_oldest_ffmpeg()

    # Assert oldest process was terminated
    assert result is True
    mock_proc1.terminate.assert_called_once()
    mock_proc1.wait.assert_called_once_with(timeout=5)


def test_get_available_mb():
    """Test get_available_mb returns reasonable value"""
    mb = get_available_mb()
    assert isinstance(mb, float)
    assert mb > 0
    # Most systems have at least 100MB available
    assert mb > 100


def test_zombie_reaper():
    """Test zombie reaper functionality"""
    from montage.utils.ffmpeg_process_manager import FFmpegProcessManager

    manager = FFmpegProcessManager()

    # Test stats initialization
    stats = manager.get_stats()
    assert stats["tracked_pids"] == 0
    assert stats["zombies_reaped_total"] == 0
    assert isinstance(stats["active_ffmpeg_count"], int)
    assert stats["active_ffmpeg_count"] >= 0

    # Test process tracking
    manager.track_ffmpeg_process(1234)
    assert manager.get_stats()["tracked_pids"] == 1

    manager.untrack_ffmpeg_process(1234)
    assert manager.get_stats()["tracked_pids"] == 0

    # Test zombie reaping (should handle no zombies gracefully)
    reaped = manager.reap_zombies()
    assert reaped >= 0  # Should be 0 or more


@pytest.mark.asyncio
async def test_zombie_reaper_loop_cancellation():
    """Test zombie reaper loop handles cancellation properly"""
    from montage.utils.ffmpeg_process_manager import zombie_reaper_loop

    # Create task
    task = asyncio.create_task(zombie_reaper_loop(interval=0.1))

    # Let it run briefly
    await asyncio.sleep(0.2)

    # Cancel it
    task.cancel()

    # Should raise CancelledError
    with pytest.raises(asyncio.CancelledError):
        await task


def test_memory_metrics_integration():
    """Test memory manager metrics integration"""
    from montage.utils.memory_manager import MemoryMonitor

    monitor = MemoryMonitor()
    stats = monitor.get_current_stats()

    # Verify stats structure
    assert stats.total_mb > 0
    assert stats.available_mb > 0
    assert stats.used_mb > 0
    assert 0 <= stats.percent_used <= 100
    assert stats.process_memory_mb >= 0
    assert stats.pressure_level is not None
