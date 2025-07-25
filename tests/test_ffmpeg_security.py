"""
P1-03: FFmpeg process management security tests

Tests the secure FFmpeg process wrapper:
- Process isolation and group management
- Resource limits and monitoring
- Timeout handling and cleanup
- Command validation and injection prevention
- Zombie process prevention
"""

import os
import time
from unittest.mock import Mock, patch

import pytest

from montage.utils.ffmpeg_process_manager import (
    FFmpegProcessManager,
    ProcessLimitError,
    ProcessLimits,
    ProcessState,
    ProcessTimeoutError,
    run_ffmpeg_command,
)


class TestFFmpegProcessManager:
    """Test FFmpeg process management security controls"""

    def setup_method(self):
        """Setup test environment"""
        self.manager = FFmpegProcessManager()
        # Use smaller limits for faster testing
        self.manager.limits.DEFAULT_TIMEOUT = 5
        self.manager.limits.MAX_CONCURRENT_PROCESSES = 2

    def teardown_method(self):
        """Cleanup after tests"""
        self.manager.shutdown_all_processes()

    def test_command_validation_valid_ffmpeg(self):
        """Test command validation allows valid FFmpeg commands"""
        valid_commands = [
            ["ffmpeg", "-i", "input.mp4", "output.mp4"],
            ["ffprobe", "-v", "quiet", "-print_format", "json", "input.mp4"],
            ["/usr/bin/ffmpeg", "-f", "mp4", "-i", "test.mp4", "out.mp4"]
        ]

        for cmd in valid_commands:
            # Should not raise exception
            validated = self.manager._validate_command(cmd)
            assert len(validated) > 0
            assert any("ffmpeg" in validated[0] or "ffprobe" in validated[0] for _ in [None])

    def test_command_validation_blocks_non_ffmpeg(self):
        """Test command validation blocks non-FFmpeg executables"""
        invalid_commands = [
            ["ls", "-la"],
            ["cat", "/etc/passwd"],
            ["rm", "-rf", "/"],
            ["python", "script.py"],
            ["bash", "-c", "echo hello"]
        ]

        for cmd in invalid_commands:
            with pytest.raises(ValueError) as exc_info:
                self.manager._validate_command(cmd)
            assert "Only FFmpeg/FFprobe executables allowed" in str(exc_info.value)

    def test_command_validation_sanitizes_dangerous_args(self):
        """Test command validation sanitizes dangerous arguments"""
        # Test command with shell metacharacters
        dangerous_cmd = ["ffmpeg", "-i", "input`rm -rf /`.mp4", "output.mp4"]

        validated = self.manager._validate_command(dangerous_cmd)

        # Should remove backticks and dollar signs
        assert "`" not in " ".join(validated)
        assert "$" not in " ".join(validated)

    def test_command_validation_empty_command(self):
        """Test command validation rejects empty commands"""
        with pytest.raises(ValueError) as exc_info:
            self.manager._validate_command([])
        assert "non-empty list" in str(exc_info.value)

        with pytest.raises(ValueError):
            self.manager._validate_command(None)

    @patch('subprocess.Popen')
    @patch('psutil.Process')
    def test_process_isolation_settings(self, mock_psutil, mock_popen):
        """Test that process isolation settings are applied correctly"""
        mock_process = Mock()
        mock_process.pid = 12345
        mock_process.poll.return_value = None
        mock_popen.return_value = mock_process

        mock_psutil_process = Mock()
        mock_psutil.return_value = mock_psutil_process

        cmd = ["ffmpeg", "-i", "test.mp4", "output.mp4"]

        with self.manager.managed_process(cmd) as process_info:
            # Verify subprocess.Popen was called with isolation settings
            mock_popen.assert_called_once()
            call_kwargs = mock_popen.call_args[1]

            # Check critical security settings
            assert call_kwargs['start_new_session'] is True
            assert call_kwargs['stdin'] == subprocess.DEVNULL

            # Check Unix-specific settings (if not Windows)
            if os.name != 'nt':
                assert call_kwargs['preexec_fn'] is not None
            else:
                assert call_kwargs['creationflags'] == subprocess.CREATE_NEW_PROCESS_GROUP

    def test_concurrent_process_limit(self):
        """Test concurrent process limiting"""
        # Mock successful processes that don't actually run
        with patch('subprocess.Popen') as mock_popen, \
             patch('psutil.Process') as mock_psutil:

            mock_process = Mock()
            mock_process.pid = 12345
            mock_process.poll.return_value = 0  # Completed immediately
            mock_process.communicate.return_value = (b"", b"")
            mock_popen.return_value = mock_process

            mock_psutil_process = Mock()
            mock_psutil.return_value = mock_psutil_process

            cmd = ["ffmpeg", "-i", "test.mp4", "output.mp4"]

            # Fill up to the limit
            contexts = []
            for i in range(self.manager.limits.MAX_CONCURRENT_PROCESSES):
                mock_process.pid = 12345 + i  # Different PID for each
                ctx = self.manager.managed_process(cmd)
                contexts.append(ctx)
                ctx.__enter__()

            # Next process should fail
            with pytest.raises(ProcessLimitError) as exc_info:
                with self.manager.managed_process(cmd):
                    pass

            assert "Too many concurrent processes" in str(exc_info.value)

            # Cleanup
            for ctx in contexts:
                try:
                    ctx.__exit__(None, None, None)
                except:
                    pass

    @patch('subprocess.Popen')
    @patch('psutil.Process')
    def test_process_timeout_handling(self, mock_psutil, mock_popen):
        """Test process timeout is enforced"""
        # Mock a process that never completes
        mock_process = Mock()
        mock_process.pid = 12345
        mock_process.poll.return_value = None  # Never completes
        mock_process.communicate.side_effect = subprocess.TimeoutExpired(["ffmpeg"], 1)
        mock_popen.return_value = mock_process

        mock_psutil_process = Mock()
        mock_psutil_process.is_running.return_value = True
        mock_psutil_process.children.return_value = []
        mock_psutil.return_value = mock_psutil_process

        cmd = ["ffmpeg", "-i", "test.mp4", "output.mp4"]

        with pytest.raises(ProcessTimeoutError):
            with self.manager.managed_process(cmd, timeout_sec=1):  # Very short timeout
                pass

    @patch('subprocess.Popen')
    @patch('psutil.Process')
    def test_memory_limit_enforcement(self, mock_psutil, mock_popen):
        """Test memory usage monitoring and limits"""
        mock_process = Mock()
        mock_process.pid = 12345
        mock_process.poll.return_value = None
        mock_popen.return_value = mock_process

        # Mock process that exceeds memory limit
        mock_psutil_process = Mock()
        mock_memory_info = Mock()
        mock_memory_info.rss = (self.manager.limits.MAX_MEMORY_MB + 100) * 1024 * 1024  # Exceed limit
        mock_psutil_process.memory_info.return_value = mock_memory_info
        mock_psutil_process.cpu_percent.return_value = 50.0
        mock_psutil_process.children.return_value = []
        mock_psutil_process.is_running.return_value = True
        mock_psutil.return_value = mock_psutil_process

        cmd = ["ffmpeg", "-i", "test.mp4", "output.mp4"]

        # Process should be killed due to memory limit
        with self.manager.managed_process(cmd, timeout_sec=10) as process_info:
            # Wait a bit for monitoring to kick in
            time.sleep(0.1)

        # Process should have been killed due to memory
        assert process_info.state == ProcessState.KILLED
        assert "Memory limit exceeded" in process_info.error_message

    def test_process_stats_tracking(self):
        """Test process statistics are tracked correctly"""
        initial_stats = self.manager.get_process_stats()
        assert initial_stats["active_processes"] == 0
        assert initial_stats["total_processes_tracked"] == 0

        with patch('subprocess.Popen') as mock_popen, \
             patch('psutil.Process') as mock_psutil:

            mock_process = Mock()
            mock_process.pid = 12345
            mock_process.poll.return_value = 0
            mock_process.communicate.return_value = (b"success", b"")
            mock_popen.return_value = mock_process

            mock_psutil_process = Mock()
            mock_psutil.return_value = mock_psutil_process

            cmd = ["ffmpeg", "-i", "test.mp4", "output.mp4"]

            with self.manager.managed_process(cmd):
                # During execution
                stats = self.manager.get_process_stats()
                assert stats["active_processes"] == 1
                assert stats["total_processes_tracked"] == 1

        # After completion
        final_stats = self.manager.get_process_stats()
        assert final_stats["active_processes"] == 0

    def test_process_tree_termination(self):
        """Test that child processes are terminated properly"""
        with patch('subprocess.Popen') as mock_popen, \
             patch('psutil.Process') as mock_psutil, \
             patch('os.killpg') as mock_killpg:

            mock_process = Mock()
            mock_process.pid = 12345
            mock_process.poll.return_value = None
            mock_popen.return_value = mock_process

            # Mock parent process with children
            mock_child1 = Mock()
            mock_child1.is_running.return_value = True
            mock_child2 = Mock()
            mock_child2.is_running.return_value = True

            mock_psutil_process = Mock()
            mock_psutil_process.pid = 12345
            mock_psutil_process.is_running.return_value = True
            mock_psutil_process.children.return_value = [mock_child1, mock_child2]
            mock_psutil.return_value = mock_psutil_process

            cmd = ["ffmpeg", "-i", "test.mp4", "output.mp4"]

            with self.manager.managed_process(cmd, timeout_sec=1) as process_info:
                # Let timeout trigger
                time.sleep(0.1)

            # Verify process group was killed
            if os.name != 'nt':
                mock_killpg.assert_called()

            # Verify children were terminated
            mock_child1.terminate.assert_called()
            mock_child2.terminate.assert_called()


class TestRunFFmpegCommand:
    """Test the convenience wrapper function"""

    @patch('src.utils.ffmpeg_process_manager.ffmpeg_process_manager')
    def test_run_ffmpeg_command_success(self, mock_manager):
        """Test successful command execution"""
        mock_context = Mock()
        mock_process_info = Mock()
        mock_process_info.process.communicate.return_value = (b"stdout", b"stderr")
        mock_context.__enter__.return_value = mock_process_info
        mock_context.__exit__.return_value = None
        mock_manager.managed_process.return_value = mock_context

        cmd = ["ffmpeg", "-i", "test.mp4", "output.mp4"]
        stdout, stderr = run_ffmpeg_command(cmd)

        assert stdout == "stdout"
        assert stderr == "stderr"
        mock_manager.managed_process.assert_called_once_with(cmd, None, None, None)

    @patch('src.utils.ffmpeg_process_manager.ffmpeg_process_manager')
    def test_run_ffmpeg_command_with_options(self, mock_manager):
        """Test command execution with timeout and environment"""
        mock_context = Mock()
        mock_process_info = Mock()
        mock_process_info.process.communicate.return_value = (b"out", b"err")
        mock_context.__enter__.return_value = mock_process_info
        mock_context.__exit__.return_value = None
        mock_manager.managed_process.return_value = mock_context

        cmd = ["ffmpeg", "-i", "test.mp4", "output.mp4"]
        env = {"PATH": "/usr/bin"}
        cwd = "/tmp"

        stdout, stderr = run_ffmpeg_command(cmd, timeout_sec=300, cwd=cwd, env=env)

        assert stdout == "out"
        assert stderr == "err"
        mock_manager.managed_process.assert_called_once_with(cmd, 300, cwd, env)


class TestProcessLimits:
    """Test process limits configuration"""

    def test_default_limits(self):
        """Test default process limits are reasonable"""
        limits = ProcessLimits()

        assert limits.DEFAULT_TIMEOUT == 300  # 5 minutes
        assert limits.MAX_TIMEOUT == 1800     # 30 minutes
        assert limits.MAX_MEMORY_MB == 1024   # 1GB
        assert limits.MAX_CONCURRENT_PROCESSES == 4
        assert limits.MONITOR_INTERVAL_SEC == 2.0

    def test_limits_from_environment(self):
        """Test limits can be configured from environment variables"""
        with patch.dict(os.environ, {
            'FFMPEG_DEFAULT_TIMEOUT_SEC': '600',
            'FFMPEG_MAX_MEMORY_MB': '2048',
            'FFMPEG_MAX_CONCURRENT': '8'
        }):
            limits = ProcessLimits()

            assert limits.DEFAULT_TIMEOUT == 600
            assert limits.MAX_MEMORY_MB == 2048
            assert limits.MAX_CONCURRENT_PROCESSES == 8
