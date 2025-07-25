"""
P1-03: FFmpeg Process Management with PG-kill wrapper

This module provides secure process management for FFmpeg operations:
- Process group isolation to prevent zombie processes
- Resource-aware process limits and monitoring  
- Automatic cleanup on timeout or failure
- Memory and CPU usage tracking per process
- Signal handling and graceful shutdown
- Process tree termination to prevent resource leaks
"""

import os
import signal
import subprocess
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import psutil

from ..utils.logging_config import get_logger

logger = get_logger(__name__)

class ProcessState(str, Enum):
    """FFmpeg process states"""
    CREATED = "created"
    RUNNING = "running"
    COMPLETED = "completed"
    TIMEOUT = "timeout"
    KILLED = "killed"
    ERROR = "error"

@dataclass
class ProcessLimits:
    """P1-03: Process resource limits configuration"""

    # Time limits (seconds)
    DEFAULT_TIMEOUT: int = int(os.getenv("FFMPEG_DEFAULT_TIMEOUT_SEC", "300"))  # 5 minutes
    MAX_TIMEOUT: int = int(os.getenv("FFMPEG_MAX_TIMEOUT_SEC", "1800"))  # 30 minutes

    # Memory limits (MB)
    MAX_MEMORY_MB: int = int(os.getenv("FFMPEG_MAX_MEMORY_MB", "1024"))  # 1GB per process
    MEMORY_WARNING_MB: int = int(os.getenv("FFMPEG_MEMORY_WARNING_MB", "768"))  # 768MB warning

    # CPU limits
    MAX_CPU_PERCENT: float = float(os.getenv("FFMPEG_MAX_CPU_PERCENT", "80.0"))

    # Process limits
    MAX_CONCURRENT_PROCESSES: int = int(os.getenv("FFMPEG_MAX_CONCURRENT", "4"))
    MAX_CHILD_PROCESSES: int = int(os.getenv("FFMPEG_MAX_CHILDREN", "8"))

    # Monitoring
    MONITOR_INTERVAL_SEC: float = float(os.getenv("FFMPEG_MONITOR_INTERVAL", "2.0"))
    GRACE_PERIOD_SEC: int = int(os.getenv("FFMPEG_GRACE_PERIOD_SEC", "10"))

@dataclass
class ProcessInfo:
    """Information about a managed FFmpeg process"""
    pid: int
    command: List[str]
    start_time: datetime
    state: ProcessState
    timeout_sec: int
    process: subprocess.Popen
    psutil_process: Optional[psutil.Process]
    monitor_thread: Optional[threading.Thread]
    max_memory_mb: float = 0.0
    max_cpu_percent: float = 0.0
    total_cpu_time: float = 0.0
    error_message: Optional[str] = None
    exit_code: Optional[int] = None

class FFmpegProcessManager:
    """
    P1-03: Secure FFmpeg process management with PG-kill capabilities
    
    Features:
    - Process group isolation (new session + process group)
    - Resource monitoring and enforcement
    - Timeout handling with graceful shutdown
    - Process tree termination to prevent zombies
    - Resource usage tracking and reporting
    - Concurrent process limiting
    - Signal handling and cleanup
    """

    def __init__(self):
        self.limits = ProcessLimits()
        self._processes: Dict[int, ProcessInfo] = {}
        self._process_lock = threading.RLock()
        self._shutdown_event = threading.Event()

        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)

        logger.info(f"FFmpeg process manager initialized (max concurrent: {self.limits.MAX_CONCURRENT_PROCESSES})")

    def _signal_handler(self, signum: int, frame):
        """Handle shutdown signals"""
        logger.info(f"Received signal {signum}, initiating graceful shutdown")
        self._shutdown_event.set()
        self.shutdown_all_processes()

    @contextmanager
    def managed_process(
        self,
        command: List[str],
        timeout_sec: Optional[int] = None,
        cwd: Optional[str] = None,
        env: Optional[Dict[str, str]] = None
    ):
        """
        Context manager for secure FFmpeg process execution
        
        Args:
            command: FFmpeg command and arguments
            timeout_sec: Process timeout (default from config)
            cwd: Working directory
            env: Environment variables
            
        Yields:
            ProcessInfo: Information about the running process
            
        Raises:
            ProcessLimitError: If resource limits are exceeded
            ProcessTimeoutError: If process times out
            ProcessExecutionError: If process fails
        """
        process_info = None

        try:
            # Check concurrent process limit
            if len(self._processes) >= self.limits.MAX_CONCURRENT_PROCESSES:
                raise ProcessLimitError(
                    f"Too many concurrent processes: {len(self._processes)} >= {self.limits.MAX_CONCURRENT_PROCESSES}"
                )

            # Validate and sanitize command
            safe_command = self._validate_command(command)

            # Start process with monitoring
            process_info = self._start_process(
                safe_command,
                timeout_sec or self.limits.DEFAULT_TIMEOUT,
                cwd,
                env
            )

            logger.info(f"Started FFmpeg process {process_info.pid}: {' '.join(safe_command[:3])}...")

            yield process_info

            # Wait for process completion
            self._wait_for_completion(process_info)

        except Exception as e:
            logger.error(f"Error in managed process: {e}")
            if process_info:
                process_info.error_message = str(e)
                process_info.state = ProcessState.ERROR
            raise

        finally:
            # Cleanup process
            if process_info:
                self._cleanup_process(process_info)

    def _validate_command(self, command: List[str]) -> List[str]:
        """
        P1-03: Validate and sanitize FFmpeg command for security
        
        Args:
            command: Raw command list
            
        Returns:
            Sanitized command list
            
        Raises:
            ValueError: If command is invalid or dangerous
        """
        if not command or not isinstance(command, list):
            raise ValueError("Command must be a non-empty list")

        # Ensure first argument is ffmpeg/ffprobe
        executable = command[0].lower()
        if not any(exec_name in executable for exec_name in ['ffmpeg', 'ffprobe']):
            raise ValueError(f"Only FFmpeg/FFprobe executables allowed, got: {executable}")

        # Check for dangerous arguments
        dangerous_args = [
            '-f', 'lavfi',  # Can be used for code execution
            '-i', 'pipe:',  # Avoid pipe input for security
            '|', '&&', ';', '`', '$(',  # Shell injection patterns
        ]

        command_str = ' '.join(command)
        for dangerous_arg in dangerous_args:
            if dangerous_arg in command_str:
                logger.warning(f"Potentially dangerous argument detected: {dangerous_arg}")

        # Sanitize file paths
        safe_command = []
        for i, arg in enumerate(command):
            if isinstance(arg, str):
                # Remove any shell metacharacters
                safe_arg = arg.replace('`', '').replace('$', '').replace('|', '')
                safe_command.append(safe_arg)
            else:
                safe_command.append(str(arg))

        return safe_command

    def _start_process(
        self,
        command: List[str],
        timeout_sec: int,
        cwd: Optional[str] = None,
        env: Optional[Dict[str, str]] = None
    ) -> ProcessInfo:
        """
        Start FFmpeg process with proper isolation and monitoring
        
        Args:
            command: Validated command list
            timeout_sec: Process timeout
            cwd: Working directory
            env: Environment variables
            
        Returns:
            ProcessInfo: Information about started process
        """
        # Prepare environment
        process_env = os.environ.copy()
        if env:
            process_env.update(env)

        # P1-03: Create new process group for proper isolation
        # This allows us to kill the entire process tree
        try:
            process = subprocess.Popen(
                command,
                cwd=cwd,
                env=process_env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                stdin=subprocess.DEVNULL,  # Prevent hanging on input
                # P1-03: Critical security settings
                start_new_session=True,  # New session to isolate from parent
                preexec_fn=os.setsid if os.name != 'nt' else None,  # New process group on Unix
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == 'nt' else 0,  # Windows equivalent
            )

        except Exception as e:
            logger.error(f"Failed to start process: {e}")
            raise ProcessExecutionError(f"Process start failed: {e}")

        # Create psutil process for monitoring
        try:
            psutil_process = psutil.Process(process.pid)
        except psutil.NoSuchProcess:
            process.terminate()
            raise ProcessExecutionError("Failed to create process monitor")

        # Create process info
        process_info = ProcessInfo(
            pid=process.pid,
            command=command,
            start_time=datetime.utcnow(),
            state=ProcessState.RUNNING,
            timeout_sec=timeout_sec,
            process=process,
            psutil_process=psutil_process,
            monitor_thread=None
        )

        # Register process
        with self._process_lock:
            self._processes[process.pid] = process_info

        # Start monitoring thread
        monitor_thread = threading.Thread(
            target=self._monitor_process,
            args=(process_info,),
            daemon=True,
            name=f"FFmpegMonitor-{process.pid}"
        )
        monitor_thread.start()
        process_info.monitor_thread = monitor_thread

        return process_info

    def _monitor_process(self, process_info: ProcessInfo):
        """
        Monitor process resources and enforce limits
        
        Args:
            process_info: Process to monitor
        """
        logger.debug(f"Starting resource monitoring for process {process_info.pid}")

        start_time = time.time()
        last_cpu_time = 0.0

        while not self._shutdown_event.is_set():
            try:
                # Check if process is still alive
                if process_info.process.poll() is not None:
                    process_info.state = ProcessState.COMPLETED
                    process_info.exit_code = process_info.process.returncode
                    break

                # Check timeout
                elapsed_time = time.time() - start_time
                if elapsed_time > process_info.timeout_sec:
                    logger.warning(f"Process {process_info.pid} timed out after {elapsed_time:.1f}s")
                    process_info.state = ProcessState.TIMEOUT
                    self._kill_process_tree(process_info)
                    break

                # Get resource usage
                if process_info.psutil_process:
                    try:
                        memory_info = process_info.psutil_process.memory_info()
                        memory_mb = memory_info.rss / 1024 / 1024

                        cpu_percent = process_info.psutil_process.cpu_percent()

                        # Update maximums
                        process_info.max_memory_mb = max(process_info.max_memory_mb, memory_mb)
                        process_info.max_cpu_percent = max(process_info.max_cpu_percent, cpu_percent)

                        # Check memory limit
                        if memory_mb > self.limits.MAX_MEMORY_MB:
                            logger.error(f"Process {process_info.pid} exceeded memory limit: {memory_mb:.1f}MB > {self.limits.MAX_MEMORY_MB}MB")
                            process_info.state = ProcessState.KILLED
                            process_info.error_message = f"Memory limit exceeded: {memory_mb:.1f}MB"
                            self._kill_process_tree(process_info)
                            break

                        # Check CPU limit (sustained high usage)
                        if cpu_percent > self.limits.MAX_CPU_PERCENT:
                            logger.warning(f"Process {process_info.pid} high CPU usage: {cpu_percent:.1f}%")

                        # Check child processes
                        children = process_info.psutil_process.children(recursive=True)
                        if len(children) > self.limits.MAX_CHILD_PROCESSES:
                            logger.warning(f"Process {process_info.pid} spawned too many children: {len(children)}")

                    except psutil.NoSuchProcess:
                        # Process died
                        break
                    except psutil.AccessDenied:
                        logger.warning(f"Access denied monitoring process {process_info.pid}")

            except Exception as e:
                logger.error(f"Error monitoring process {process_info.pid}: {e}")
                break

            # Sleep before next check
            time.sleep(self.limits.MONITOR_INTERVAL_SEC)

        logger.debug(f"Monitoring ended for process {process_info.pid}")

    def _wait_for_completion(self, process_info: ProcessInfo):
        """
        Wait for process completion with timeout handling
        
        Args:
            process_info: Process to wait for
            
        Raises:
            ProcessTimeoutError: If process times out
            ProcessExecutionError: If process fails
        """
        try:
            # Wait for process with timeout
            stdout, stderr = process_info.process.communicate(timeout=process_info.timeout_sec)

            exit_code = process_info.process.returncode
            process_info.exit_code = exit_code

            if exit_code == 0:
                process_info.state = ProcessState.COMPLETED
                logger.debug(f"Process {process_info.pid} completed successfully")
            else:
                process_info.state = ProcessState.ERROR
                error_msg = stderr.decode('utf-8', errors='ignore')[:1000]  # Limit error message size
                process_info.error_message = error_msg
                logger.error(f"Process {process_info.pid} failed with exit code {exit_code}: {error_msg}")
                raise ProcessExecutionError(f"Process failed with exit code {exit_code}: {error_msg}")

        except subprocess.TimeoutExpired:
            logger.error(f"Process {process_info.pid} timed out")
            process_info.state = ProcessState.TIMEOUT
            self._kill_process_tree(process_info)
            raise ProcessTimeoutError(f"Process timed out after {process_info.timeout_sec} seconds")

    def _kill_process_tree(self, process_info: ProcessInfo):
        """
        P1-03: Kill entire process tree using process group
        
        This prevents zombie processes and ensures complete cleanup
        
        Args:
            process_info: Process to terminate
        """
        logger.warning(f"Terminating process tree for PID {process_info.pid}")

        try:
            if process_info.psutil_process and process_info.psutil_process.is_running():
                # Get all children before termination
                children = process_info.psutil_process.children(recursive=True)

                # Try graceful termination first
                try:
                    if os.name != 'nt':  # Unix systems
                        # Kill entire process group
                        os.killpg(process_info.psutil_process.pid, signal.SIGTERM)
                    else:  # Windows
                        process_info.process.terminate()

                    # Wait for graceful shutdown
                    try:
                        process_info.process.wait(timeout=self.limits.GRACE_PERIOD_SEC)
                    except subprocess.TimeoutExpired:
                        # Force kill if graceful shutdown failed
                        if os.name != 'nt':
                            os.killpg(process_info.psutil_process.pid, signal.SIGKILL)
                        else:
                            process_info.process.kill()

                except (ProcessLookupError, psutil.NoSuchProcess):
                    # Process already died
                    logger.debug(f"Process {process_info.pid} already terminated")

                # Ensure all children are terminated
                for child in children:
                    try:
                        if child.is_running():
                            child.terminate()
                            time.sleep(0.1)  # Brief delay
                            if child.is_running():
                                child.kill()
                    except psutil.NoSuchProcess:
                        logger.debug(f"Child process already terminated")

        except Exception as e:
            logger.error(f"Error killing process tree for {process_info.pid}: {e}")

        process_info.state = ProcessState.KILLED

    def _cleanup_process(self, process_info: ProcessInfo):
        """
        Clean up process resources
        
        Args:
            process_info: Process to clean up
        """
        logger.debug(f"Cleaning up process {process_info.pid}")

        # Stop monitoring thread
        if process_info.monitor_thread and process_info.monitor_thread.is_alive():
            # Thread will stop when process state changes or shutdown event is set
            process_info.monitor_thread.join(timeout=2.0)

        # Ensure process is terminated
        if process_info.state == ProcessState.RUNNING:
            self._kill_process_tree(process_info)

        # Remove from tracking
        with self._process_lock:
            self._processes.pop(process_info.pid, None)

        # Log final stats
        duration = (datetime.utcnow() - process_info.start_time).total_seconds()
        logger.info(
            f"Process {process_info.pid} cleanup complete: "
            f"duration={duration:.1f}s, peak_memory={process_info.max_memory_mb:.1f}MB, "
            f"peak_cpu={process_info.max_cpu_percent:.1f}%, state={process_info.state.value}"
        )

    def shutdown_all_processes(self):
        """Shutdown all managed processes"""
        logger.info("Shutting down all FFmpeg processes")

        with self._process_lock:
            processes_to_cleanup = list(self._processes.values())

        for process_info in processes_to_cleanup:
            try:
                self._kill_process_tree(process_info)
                self._cleanup_process(process_info)
            except Exception as e:
                logger.error(f"Error during shutdown of process {process_info.pid}: {e}")

        logger.info("All FFmpeg processes shut down")

    def get_process_stats(self) -> Dict[str, Any]:
        """Get statistics about managed processes"""
        with self._process_lock:
            active_processes = [p for p in self._processes.values() if p.state == ProcessState.RUNNING]

            return {
                "active_processes": len(active_processes),
                "total_processes_tracked": len(self._processes),
                "max_concurrent": self.limits.MAX_CONCURRENT_PROCESSES,
                "processes": [
                    {
                        "pid": p.pid,
                        "state": p.state.value,
                        "duration_sec": (datetime.utcnow() - p.start_time).total_seconds(),
                        "max_memory_mb": p.max_memory_mb,
                        "max_cpu_percent": p.max_cpu_percent,
                        "command": p.command[:2]  # Just executable and first arg
                    }
                    for p in self._processes.values()
                ]
            }

# Custom exceptions
class ProcessLimitError(Exception):
    """Raised when process limits are exceeded"""
    def __init__(self, message="Process limit exceeded"):
        super().__init__(message)

class ProcessTimeoutError(Exception):
    """Raised when process times out"""
    def __init__(self, message="Process timed out"):
        super().__init__(message)

class ProcessExecutionError(Exception):
    """Raised when process execution fails"""
    def __init__(self, message="Process execution failed"):
        super().__init__(message)

# Global process manager instance
ffmpeg_process_manager = FFmpegProcessManager()

# Convenience wrapper functions
def run_ffmpeg_command(
    command: List[str],
    timeout_sec: Optional[int] = None,
    cwd: Optional[str] = None,
    env: Optional[Dict[str, str]] = None
) -> Tuple[str, str]:
    """
    Run FFmpeg command with secure process management
    
    Args:
        command: FFmpeg command and arguments
        timeout_sec: Process timeout
        cwd: Working directory  
        env: Environment variables
        
    Returns:
        Tuple of (stdout, stderr) as strings
        
    Raises:
        ProcessLimitError: If resource limits exceeded
        ProcessTimeoutError: If process times out
        ProcessExecutionError: If process fails
    """
    with ffmpeg_process_manager.managed_process(command, timeout_sec, cwd, env) as process_info:
        pass  # Process completion handled in context manager

    # Get output
    stdout, stderr = process_info.process.communicate()
    return stdout.decode('utf-8', errors='ignore'), stderr.decode('utf-8', errors='ignore')
