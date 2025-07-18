"""Resource cleanup and error recovery manager"""

import os
import logging
import atexit
import signal
import psutil
from typing import Set, Dict, Any
from contextlib import contextmanager
import subprocess

logger = logging.getLogger(__name__)


class CleanupManager:
    """Manages resource cleanup and error recovery"""
    
    def __init__(self):
        self._resources: Dict[str, Set[str]] = {
            'fifos': set(),
            'temp_files': set(),
            'processes': set(),
            'temp_dirs': set()
        }
        self._cleanup_in_progress = False
        self._register_handlers()
    
    def _register_handlers(self):
        """Register cleanup handlers"""
        atexit.register(self.cleanup_all)
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)
        
        # Handle SIGPIPE for broken pipe errors
        if hasattr(signal, 'SIGPIPE'):
            signal.signal(signal.SIGPIPE, signal.SIG_DFL)
    
    def _signal_handler(self, signum, frame):
        """Handle termination signals"""
        if not self._cleanup_in_progress:
            logger.info(f"Received signal {signum}, cleaning up...")
            self.cleanup_all()
    
    def register_fifo(self, path: str):
        """Register a FIFO for cleanup"""
        self._resources['fifos'].add(path)
        logger.debug(f"Registered FIFO: {path}")
    
    def register_temp_file(self, path: str):
        """Register a temporary file for cleanup"""
        self._resources['temp_files'].add(path)
        logger.debug(f"Registered temp file: {path}")
    
    def register_temp_dir(self, path: str):
        """Register a temporary directory for cleanup"""
        self._resources['temp_dirs'].add(path)
        logger.debug(f"Registered temp dir: {path}")
    
    def register_process(self, pid: int):
        """Register a process for cleanup"""
        self._resources['processes'].add(str(pid))
        logger.debug(f"Registered process: {pid}")
    
    def cleanup_fifo(self, path: str):
        """Clean up a FIFO"""
        try:
            if os.path.exists(path):
                os.unlink(path)
                logger.debug(f"Cleaned up FIFO: {path}")
            self._resources['fifos'].discard(path)
        except Exception as e:
            logger.error(f"Failed to clean up FIFO {path}: {e}")
    
    def cleanup_temp_file(self, path: str):
        """Clean up a temporary file"""
        try:
            if os.path.exists(path):
                os.remove(path)
                logger.debug(f"Cleaned up temp file: {path}")
            self._resources['temp_files'].discard(path)
        except Exception as e:
            logger.error(f"Failed to clean up file {path}: {e}")
    
    def cleanup_temp_dir(self, path: str):
        """Clean up a temporary directory"""
        try:
            if os.path.exists(path):
                # Remove all files in directory first
                for root, dirs, files in os.walk(path, topdown=False):
                    for name in files:
                        os.remove(os.path.join(root, name))
                    for name in dirs:
                        os.rmdir(os.path.join(root, name))
                os.rmdir(path)
                logger.debug(f"Cleaned up temp dir: {path}")
            self._resources['temp_dirs'].discard(path)
        except Exception as e:
            logger.error(f"Failed to clean up directory {path}: {e}")
    
    def cleanup_process(self, pid: str):
        """Clean up a process"""
        try:
            pid_int = int(pid)
            if psutil.pid_exists(pid_int):
                proc = psutil.Process(pid_int)
                
                # Check if it's an FFmpeg process
                try:
                    cmdline = proc.cmdline()
                    if cmdline and 'ffmpeg' in cmdline[0]:
                        logger.info(f"Terminating FFmpeg process {pid}")
                        proc.terminate()
                        try:
                            proc.wait(timeout=5)
                        except psutil.TimeoutExpired:
                            logger.warning(f"Process {pid} didn't terminate, killing...")
                            proc.kill()
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
                    
            self._resources['processes'].discard(pid)
        except Exception as e:
            logger.error(f"Failed to clean up process {pid}: {e}")
    
    def cleanup_all(self):
        """Clean up all registered resources"""
        if self._cleanup_in_progress:
            return
            
        self._cleanup_in_progress = True
        logger.info("Cleaning up all resources...")
        
        # Clean up processes first (important for FFmpeg)
        for pid in list(self._resources['processes']):
            self.cleanup_process(pid)
        
        # Clean up FIFOs
        for fifo in list(self._resources['fifos']):
            self.cleanup_fifo(fifo)
        
        # Clean up temp files
        for temp_file in list(self._resources['temp_files']):
            self.cleanup_temp_file(temp_file)
        
        # Clean up temp directories
        for temp_dir in list(self._resources['temp_dirs']):
            self.cleanup_temp_dir(temp_dir)
        
        logger.info("Resource cleanup complete")
        self._cleanup_in_progress = False
    
    @contextmanager
    def managed_resource(self, resource_type: str, resource_id: str):
        """Context manager for automatic resource cleanup"""
        try:
            if resource_type == 'fifo':
                self.register_fifo(resource_id)
            elif resource_type == 'temp_file':
                self.register_temp_file(resource_id)
            elif resource_type == 'temp_dir':
                self.register_temp_dir(resource_id)
            elif resource_type == 'process':
                self.register_process(int(resource_id))
            
            yield resource_id
            
        finally:
            if resource_type == 'fifo':
                self.cleanup_fifo(resource_id)
            elif resource_type == 'temp_file':
                self.cleanup_temp_file(resource_id)
            elif resource_type == 'temp_dir':
                self.cleanup_temp_dir(resource_id)
            elif resource_type == 'process':
                self.cleanup_process(resource_id)
    
    @contextmanager
    def managed_ffmpeg_process(self, cmd: list):
        """Context manager for FFmpeg processes with automatic cleanup"""
        proc = None
        try:
            proc = subprocess.Popen(cmd, 
                                  stdout=subprocess.PIPE, 
                                  stderr=subprocess.PIPE)
            self.register_process(proc.pid)
            yield proc
        finally:
            if proc:
                if proc.poll() is None:  # Process still running
                    proc.terminate()
                    try:
                        proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                self.cleanup_process(str(proc.pid))
    
    def unregister_all(self):
        """Unregister all resources without cleanup (for testing)"""
        for resource_type in self._resources:
            self._resources[resource_type].clear()


# Global cleanup manager instance
cleanup_manager = CleanupManager()