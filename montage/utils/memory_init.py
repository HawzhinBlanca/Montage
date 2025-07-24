#!/usr/bin/env python3
"""
Memory management system initialization for Montage video processing.
Call this at the start of your application to enable comprehensive memory management.
"""

import atexit
import signal
from typing import Optional

# Import memory management components
try:
    from ..utils.logging_config import get_logger
    from .ffmpeg_memory_manager import get_ffmpeg_memory_manager
    from .memory_manager import initialize_memory_management, shutdown_memory_management
    from .resource_manager import cleanup_video_resources, get_resource_tracker

    memory_components_available = True
except ImportError as e:
    memory_components_available = False
    print(f"WARNING: Memory management components not available: {e}")

    # Create no-op functions
    def initialize_memory_management():
        return None, None, None

    def shutdown_memory_management():
        logger.debug("Memory management shutdown called (stub mode)")

    def get_resource_tracker():
        return None

    def cleanup_video_resources():
        logger.debug("Video resource cleanup called (stub mode)")

    def get_ffmpeg_memory_manager():
        return None


if memory_components_available:
    logger = get_logger(__name__)
else:
    import logging

    logger = logging.getLogger(__name__)


class MemoryManagementSystem:
    """Centralized memory management system for Montage video processing"""

    def __init__(self):
        self.initialized = False
        self.monitor = None
        self.pressure_manager = None
        self.adaptive_config = None
        self.resource_tracker = None
        self.ffmpeg_manager = None

    def initialize(
        self, enable_monitoring: bool = True, monitoring_interval: float = 2.0
    ) -> bool:
        """Initialize the memory management system"""

        if not memory_components_available:
            logger.warning(
                "Memory management components not available - skipping initialization"
            )
            return False

        if self.initialized:
            logger.info("Memory management already initialized")
            return True

        try:
            logger.info("Initializing comprehensive memory management system...")

            # Initialize core components
            self.monitor, self.pressure_manager, self.adaptive_config = (
                initialize_memory_management()
            )
            self.resource_tracker = get_resource_tracker()
            self.ffmpeg_manager = get_ffmpeg_memory_manager()

            # Start monitoring if requested
            if enable_monitoring and self.monitor:
                self.monitor.start_monitoring()
                logger.info(
                    f"Memory monitoring started (interval: {monitoring_interval}s)"
                )

            # Register cleanup handlers
            atexit.register(self.shutdown)
            signal.signal(signal.SIGINT, self._signal_handler)
            signal.signal(signal.SIGTERM, self._signal_handler)

            # Log system information
            if self.monitor:
                stats = self.monitor.get_current_stats()
                logger.info(
                    f"System memory: {stats.total_mb:.0f}MB total, "
                    f"{stats.available_mb:.0f}MB available, "
                    f"pressure level: {stats.pressure_level.value}"
                )

            # Log adaptive configuration
            if self.adaptive_config:
                config = self.adaptive_config.get_current_config()
                logger.info(
                    f"Adaptive config: {config['max_workers']} workers, "
                    f"{config['chunk_size_mb']}MB chunks, "
                    f"{config['quality_preset']} preset"
                )

            self.initialized = True
            logger.info("Memory management system initialized successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize memory management: {e}")
            return False

    def shutdown(self):
        """Shutdown the memory management system"""
        if not self.initialized:
            return

        logger.info("Shutting down memory management system...")

        try:
            # Stop monitoring
            if self.monitor:
                self.monitor.stop_monitoring()

            # Force cleanup of FFmpeg processes
            if self.ffmpeg_manager:
                self.ffmpeg_manager.force_cleanup_processes()

            # Clean up all video resources
            cleanup_video_resources()

            # Shutdown core system
            shutdown_memory_management()

            self.initialized = False
            logger.info("Memory management system shutdown completed")

        except Exception as e:
            logger.error(f"Error during memory management shutdown: {e}")

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"Received signal {signum}, shutting down memory management...")
        self.shutdown()

    def get_status(self) -> dict:
        """Get current status of memory management system"""
        if not self.initialized or not memory_components_available:
            return {
                "initialized": False,
                "memory_management_available": memory_components_available,
            }

        status = {"initialized": True, "memory_management_available": True}

        try:
            # Memory stats
            if self.monitor:
                stats = self.monitor.get_current_stats()
                status["memory"] = {
                    "total_mb": stats.total_mb,
                    "available_mb": stats.available_mb,
                    "used_percent": stats.percent_used,
                    "pressure_level": stats.pressure_level.value,
                    "process_memory_mb": stats.process_memory_mb,
                }

            # Resource usage
            if self.resource_tracker:
                usage = self.resource_tracker.get_resource_usage()
                status["resources"] = usage

            # FFmpeg processes
            if self.ffmpeg_manager:
                status["ffmpeg"] = {
                    "active_processes": self.ffmpeg_manager.get_active_process_count(),
                    "total_memory_mb": self.ffmpeg_manager.get_total_memory_usage(),
                }

            # Adaptive configuration
            if self.adaptive_config:
                config = self.adaptive_config.get_current_config()
                status["config"] = config

        except Exception as e:
            logger.error(f"Error getting memory management status: {e}")
            status["error"] = str(e)

        return status

    def force_cleanup(self):
        """Force cleanup of all resources"""
        logger.info("Forcing cleanup of all resources...")

        try:
            if self.ffmpeg_manager:
                self.ffmpeg_manager.force_cleanup_processes()

            cleanup_video_resources()

            # Force garbage collection
            import gc

            gc.collect()

            logger.info("Force cleanup completed")

        except Exception as e:
            logger.error(f"Error during force cleanup: {e}")

    def get_memory_safe_config(self, video_path: Optional[str] = None) -> dict:
        """Get memory-safe configuration for video processing"""
        if not self.initialized or not self.adaptive_config:
            # Return safe defaults
            return {
                "max_workers": 2,
                "chunk_size_mb": 256,
                "quality_preset": "medium",
                "enable_hardware_accel": False,
                "max_concurrent_ffmpeg": 1,
            }

        return self.adaptive_config.get_current_config()


# Global instance
_memory_system = MemoryManagementSystem()


def init_memory_management(
    enable_monitoring: bool = True, monitoring_interval: float = 2.0
) -> bool:
    """Initialize global memory management system"""
    return _memory_system.initialize(enable_monitoring, monitoring_interval)


def shutdown_memory_management_system():
    """Shutdown global memory management system"""
    _memory_system.shutdown()


def get_memory_system() -> MemoryManagementSystem:
    """Get global memory management system"""
    return _memory_system


def get_memory_status() -> dict:
    """Get current memory management status"""
    return _memory_system.get_status()


def force_memory_cleanup():
    """Force cleanup of all memory resources"""
    _memory_system.force_cleanup()


def get_safe_processing_config(video_path: Optional[str] = None) -> dict:
    """Get memory-safe configuration for processing"""
    return _memory_system.get_memory_safe_config(video_path)


def is_memory_management_available() -> bool:
    """Check if memory management components are available"""
    return memory_components_available and _memory_system.initialized


# Convenience function for quick setup
def setup_memory_management() -> bool:
    """Quick setup function for memory management"""
    success = init_memory_management()

    if success:
        logger.info("Memory management setup completed successfully")

        # Log current status
        status = get_memory_status()
        if "memory" in status:
            memory_info = status["memory"]
            logger.info(
                f"Available memory: {memory_info['available_mb']:.0f}MB "
                f"({100 - memory_info['used_percent']:.1f}% free)"
            )

    else:
        logger.warning(
            "Memory management setup failed - continuing without memory optimization"
        )

    return success


if __name__ == "__main__":
    # Test memory management system
    print("Testing memory management system...")

    # Initialize
    success = setup_memory_management()
    print(f"Initialization: {'SUCCESS' if success else 'FAILED'}")

    if success:
        # Get status
        status = get_memory_status()
        print("System Status:")

        if "memory" in status:
            memory = status["memory"]
            print(
                f"  Memory: {memory['available_mb']:.0f}MB available ({memory['pressure_level']} pressure)"
            )

        if "resources" in status:
            resources = status["resources"]
            print(
                f"  Resources: {resources['tracked_files']} files, {resources['tracked_processes']} processes"
            )

        if "config" in status:
            config = status["config"]
            print(
                f"  Config: {config['max_workers']} workers, {config['chunk_size_mb']}MB chunks"
            )

        # Get safe config
        safe_config = get_safe_processing_config()
        print(f"Safe processing config: {safe_config}")

        print("\nMemory management test completed")

    # Cleanup
    shutdown_memory_management_system()
