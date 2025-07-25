#!/usr/bin/env python3
"""
Process Metrics and Monitoring Endpoint
Real-time resource monitoring and health checks
"""
import os
import psutil
import asyncio
import json
import logging
import time
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
from aiohttp import web
import threading

logger = logging.getLogger(__name__)


@dataclass
class ProcessMetrics:
    """Process resource metrics"""
    timestamp: float
    cpu_percent: float
    memory_mb: float
    memory_percent: float
    disk_io_read_mb: float
    disk_io_write_mb: float
    open_files: int
    threads: int
    gpu_usage_percent: Optional[float] = None
    gpu_memory_mb: Optional[float] = None


@dataclass
class SystemMetrics:
    """System-wide metrics"""
    total_cpu_percent: float
    total_memory_mb: float
    available_memory_mb: float
    disk_usage_percent: float
    network_sent_mb: float
    network_recv_mb: float


class MetricsCollector:
    """Collect and store process metrics"""
    
    def __init__(self, max_history: int = 300):
        """
        Initialize metrics collector
        
        Args:
            max_history: Maximum number of historical data points
        """
        self.max_history = max_history
        self.metrics_history: List[ProcessMetrics] = []
        self.system_history: List[SystemMetrics] = []
        self.process = psutil.Process()
        self.start_time = time.time()
        
        # Track cumulative I/O
        self.last_disk_io = self.process.io_counters()
        self.last_net_io = psutil.net_io_counters()
        
        # GPU monitoring (if available)
        self.gpu_available = self._check_gpu_availability()
        
    def _check_gpu_availability(self) -> bool:
        """Check if GPU monitoring is available"""
        try:
            import pynvml
            pynvml.nvmlInit()
            self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            return True
        except:
            logger.info("GPU monitoring not available")
            return False
            
    def collect_metrics(self) -> ProcessMetrics:
        """Collect current process metrics"""
        # CPU and memory
        cpu_percent = self.process.cpu_percent(interval=0.1)
        memory_info = self.process.memory_info()
        memory_mb = memory_info.rss / (1024 * 1024)
        memory_percent = self.process.memory_percent()
        
        # Disk I/O
        try:
            io_counters = self.process.io_counters()
            disk_read_mb = (io_counters.read_bytes - self.last_disk_io.read_bytes) / (1024 * 1024)
            disk_write_mb = (io_counters.write_bytes - self.last_disk_io.write_bytes) / (1024 * 1024)
            self.last_disk_io = io_counters
        except:
            disk_read_mb = 0
            disk_write_mb = 0
            
        # File handles and threads
        try:
            open_files = len(self.process.open_files())
        except:
            open_files = 0
            
        threads = self.process.num_threads()
        
        # GPU metrics
        gpu_usage = None
        gpu_memory = None
        if self.gpu_available:
            gpu_usage, gpu_memory = self._get_gpu_metrics()
            
        metrics = ProcessMetrics(
            timestamp=time.time(),
            cpu_percent=cpu_percent,
            memory_mb=memory_mb,
            memory_percent=memory_percent,
            disk_io_read_mb=disk_read_mb,
            disk_io_write_mb=disk_write_mb,
            open_files=open_files,
            threads=threads,
            gpu_usage_percent=gpu_usage,
            gpu_memory_mb=gpu_memory
        )
        
        # Add to history
        self.metrics_history.append(metrics)
        if len(self.metrics_history) > self.max_history:
            self.metrics_history.pop(0)
            
        return metrics
        
    def _get_gpu_metrics(self) -> Tuple[Optional[float], Optional[float]]:
        """Get GPU usage metrics"""
        try:
            import pynvml
            
            # GPU utilization
            util = pynvml.nvmlDeviceGetUtilizationRates(self.gpu_handle)
            gpu_usage = util.gpu
            
            # GPU memory
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(self.gpu_handle)
            gpu_memory_mb = mem_info.used / (1024 * 1024)
            
            return gpu_usage, gpu_memory_mb
        except:
            return None, None
            
    def collect_system_metrics(self) -> SystemMetrics:
        """Collect system-wide metrics"""
        # CPU
        cpu_percent = psutil.cpu_percent(interval=0.1)
        
        # Memory
        memory = psutil.virtual_memory()
        total_memory_mb = memory.total / (1024 * 1024)
        available_memory_mb = memory.available / (1024 * 1024)
        
        # Disk
        disk = psutil.disk_usage('/')
        disk_usage_percent = disk.percent
        
        # Network
        net_io = psutil.net_io_counters()
        network_sent_mb = (net_io.bytes_sent - self.last_net_io.bytes_sent) / (1024 * 1024)
        network_recv_mb = (net_io.bytes_recv - self.last_net_io.bytes_recv) / (1024 * 1024)
        self.last_net_io = net_io
        
        system_metrics = SystemMetrics(
            total_cpu_percent=cpu_percent,
            total_memory_mb=total_memory_mb,
            available_memory_mb=available_memory_mb,
            disk_usage_percent=disk_usage_percent,
            network_sent_mb=network_sent_mb,
            network_recv_mb=network_recv_mb
        )
        
        self.system_history.append(system_metrics)
        if len(self.system_history) > self.max_history:
            self.system_history.pop(0)
            
        return system_metrics
        
    def get_summary(self) -> Dict[str, Any]:
        """Get metrics summary"""
        if not self.metrics_history:
            return {}
            
        recent_metrics = self.metrics_history[-10:]  # Last 10 samples
        
        return {
            "current": asdict(self.metrics_history[-1]),
            "average": {
                "cpu_percent": sum(m.cpu_percent for m in recent_metrics) / len(recent_metrics),
                "memory_mb": sum(m.memory_mb for m in recent_metrics) / len(recent_metrics),
            },
            "peak": {
                "cpu_percent": max(m.cpu_percent for m in self.metrics_history),
                "memory_mb": max(m.memory_mb for m in self.metrics_history),
            },
            "uptime_seconds": time.time() - self.start_time
        }


class MetricsServer:
    """HTTP server for metrics endpoint"""
    
    def __init__(self, collector: MetricsCollector, port: int = 8000):
        self.collector = collector
        self.port = port
        self.app = web.Application()
        self._setup_routes()
        self.runner = None
        self.site = None
        
    def _setup_routes(self):
        """Setup HTTP routes"""
        self.app.router.add_get('/metrics/proc_mem', self.handle_proc_mem)
        self.app.router.add_get('/metrics/summary', self.handle_summary)
        self.app.router.add_get('/metrics/history', self.handle_history)
        self.app.router.add_get('/health', self.handle_health)
        
    async def handle_proc_mem(self, request):
        """Handle /metrics/proc_mem endpoint"""
        metrics = self.collector.collect_metrics()
        
        response = {
            "timestamp": datetime.fromtimestamp(metrics.timestamp).isoformat(),
            "process": {
                "cpu_percent": metrics.cpu_percent,
                "memory_mb": round(metrics.memory_mb, 2),
                "memory_percent": round(metrics.memory_percent, 2),
                "threads": metrics.threads,
                "open_files": metrics.open_files
            }
        }
        
        if metrics.gpu_usage_percent is not None:
            response["gpu"] = {
                "usage_percent": metrics.gpu_usage_percent,
                "memory_mb": round(metrics.gpu_memory_mb, 2)
            }
            
        return web.json_response(response)
        
    async def handle_summary(self, request):
        """Handle /metrics/summary endpoint"""
        summary = self.collector.get_summary()
        return web.json_response(summary)
        
    async def handle_history(self, request):
        """Handle /metrics/history endpoint"""
        # Get last N samples
        count = int(request.query.get('count', 100))
        
        history = []
        for metrics in self.collector.metrics_history[-count:]:
            history.append({
                "timestamp": metrics.timestamp,
                "cpu": metrics.cpu_percent,
                "memory_mb": metrics.memory_mb
            })
            
        return web.json_response({"history": history})
        
    async def handle_health(self, request):
        """Handle /health endpoint"""
        metrics = self.collector.collect_metrics()
        
        # Health checks
        is_healthy = True
        issues = []
        
        if metrics.cpu_percent > 90:
            is_healthy = False
            issues.append("High CPU usage")
            
        if metrics.memory_percent > 90:
            is_healthy = False
            issues.append("High memory usage")
            
        if metrics.open_files > 1000:
            issues.append("Many open files")
            
        return web.json_response({
            "status": "healthy" if is_healthy else "unhealthy",
            "issues": issues,
            "metrics": {
                "cpu_percent": metrics.cpu_percent,
                "memory_percent": metrics.memory_percent,
                "open_files": metrics.open_files
            }
        })
        
    async def start(self):
        """Start metrics server"""
        self.runner = web.AppRunner(self.app)
        await self.runner.setup()
        self.site = web.TCPSite(self.runner, 'localhost', self.port)
        await self.site.start()
        logger.info(f"Metrics server started on http://localhost:{self.port}")
        
    async def stop(self):
        """Stop metrics server"""
        if self.site:
            await self.site.stop()
        if self.runner:
            await self.runner.cleanup()
            

class ResourceMonitor:
    """Monitor and manage system resources"""
    
    def __init__(self, memory_limit_mb: int = 4096):
        self.memory_limit_mb = memory_limit_mb
        self.collector = MetricsCollector()
        self.server = MetricsServer(self.collector)
        self.monitoring = False
        self.monitor_thread = None
        
    def start_monitoring(self):
        """Start resource monitoring"""
        self.monitoring = True
        
        # Start metrics collection thread
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        
        # Start metrics server
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(self.server.start())
        
    def stop_monitoring(self):
        """Stop resource monitoring"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
            
        # Stop server
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(self.server.stop())
        
    def _monitor_loop(self):
        """Background monitoring loop"""
        while self.monitoring:
            try:
                # Collect metrics
                metrics = self.collector.collect_metrics()
                system_metrics = self.collector.collect_system_metrics()
                
                # Check memory limit
                if metrics.memory_mb > self.memory_limit_mb:
                    logger.warning(
                        f"Memory usage ({metrics.memory_mb:.0f}MB) "
                        f"exceeds limit ({self.memory_limit_mb}MB)"
                    )
                    self._handle_memory_limit()
                    
                # Sleep before next collection
                time.sleep(1)
                
            except Exception as e:
                logger.error(f"Error in monitor loop: {e}")
                time.sleep(5)
                
    def _handle_memory_limit(self):
        """Handle memory limit exceeded"""
        # Trigger garbage collection
        import gc
        gc.collect()
        
        # Log memory usage after GC
        metrics = self.collector.collect_metrics()
        logger.info(f"Memory after GC: {metrics.memory_mb:.0f}MB")


# Singleton instance
_monitor_instance = None


def get_resource_monitor() -> ResourceMonitor:
    """Get or create resource monitor instance"""
    global _monitor_instance
    if _monitor_instance is None:
        _monitor_instance = ResourceMonitor()
    return _monitor_instance