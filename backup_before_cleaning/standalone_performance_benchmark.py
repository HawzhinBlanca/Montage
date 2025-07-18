"""Standalone performance benchmarking suite for Phase 3 Task 17"""

import os
import time
import logging
import subprocess
import tempfile
import json
import psutil
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Any, List, Tuple
from datetime import datetime
import statistics

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StandalonePerformanceBenchmark:
    """Performance benchmarking without external dependencies"""
    
    def __init__(self):
        self.results = {
            'start_time': datetime.utcnow().isoformat(),
            'system_info': self._get_system_info(),
            'benchmarks': {},
            'summary': {}
        }
        
    def run_full_benchmark_suite(self) -> Dict[str, Any]:
        """Run complete performance benchmark suite"""
        logger.info("🚀 Starting Standalone Performance Benchmark Suite")
        logger.info("=" * 60)
        
        benchmarks = [
            ('ffmpeg_performance', self._benchmark_ffmpeg_performance),
            ('concurrent_ffmpeg', self._benchmark_concurrent_ffmpeg),
            ('memory_usage_patterns', self._benchmark_memory_usage_patterns),
            ('io_throughput', self._benchmark_io_throughput),
            ('cpu_utilization', self._benchmark_cpu_utilization),
            ('audio_processing_speed', self._benchmark_audio_processing_speed),
            ('video_encoding_efficiency', self._benchmark_video_encoding_efficiency),
            ('real_world_scenarios', self._benchmark_real_world_scenarios)
        ]
        
        for benchmark_name, benchmark_func in benchmarks:
            logger.info(f"\n📊 Running {benchmark_name} benchmark...")
            try:
                start_time = time.time()
                result = benchmark_func()
                duration = time.time() - start_time
                
                self.results['benchmarks'][benchmark_name] = {
                    'result': result,
                    'duration': duration,
                    'status': 'completed'
                }
                
                logger.info(f"✅ {benchmark_name} completed in {duration:.1f}s")
                
            except Exception as e:
                logger.error(f"❌ {benchmark_name} failed: {e}")
                self.results['benchmarks'][benchmark_name] = {
                    'status': 'failed',
                    'error': str(e)
                }
        
        # Generate summary and reports
        self._generate_summary()
        self._save_results()
        
        return self.results
    
    def _get_system_info(self) -> Dict[str, Any]:
        """Collect system information"""
        return {
            'cpu_count': psutil.cpu_count(),
            'cpu_count_logical': psutil.cpu_count(logical=True),
            'memory_total_gb': psutil.virtual_memory().total / (1024**3),
            'memory_available_gb': psutil.virtual_memory().available / (1024**3),
            'disk_usage': {
                'total_gb': psutil.disk_usage('/').total / (1024**3),
                'available_gb': psutil.disk_usage('/').free / (1024**3)
            },
            'platform': os.uname().sysname,
            'ffmpeg_version': self._get_ffmpeg_version()
        }
    
    def _get_ffmpeg_version(self) -> str:
        """Get FFmpeg version information"""
        try:
            result = subprocess.run(['ffmpeg', '-version'], 
                                  capture_output=True, text=True)
            return result.stdout.split('\n')[0]
        except Exception:
            return "Unknown"
    
    def _benchmark_ffmpeg_performance(self) -> Dict[str, Any]:
        """Benchmark FFmpeg processing performance across different video durations"""
        durations = [30, 60, 300, 600, 1800, 2700]  # 30s to 45min
        results = []
        
        for duration in durations:
            logger.info(f"  Testing {duration}s video processing...")
            
            # Create test video
            input_path = self._create_test_video(duration)
            output_path = f"/tmp/perf_output_{duration}.mp4"
            
            try:
                # Measure pure FFmpeg processing time
                start_time = time.time()
                
                cmd = [
                    'ffmpeg', '-y', '-i', input_path,
                    '-vf', 'crop=607:1080:656:0',  # 9:16 crop from 16:9
                    '-af', 'loudnorm=I=-16:TP=-1.5:LRA=7',
                    '-c:v', 'libx264', '-preset', 'medium', '-crf', '23',
                    '-c:a', 'aac', '-b:a', '128k',
                    output_path
                ]
                
                result = subprocess.run(cmd, capture_output=True, text=True, check=True)
                processing_time = time.time() - start_time
                
                # Get output file info
                output_size = os.path.getsize(output_path) / (1024**2)  # MB
                
                # Calculate metrics
                ratio = processing_time / duration
                throughput = duration / processing_time  # seconds of video per second
                
                results.append({
                    'duration': duration,
                    'processing_time': processing_time,
                    'ratio': ratio,
                    'throughput': throughput,
                    'output_size_mb': output_size,
                    'compression_ratio': os.path.getsize(input_path) / os.path.getsize(output_path),
                    'memory_peak': self._get_memory_usage()
                })
                
                logger.info(f"    {duration}s → {processing_time:.1f}s (ratio: {ratio:.2f}x)")
                
            except subprocess.CalledProcessError as e:
                logger.warning(f"    FFmpeg failed for {duration}s video: {e}")
                results.append({
                    'duration': duration,
                    'error': str(e),
                    'failed': True
                })
                
            finally:
                # Cleanup
                for path in [input_path, output_path]:
                    if os.path.exists(path):
                        os.remove(path)
        
        valid_results = [r for r in results if not r.get('failed', False)]
        
        return {
            'individual_results': results,
            'valid_results_count': len(valid_results),
            'average_ratio': statistics.mean([r['ratio'] for r in valid_results]) if valid_results else 0,
            'max_ratio': max([r['ratio'] for r in valid_results]) if valid_results else 0,
            'min_ratio': min([r['ratio'] for r in valid_results]) if valid_results else 0,
            'performance_target_met': all(r['ratio'] <= 1.2 for r in valid_results) if valid_results else False,
            'total_throughput': sum([r['throughput'] for r in valid_results]) if valid_results else 0
        }
    
    def _benchmark_concurrent_ffmpeg(self) -> Dict[str, Any]:
        """Benchmark concurrent FFmpeg processing"""
        cpu_count = psutil.cpu_count()
        concurrency_levels = [1, 2, min(4, cpu_count), min(8, cpu_count)]
        results = []
        
        for concurrency in concurrency_levels:
            logger.info(f"  Testing {concurrency} concurrent FFmpeg processes...")
            
            # Create test videos
            video_paths = []
            output_paths = []
            for i in range(concurrency):
                input_path = self._create_test_video(120)  # 2-minute videos
                output_path = f"/tmp/concurrent_{concurrency}_{i}.mp4"
                video_paths.append(input_path)
                output_paths.append(output_path)
            
            try:
                start_time = time.time()
                
                def process_video(input_path, output_path):
                    cmd = [
                        'ffmpeg', '-y', '-i', input_path,
                        '-vf', 'scale=1280:720',  # Simple scaling
                        '-c:v', 'libx264', '-preset', 'ultrafast',
                        '-c:a', 'copy',
                        output_path
                    ]
                    return subprocess.run(cmd, capture_output=True, text=True, check=True)
                
                # Process concurrently
                with ThreadPoolExecutor(max_workers=concurrency) as executor:
                    futures = [
                        executor.submit(process_video, input_path, output_path)
                        for input_path, output_path in zip(video_paths, output_paths)
                    ]
                    
                    completed = 0
                    for future in as_completed(futures):
                        completed += 1
                        logger.info(f"    Job {completed}/{concurrency} completed")
                
                total_time = time.time() - start_time
                avg_time_per_job = total_time / concurrency
                
                results.append({
                    'concurrency': concurrency,
                    'total_time': total_time,
                    'avg_time_per_job': avg_time_per_job,
                    'throughput_jobs_per_sec': concurrency / total_time,
                    'memory_peak': self._get_memory_usage(),
                    'cpu_peak': self._get_cpu_usage()
                })
                
                logger.info(f"    {concurrency} jobs → {total_time:.1f}s total, {avg_time_per_job:.1f}s average")
                
            except Exception as e:
                logger.warning(f"    Concurrent processing failed: {e}")
                results.append({
                    'concurrency': concurrency,
                    'error': str(e),
                    'failed': True
                })
                
            finally:
                # Cleanup
                for path in video_paths + output_paths:
                    if os.path.exists(path):
                        os.remove(path)
        
        valid_results = [r for r in results if not r.get('failed', False)]
        
        return {
            'results': results,
            'valid_results_count': len(valid_results),
            'optimal_concurrency': self._find_optimal_concurrency(valid_results),
            'scaling_efficiency': self._calculate_scaling_efficiency(valid_results)
        }
    
    def _benchmark_memory_usage_patterns(self) -> Dict[str, Any]:
        """Benchmark memory usage patterns during video processing"""
        video_configs = [
            {'duration': 300, 'resolution': (1280, 720)},   # 5min 720p
            {'duration': 300, 'resolution': (1920, 1080)},  # 5min 1080p
            {'duration': 600, 'resolution': (1920, 1080)},  # 10min 1080p
        ]
        results = []
        
        for config in video_configs:
            duration = config['duration']
            width, height = config['resolution']
            
            logger.info(f"  Testing memory usage: {duration}s {width}x{height}...")
            
            # Monitor memory during processing
            memory_samples = []
            
            def memory_monitor():
                while not stop_monitoring:
                    memory_samples.append(psutil.virtual_memory().used / (1024**2))  # MB
                    time.sleep(0.5)
            
            input_path = self._create_test_video(duration, resolution=(width, height))
            output_path = f"/tmp/memory_test_{width}x{height}_{duration}.mp4"
            
            try:
                stop_monitoring = False
                monitor_thread = threading.Thread(target=memory_monitor)
                monitor_thread.start()
                
                # Process video
                cmd = [
                    'ffmpeg', '-y', '-i', input_path,
                    '-vf', 'scale=1280:720',
                    '-c:v', 'libx264', '-preset', 'medium',
                    '-c:a', 'aac',
                    output_path
                ]
                
                start_time = time.time()
                subprocess.run(cmd, check=True, capture_output=True)
                processing_time = time.time() - start_time
                
                stop_monitoring = True
                monitor_thread.join()
                
                if memory_samples:
                    results.append({
                        'config': config,
                        'processing_time': processing_time,
                        'memory_baseline': memory_samples[0],
                        'memory_peak': max(memory_samples),
                        'memory_avg': statistics.mean(memory_samples),
                        'memory_increase': max(memory_samples) - memory_samples[0],
                        'samples_count': len(memory_samples)
                    })
                    
                    logger.info(f"    Peak memory: {max(memory_samples):.0f}MB, "
                              f"increase: {max(memory_samples) - memory_samples[0]:.0f}MB")
                
            except Exception as e:
                logger.warning(f"    Memory test failed: {e}")
                
            finally:
                for path in [input_path, output_path]:
                    if os.path.exists(path):
                        os.remove(path)
        
        return {
            'resolution_tests': results,
            'memory_efficiency': self._calculate_memory_efficiency(results)
        }
    
    def _benchmark_io_throughput(self) -> Dict[str, Any]:
        """Benchmark I/O throughput during video operations"""
        logger.info("  Testing I/O throughput...")
        
        # Create test files of different sizes
        test_sizes = [100, 500, 1000]  # MB
        results = []
        
        for size_mb in test_sizes:
            logger.info(f"    Testing {size_mb}MB file I/O...")
            
            input_path = f"/tmp/io_test_{size_mb}mb.dat"
            output_path = f"/tmp/io_test_copy_{size_mb}mb.dat"
            
            try:
                # Create test file
                with open(input_path, 'wb') as f:
                    data = os.urandom(1024 * 1024)  # 1MB chunk
                    for _ in range(size_mb):
                        f.write(data)
                
                # Test read throughput
                start_time = time.time()
                with open(input_path, 'rb') as f:
                    data = f.read()
                read_time = time.time() - start_time
                read_throughput = size_mb / read_time
                
                # Test write throughput
                start_time = time.time()
                with open(output_path, 'wb') as f:
                    f.write(data)
                write_time = time.time() - start_time
                write_throughput = size_mb / write_time
                
                results.append({
                    'size_mb': size_mb,
                    'read_time': read_time,
                    'write_time': write_time,
                    'read_throughput_mbps': read_throughput,
                    'write_throughput_mbps': write_throughput
                })
                
                logger.info(f"      Read: {read_throughput:.1f} MB/s, Write: {write_throughput:.1f} MB/s")
                
            finally:
                for path in [input_path, output_path]:
                    if os.path.exists(path):
                        os.remove(path)
        
        return {
            'io_tests': results,
            'avg_read_throughput': statistics.mean([r['read_throughput_mbps'] for r in results]),
            'avg_write_throughput': statistics.mean([r['write_throughput_mbps'] for r in results]),
            'io_efficiency_score': min(statistics.mean([r['read_throughput_mbps'] for r in results]) / 100, 1.0)
        }
    
    def _benchmark_cpu_utilization(self) -> Dict[str, Any]:
        """Benchmark CPU utilization during video processing"""
        logger.info("  Testing CPU utilization patterns...")
        
        cpu_samples = []
        
        def cpu_monitor():
            while not stop_monitoring:
                cpu_samples.append(psutil.cpu_percent(interval=None))
                time.sleep(0.5)
        
        # Create test video
        input_path = self._create_test_video(300)  # 5 minutes
        output_path = "/tmp/cpu_test.mp4"
        
        try:
            stop_monitoring = False
            monitor_thread = threading.Thread(target=cpu_monitor)
            monitor_thread.start()
            
            # Process video with intensive settings
            cmd = [
                'ffmpeg', '-y', '-i', input_path,
                '-vf', 'scale=1920:1080,unsharp=5:5:1.0',
                '-c:v', 'libx264', '-preset', 'slow', '-crf', '18',
                '-c:a', 'aac',
                output_path
            ]
            
            start_time = time.time()
            subprocess.run(cmd, check=True, capture_output=True)
            processing_time = time.time() - start_time
            
            stop_monitoring = True
            monitor_thread.join()
            
            return {
                'processing_time': processing_time,
                'cpu_utilization': {
                    'average': statistics.mean(cpu_samples),
                    'peak': max(cpu_samples),
                    'minimum': min(cpu_samples),
                    'samples': len(cpu_samples)
                },
                'efficiency_score': min(statistics.mean(cpu_samples) / 80.0, 1.0)  # Target 80% utilization
            }
            
        finally:
            for path in [input_path, output_path]:
                if os.path.exists(path):
                    os.remove(path)
    
    def _benchmark_audio_processing_speed(self) -> Dict[str, Any]:
        """Benchmark audio processing speed"""
        logger.info("  Testing audio processing speed...")
        
        audio_durations = [60, 300, 1800]  # 1min, 5min, 30min
        results = []
        
        for duration in audio_durations:
            logger.info(f"    Testing {duration}s audio processing...")
            
            input_path = self._create_test_audio(duration)
            output_path = f"/tmp/audio_test_{duration}.wav"
            
            try:
                # Test loudnorm processing
                start_time = time.time()
                cmd = [
                    'ffmpeg', '-y', '-i', input_path,
                    '-af', 'loudnorm=I=-16:TP=-1.5:LRA=7',
                    output_path
                ]
                
                subprocess.run(cmd, check=True, capture_output=True)
                processing_time = time.time() - start_time
                
                ratio = processing_time / duration
                
                results.append({
                    'duration': duration,
                    'processing_time': processing_time,
                    'ratio': ratio
                })
                
                logger.info(f"      {duration}s → {processing_time:.1f}s (ratio: {ratio:.2f}x)")
                
            finally:
                for path in [input_path, output_path]:
                    if os.path.exists(path):
                        os.remove(path)
        
        return {
            'audio_tests': results,
            'avg_ratio': statistics.mean([r['ratio'] for r in results]),
            'audio_efficiency': all(r['ratio'] < 0.1 for r in results)  # Audio should be very fast
        }
    
    def _benchmark_video_encoding_efficiency(self) -> Dict[str, Any]:
        """Benchmark video encoding efficiency with different presets"""
        logger.info("  Testing video encoding efficiency...")
        
        presets = ['ultrafast', 'fast', 'medium', 'slow']
        results = []
        
        input_path = self._create_test_video(180)  # 3-minute test video
        
        try:
            for preset in presets:
                logger.info(f"    Testing {preset} preset...")
                
                output_path = f"/tmp/encode_test_{preset}.mp4"
                
                try:
                    start_time = time.time()
                    cmd = [
                        'ffmpeg', '-y', '-i', input_path,
                        '-c:v', 'libx264', '-preset', preset, '-crf', '23',
                        '-c:a', 'aac',
                        output_path
                    ]
                    
                    subprocess.run(cmd, check=True, capture_output=True)
                    encoding_time = time.time() - start_time
                    
                    # Get file size
                    output_size = os.path.getsize(output_path) / (1024**2)  # MB
                    
                    results.append({
                        'preset': preset,
                        'encoding_time': encoding_time,
                        'output_size_mb': output_size,
                        'encoding_speed': 180 / encoding_time,  # fps equivalent
                        'size_efficiency': output_size / encoding_time  # MB per second
                    })
                    
                    logger.info(f"      {preset}: {encoding_time:.1f}s, {output_size:.1f}MB")
                    
                finally:
                    if os.path.exists(output_path):
                        os.remove(output_path)
                        
        finally:
            if os.path.exists(input_path):
                os.remove(input_path)
        
        return {
            'preset_tests': results,
            'fastest_preset': min(results, key=lambda x: x['encoding_time'])['preset'],
            'most_efficient': min(results, key=lambda x: x['output_size_mb'])['preset']
        }
    
    def _benchmark_real_world_scenarios(self) -> Dict[str, Any]:
        """Benchmark real-world video processing scenarios"""
        logger.info("  Testing real-world scenarios...")
        
        scenarios = [
            {
                'name': 'podcast_highlights',
                'duration': 2700,  # 45 minutes
                'operations': ['crop=607:1080:656:0', 'loudnorm=I=-16:TP=-1.5:LRA=7']
            },
            {
                'name': 'social_media_short',
                'duration': 30,
                'operations': ['scale=1080:1920', 'loudnorm=I=-14:TP=-1:LRA=8']
            },
            {
                'name': 'webinar_clip',
                'duration': 600,  # 10 minutes
                'operations': ['scale=1280:720', 'loudnorm=I=-16:TP=-2:LRA=6']
            }
        ]
        
        results = []
        
        for scenario in scenarios:
            logger.info(f"    Testing scenario: {scenario['name']}...")
            
            input_path = self._create_test_video(scenario['duration'])
            output_path = f"/tmp/scenario_{scenario['name']}.mp4"
            
            try:
                # Build FFmpeg command
                vf_filters = [op for op in scenario['operations'] if not op.startswith('loud')]
                af_filters = [op for op in scenario['operations'] if op.startswith('loud')]
                
                cmd = ['ffmpeg', '-y', '-i', input_path]
                if vf_filters:
                    cmd.extend(['-vf', ','.join(vf_filters)])
                if af_filters:
                    cmd.extend(['-af', ','.join(af_filters)])
                cmd.extend(['-c:v', 'libx264', '-preset', 'medium', '-c:a', 'aac', output_path])
                
                start_time = time.time()
                subprocess.run(cmd, check=True, capture_output=True)
                processing_time = time.time() - start_time
                
                ratio = processing_time / scenario['duration']
                
                results.append({
                    'scenario': scenario['name'],
                    'duration': scenario['duration'],
                    'processing_time': processing_time,
                    'ratio': ratio,
                    'meets_target': ratio <= 1.2
                })
                
                logger.info(f"      {scenario['name']}: {processing_time:.1f}s (ratio: {ratio:.2f}x)")
                
            except Exception as e:
                logger.warning(f"      {scenario['name']} failed: {e}")
                results.append({
                    'scenario': scenario['name'],
                    'error': str(e),
                    'failed': True
                })
                
            finally:
                for path in [input_path, output_path]:
                    if os.path.exists(path):
                        os.remove(path)
        
        valid_results = [r for r in results if not r.get('failed', False)]
        
        return {
            'scenario_tests': results,
            'scenarios_passed': len(valid_results),
            'all_scenarios_meet_target': all(r.get('meets_target', False) for r in valid_results),
            'avg_ratio': statistics.mean([r['ratio'] for r in valid_results]) if valid_results else 0
        }
    
    def _create_test_video(self, duration: int, resolution: Tuple[int, int] = (1920, 1080)) -> str:
        """Create a test video with specified duration and resolution"""
        output_path = tempfile.mktemp(suffix='.mp4')
        width, height = resolution
        
        cmd = [
            'ffmpeg', '-y',
            '-f', 'lavfi',
            '-i', f'testsrc2=duration={duration}:size={width}x{height}:rate=30',
            '-f', 'lavfi',
            '-i', f'sine=frequency=440:duration={duration}',
            '-c:v', 'libx264',
            '-preset', 'ultrafast',
            '-c:a', 'aac',
            output_path
        ]
        
        subprocess.run(cmd, check=True, capture_output=True)
        return output_path
    
    def _create_test_audio(self, duration: int) -> str:
        """Create a test audio file"""
        output_path = tempfile.mktemp(suffix='.wav')
        
        cmd = [
            'ffmpeg', '-y',
            '-f', 'lavfi',
            '-i', f'sine=frequency=440:duration={duration}',
            '-c:a', 'pcm_s16le',
            output_path
        ]
        
        subprocess.run(cmd, check=True, capture_output=True)
        return output_path
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        return psutil.virtual_memory().used / (1024**2)
    
    def _get_cpu_usage(self) -> float:
        """Get current CPU usage percentage"""
        return psutil.cpu_percent(interval=1)
    
    def _find_optimal_concurrency(self, results: List[Dict]) -> int:
        """Find optimal concurrency level based on throughput"""
        if not results:
            return 1
            
        best_throughput = 0
        optimal_level = 1
        
        for result in results:
            throughput = result.get('throughput_jobs_per_sec', 0)
            if throughput > best_throughput:
                best_throughput = throughput
                optimal_level = result['concurrency']
        
        return optimal_level
    
    def _calculate_scaling_efficiency(self, results: List[Dict]) -> float:
        """Calculate how efficiently the system scales with concurrency"""
        if len(results) < 2:
            return 1.0
        
        baseline = results[0].get('throughput_jobs_per_sec', 1)
        max_throughput = max(r.get('throughput_jobs_per_sec', 0) for r in results)
        max_concurrency = max(r['concurrency'] for r in results)
        
        # Ideal scaling would be linear
        ideal_max_throughput = baseline * max_concurrency
        efficiency = max_throughput / ideal_max_throughput if ideal_max_throughput > 0 else 0
        
        return min(efficiency, 1.0)  # Cap at 100%
    
    def _calculate_memory_efficiency(self, results: List[Dict]) -> Dict[str, float]:
        """Calculate memory usage efficiency"""
        if not results:
            return {'avg_mb_per_second': 0, 'memory_scaling_factor': 1.0}
            
        memory_per_second = []
        
        for result in results:
            config = result['config']
            duration = config['duration']
            memory_increase = result['memory_increase']
            
            memory_per_second.append(memory_increase / duration)
        
        return {
            'avg_mb_per_second': statistics.mean(memory_per_second),
            'memory_scaling_factor': max(memory_per_second) / min(memory_per_second) if min(memory_per_second) > 0 else 1.0
        }
    
    def _generate_summary(self):
        """Generate performance benchmark summary"""
        summary = {
            'total_benchmarks': len(self.results['benchmarks']),
            'successful_benchmarks': sum(1 for b in self.results['benchmarks'].values() 
                                       if b.get('status') == 'completed'),
            'failed_benchmarks': sum(1 for b in self.results['benchmarks'].values() 
                                   if b.get('status') == 'failed'),
            'total_duration': sum(b.get('duration', 0) for b in self.results['benchmarks'].values()),
            'performance_grade': self._calculate_performance_grade()
        }
        
        # Extract key metrics
        ffmpeg_perf = self.results['benchmarks'].get('ffmpeg_performance', {}).get('result', {})
        concurrent = self.results['benchmarks'].get('concurrent_ffmpeg', {}).get('result', {})
        real_world = self.results['benchmarks'].get('real_world_scenarios', {}).get('result', {})
        
        if ffmpeg_perf:
            summary['avg_processing_ratio'] = ffmpeg_perf.get('average_ratio', 0)
            summary['performance_target_met'] = ffmpeg_perf.get('performance_target_met', False)
        
        if concurrent:
            summary['optimal_concurrency'] = concurrent.get('optimal_concurrency', 1)
            summary['scaling_efficiency'] = concurrent.get('scaling_efficiency', 0)
        
        if real_world:
            summary['real_world_scenarios_passed'] = real_world.get('scenarios_passed', 0)
            summary['all_scenarios_meet_target'] = real_world.get('all_scenarios_meet_target', False)
        
        self.results['summary'] = summary
    
    def _calculate_performance_grade(self) -> str:
        """Calculate overall performance grade A-F"""
        completed = sum(1 for b in self.results['benchmarks'].values() 
                       if b.get('status') == 'completed')
        total = len(self.results['benchmarks'])
        
        completion_rate = completed / total if total > 0 else 0
        
        if completion_rate >= 0.9:
            return 'A'
        elif completion_rate >= 0.8:
            return 'B'
        elif completion_rate >= 0.7:
            return 'C'
        elif completion_rate >= 0.6:
            return 'D'
        else:
            return 'F'
    
    def _save_results(self):
        """Save performance results to file"""
        timestamp = int(time.time())
        report_path = f"performance_benchmark_report_{timestamp}.json"
        
        with open(report_path, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        logger.info(f"Performance report saved to: {report_path}")


def main():
    """Run standalone performance benchmark suite"""
    benchmark = StandalonePerformanceBenchmark()
    
    try:
        results = benchmark.run_full_benchmark_suite()
        
        # Print summary
        summary = results['summary']
        system_info = results['system_info']
        
        logger.info("\n" + "=" * 60)
        logger.info("📊 PERFORMANCE BENCHMARK SUMMARY")
        logger.info("=" * 60)
        logger.info(f"System: {system_info['cpu_count']} CPU cores, "
                   f"{system_info['memory_total_gb']:.1f}GB RAM")
        logger.info(f"FFmpeg: {system_info['ffmpeg_version']}")
        logger.info(f"Total Benchmarks: {summary['total_benchmarks']}")
        logger.info(f"Successful: {summary['successful_benchmarks']}")
        logger.info(f"Failed: {summary['failed_benchmarks']}")
        logger.info(f"Total Duration: {summary['total_duration']:.1f}s")
        logger.info(f"Performance Grade: {summary['performance_grade']}")
        
        if 'avg_processing_ratio' in summary:
            logger.info(f"Avg Processing Ratio: {summary['avg_processing_ratio']:.2f}x")
            logger.info(f"Performance Target Met: {summary['performance_target_met']}")
        
        if 'optimal_concurrency' in summary:
            logger.info(f"Optimal Concurrency: {summary['optimal_concurrency']}")
            logger.info(f"Scaling Efficiency: {summary['scaling_efficiency']:.1%}")
        
        if 'real_world_scenarios_passed' in summary:
            logger.info(f"Real-world Scenarios Passed: {summary['real_world_scenarios_passed']}")
            logger.info(f"All Scenarios Meet Target: {summary['all_scenarios_meet_target']}")
        
        # Final assessment
        if summary['performance_grade'] in ['A', 'B'] and summary.get('performance_target_met', False):
            logger.info("\n✅ SYSTEM PERFORMANCE: EXCELLENT")
            logger.info("Ready for production deployment with high performance.")
        elif summary['performance_grade'] in ['C', 'D']:
            logger.info("\n⚠️ SYSTEM PERFORMANCE: ADEQUATE")
            logger.info("System meets basic requirements but could be optimized.")
        else:
            logger.info("\n❌ SYSTEM PERFORMANCE: NEEDS IMPROVEMENT")
            logger.info("System requires optimization before production deployment.")
        
        return 0 if summary['failed_benchmarks'] == 0 else 1
        
    except Exception as e:
        logger.error(f"Benchmark suite failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main())