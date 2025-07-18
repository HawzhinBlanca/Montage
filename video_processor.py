"""FIFO-based video processing pipeline using FFmpeg"""

import os
import subprocess
import tempfile
import threading
import logging
import time
import signal
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from contextlib import contextmanager
import uuid
from config import Config
from metrics import metrics, track_ffmpeg_process_start, track_ffmpeg_process_end

logger = logging.getLogger(__name__)


class VideoProcessingError(Exception):
    """Base exception for video processing errors"""
    pass


class FFmpegError(VideoProcessingError):
    """FFmpeg command failed"""
    pass


@dataclass
class VideoSegment:
    """Represents a video segment"""
    start_time: float
    end_time: float
    input_file: str
    segment_id: str = None
    
    def __post_init__(self):
        if not self.segment_id:
            self.segment_id = str(uuid.uuid4())[:8]
    
    @property
    def duration(self):
        return self.end_time - self.start_time


class FIFOManager:
    """Manages FIFO (named pipe) creation and cleanup"""
    
    def __init__(self, temp_dir: Optional[str] = None):
        self.temp_dir = temp_dir or Config.TEMP_DIR
        os.makedirs(self.temp_dir, exist_ok=True)
        self.fifos = []
        self._cleanup_registered = False
        
    def create_fifo(self, name_suffix: str = "") -> str:
        """Create a named pipe and return its path"""
        fifo_name = f"fifo_{uuid.uuid4().hex[:8]}{name_suffix}"
        fifo_path = os.path.join(self.temp_dir, fifo_name)
        
        try:
            os.mkfifo(fifo_path)
            self.fifos.append(fifo_path)
            
            # Register cleanup on first FIFO
            if not self._cleanup_registered:
                signal.signal(signal.SIGINT, self._signal_handler)
                signal.signal(signal.SIGTERM, self._signal_handler)
                self._cleanup_registered = True
                
            logger.debug(f"Created FIFO: {fifo_path}")
            return fifo_path
            
        except OSError as e:
            raise VideoProcessingError(f"Failed to create FIFO: {e}")
    
    def cleanup(self):
        """Remove all created FIFOs"""
        for fifo in self.fifos:
            try:
                if os.path.exists(fifo):
                    os.unlink(fifo)
                    logger.debug(f"Removed FIFO: {fifo}")
            except OSError as e:
                logger.warning(f"Failed to remove FIFO {fifo}: {e}")
        
        self.fifos.clear()
    
    def _signal_handler(self, signum, frame):
        """Clean up FIFOs on signal"""
        logger.info("Received signal, cleaning up FIFOs...")
        self.cleanup()
        
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()


class FFmpegPipeline:
    """Manages FFmpeg processes in a pipeline configuration"""
    
    def __init__(self):
        self.processes = []
        self.threads = []
        self.fifo_manager = FIFOManager()
        
    def add_process(self, cmd: List[str], name: str = "ffmpeg") -> subprocess.Popen:
        """Add an FFmpeg process to the pipeline"""
        try:
            logger.info(f"Starting {name} process: {' '.join(cmd[:10])}...")
            
            # Track FFmpeg process
            track_ffmpeg_process_start()
            
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=0  # Unbuffered for real-time processing
            )
            
            self.processes.append((name, process))
            
            # Start thread to monitor stderr
            error_thread = threading.Thread(
                target=self._monitor_stderr,
                args=(process, name),
                daemon=True
            )
            error_thread.start()
            self.threads.append(error_thread)
            
            return process
            
        except Exception as e:
            track_ffmpeg_process_end()
            raise FFmpegError(f"Failed to start {name}: {e}")
    
    def _monitor_stderr(self, process: subprocess.Popen, name: str):
        """Monitor process stderr for errors and progress"""
        try:
            for line in iter(process.stderr.readline, b''):
                if not line:
                    break
                    
                line_str = line.decode('utf-8', errors='ignore').strip()
                
                # Log errors
                if 'error' in line_str.lower():
                    logger.error(f"{name}: {line_str}")
                # Log warnings
                elif 'warning' in line_str.lower():
                    logger.warning(f"{name}: {line_str}")
                # Progress updates
                elif 'frame=' in line_str or 'time=' in line_str:
                    logger.debug(f"{name} progress: {line_str}")
                    
        except Exception as e:
            logger.error(f"Error monitoring {name} stderr: {e}")
    
    def wait_all(self, timeout: Optional[int] = None) -> Dict[str, int]:
        """Wait for all processes to complete"""
        results = {}
        start_time = time.time()
        
        for name, process in self.processes:
            try:
                # Calculate remaining timeout
                if timeout:
                    elapsed = time.time() - start_time
                    remaining_timeout = max(1, timeout - int(elapsed))
                else:
                    remaining_timeout = None
                
                # Wait for process
                return_code = process.wait(timeout=remaining_timeout)
                results[name] = return_code
                
                if return_code != 0:
                    # Get error output
                    _, stderr = process.communicate(timeout=5)
                    logger.error(f"{name} failed with code {return_code}: {stderr.decode('utf-8', errors='ignore')}")
                    
            except subprocess.TimeoutExpired:
                logger.error(f"{name} timed out, terminating...")
                process.terminate()
                results[name] = -1
            except Exception as e:
                logger.error(f"Error waiting for {name}: {e}")
                results[name] = -2
            finally:
                track_ffmpeg_process_end()
        
        return results
    
    def cleanup(self):
        """Clean up all processes and resources"""
        # Terminate any running processes
        for name, process in self.processes:
            if process.poll() is None:
                logger.warning(f"Terminating {name} process")
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    logger.error(f"Force killing {name} process")
                    process.kill()
                
                track_ffmpeg_process_end()
        
        # Clean up FIFOs
        self.fifo_manager.cleanup()
        
        self.processes.clear()
        self.threads.clear()
    
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()


class VideoEditor:
    """High-performance video editor using FIFO pipelines"""
    
    def __init__(self):
        self.ffmpeg_path = Config.FFMPEG_PATH
        self.temp_dir = Config.TEMP_DIR
        
    def extract_segments_parallel(self, input_file: str, segments: List[VideoSegment]) -> List[str]:
        """Extract multiple segments in parallel using FIFOs"""
        with FFmpegPipeline() as pipeline:
            segment_fifos = []
            
            for segment in segments:
                # Create FIFO for segment
                fifo_path = pipeline.fifo_manager.create_fifo(f"_seg_{segment.segment_id}")
                segment_fifos.append(fifo_path)
                
                # Build extraction command
                cmd = [
                    self.ffmpeg_path,
                    '-ss', str(segment.start_time),
                    '-t', str(segment.duration),
                    '-i', input_file,
                    '-c', 'copy',  # No re-encoding for speed
                    '-f', 'mpegts',  # Use MPEG-TS for streaming
                    '-y',
                    fifo_path
                ]
                
                # Start extraction process
                pipeline.add_process(cmd, f"extract_{segment.segment_id}")
            
            # Return FIFO paths immediately - processes run in background
            return segment_fifos
    
    def concatenate_segments_fifo(self, segment_fifos: List[str], output_file: str,
                                  video_codec: str = 'libx264', audio_codec: str = 'aac') -> None:
        """Concatenate segments from FIFOs without intermediate files"""
        with FFmpegPipeline() as pipeline:
            # Create concat demuxer input
            concat_list = self._create_concat_list(segment_fifos)
            
            # Build concatenation command
            cmd = [
                self.ffmpeg_path,
                '-f', 'concat',
                '-safe', '0',
                '-i', concat_list,
                '-c:v', video_codec,
                '-c:a', audio_codec,
                '-preset', 'fast',  # Balance speed/quality
                '-movflags', '+faststart',  # Web optimization
                '-y',
                output_file
            ]
            
            # Start concatenation
            concat_process = pipeline.add_process(cmd, "concatenate")
            
            # Wait for completion
            results = pipeline.wait_all(timeout=1800)  # 30 minute timeout
            
            if results.get("concatenate", -1) != 0:
                raise FFmpegError("Concatenation failed")
            
            # Clean up concat list
            try:
                os.unlink(concat_list)
            except:
                pass
    
    def _create_concat_list(self, fifos: List[str]) -> str:
        """Create concat demuxer list file"""
        concat_file = os.path.join(self.temp_dir, f"concat_{uuid.uuid4().hex[:8]}.txt")
        
        with open(concat_file, 'w') as f:
            for fifo in fifos:
                f.write(f"file '{fifo}'\n")
        
        return concat_file
    
    def process_with_filter_fifo(self, input_file: str, output_file: str,
                                 filter_complex: str, video_codec: str = 'libx264',
                                 audio_codec: str = 'aac') -> None:
        """Process video through filter using FIFO pipeline"""
        with FFmpegPipeline() as pipeline:
            # Create intermediate FIFO
            filtered_fifo = pipeline.fifo_manager.create_fifo("_filtered")
            
            # First pass: apply filters to FIFO
            filter_cmd = [
                self.ffmpeg_path,
                '-i', input_file,
                '-filter_complex', filter_complex,
                '-f', 'mpegts',
                '-c:v', 'libx264',
                '-preset', 'ultrafast',  # Fast for intermediate
                '-c:a', 'pcm_s16le',  # Lossless audio
                '-y',
                filtered_fifo
            ]
            
            pipeline.add_process(filter_cmd, "filter")
            
            # Second pass: encode from FIFO
            encode_cmd = [
                self.ffmpeg_path,
                '-i', filtered_fifo,
                '-c:v', video_codec,
                '-preset', 'fast',
                '-c:a', audio_codec,
                '-movflags', '+faststart',
                '-y',
                output_file
            ]
            
            pipeline.add_process(encode_cmd, "encode")
            
            # Wait for completion
            results = pipeline.wait_all(timeout=1800)
            
            if any(code != 0 for code in results.values()):
                raise FFmpegError(f"Processing failed: {results}")
    
    def apply_transitions_fifo(self, input_file: str, output_file: str,
                               transition_points: List[float], transition_duration: float = 0.5) -> None:
        """Apply transitions at specified points using xfade filter"""
        if not transition_points:
            # No transitions, just copy
            subprocess.run([
                self.ffmpeg_path,
                '-i', input_file,
                '-c', 'copy',
                '-y', output_file
            ], check=True)
            return
        
        # Build xfade filter chain
        filter_parts = []
        for i, point in enumerate(transition_points):
            offset = point - (transition_duration / 2)
            filter_parts.append(f"fade=t=out:st={offset}:d={transition_duration/2}:alpha=1")
            filter_parts.append(f"fade=t=in:st={point}:d={transition_duration/2}:alpha=1")
        
        filter_complex = ",".join(filter_parts)
        
        # Process with filter
        self.process_with_filter_fifo(input_file, output_file, filter_complex)
    
    def extract_and_concatenate_efficient(self, input_file: str, segments: List[VideoSegment],
                                          output_file: str, apply_transitions: bool = True) -> None:
        """Efficient extraction and concatenation using parallel FIFOs"""
        start_time = time.time()
        
        with metrics.track_processing_time('editing', video_duration=sum(s.duration for s in segments)):
            # Start parallel extraction
            logger.info(f"Extracting {len(segments)} segments in parallel...")
            segment_fifos = self.extract_segments_parallel(input_file, segments)
            
            # Create temporary output for concatenation
            temp_output = os.path.join(self.temp_dir, f"temp_{uuid.uuid4().hex[:8]}.mp4")
            
            # Concatenate segments
            logger.info("Concatenating segments...")
            self.concatenate_segments_fifo(segment_fifos, temp_output)
            
            # Apply transitions if requested
            if apply_transitions and len(segments) > 1:
                logger.info("Applying transitions...")
                transition_points = []
                current_time = 0
                for segment in segments[:-1]:
                    current_time += segment.duration
                    transition_points.append(current_time)
                
                self.apply_transitions_fifo(temp_output, output_file, transition_points)
                
                # Clean up temp file
                try:
                    os.unlink(temp_output)
                except:
                    pass
            else:
                # Move temp to final output
                os.rename(temp_output, output_file)
        
        elapsed = time.time() - start_time
        total_duration = sum(s.duration for s in segments)
        ratio = elapsed / total_duration if total_duration > 0 else 0
        
        logger.info(f"Processing completed in {elapsed:.1f}s (ratio: {ratio:.2f}x)")
        
        # Track performance metric
        metrics.processing_ratio.labels(stage='editing').observe(ratio)
        
        # Verify ratio meets requirement (< 1.2x for 1080p)
        if ratio > 1.2:
            logger.warning(f"Processing ratio {ratio:.2f}x exceeds target of 1.2x")


@contextmanager
def video_processing_pipeline():
    """Context manager for video processing pipeline"""
    pipeline = FFmpegPipeline()
    try:
        yield pipeline
    finally:
        pipeline.cleanup()


# Example usage
def example_usage():
    """Example of using the FIFO-based video processor"""
    
    editor = VideoEditor()
    
    # Define segments to extract
    segments = [
        VideoSegment(10, 40, "input.mp4"),    # 30 second segment
        VideoSegment(60, 90, "input.mp4"),    # 30 second segment  
        VideoSegment(120, 150, "input.mp4"),  # 30 second segment
    ]
    
    # Process video efficiently
    editor.extract_and_concatenate_efficient(
        "input.mp4",
        segments,
        "output.mp4",
        apply_transitions=True
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    example_usage()