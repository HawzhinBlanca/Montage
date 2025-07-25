"""FIFO-based video processing pipeline using FFmpeg"""

import os
import subprocess
import threading
import logging
import time
import signal
from typing import List, Dict, Optional
from dataclasses import dataclass
from contextlib import contextmanager
import uuid

try:
    from ..config import get
except ImportError:
    from src.config import get
from ..core.metrics import metrics, track_ffmpeg_process_start, track_ffmpeg_process_end

# Import comprehensive memory management
try:
    from ..utils.memory_manager import (
        get_memory_monitor,
        get_adaptive_config,
        initialize_memory_management,
        memory_guard,
        MemoryPressureLevel,
    )
    from ..utils.resource_manager import (
        get_video_resource_manager,
        managed_tempfile,
        cleanup_video_resources,
        get_memory_safe_worker_count,
        estimate_processing_memory,
    )
    from ..utils.ffmpeg_memory_manager import (
        get_ffmpeg_memory_manager,
        memory_safe_ffmpeg,
        process_large_video_safely,
        FFmpegMemoryProfile,
    )
except ImportError:
    # Fallback if memory management is not available
    logger.warning("Memory management modules not available - using basic processing")

    def get_memory_monitor():
        return None

    def get_adaptive_config():
        return None

    def initialize_memory_management():
        return None, None, None

    memory_guard = contextmanager(lambda **kwargs: (yield))

    def get_video_resource_manager():
        return None

    managed_tempfile = contextmanager(lambda **kwargs: (yield "/tmp/temp_file"))

    def cleanup_video_resources():
        pass

    def get_memory_safe_worker_count():
        return 2

    def estimate_processing_memory(*args):
        return 512

    def get_ffmpeg_memory_manager():
        return None

    memory_safe_ffmpeg = contextmanager(lambda **kwargs: (yield))

    def process_large_video_safely(*args):
        return True


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
        self.temp_dir = temp_dir or get("TEMP_DIR", "/tmp/montage")
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
                bufsize=0,  # Unbuffered for real-time processing
            )

            self.processes.append((name, process))

            # Start thread to monitor stderr
            error_thread = threading.Thread(
                target=self._monitor_stderr, args=(process, name), daemon=True
            )
            error_thread.start()
            self.threads.append(error_thread)

            return process

        except (
            subprocess.CalledProcessError,
            OSError,
            FileNotFoundError,
            PermissionError,
        ) as e:
            track_ffmpeg_process_end()
            raise FFmpegError(f"Failed to start {name}: {e}") from e

    def _monitor_stderr(self, process: subprocess.Popen, name: str):
        """Monitor process stderr for errors and progress"""
        try:
            for line in iter(process.stderr.readline, b""):
                if not line:
                    break

                line_str = line.decode("utf-8", errors="ignore").strip()

                # Log errors
                if "error" in line_str.lower():
                    logger.error(f"{name}: {line_str}")
                # Log warnings
                elif "warning" in line_str.lower():
                    logger.warning(f"{name}: {line_str}")
                # Progress updates
                elif "frame=" in line_str or "time=" in line_str:
                    logger.debug(f"{name} progress: {line_str}")

        except (OSError, ValueError, BrokenPipeError, UnicodeDecodeError) as e:
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
                    logger.error(
                        f"{name} failed with code {return_code}: {stderr.decode('utf-8', errors='ignore')}"
                    )

            except subprocess.TimeoutExpired:
                logger.error(f"{name} timed out, terminating...")
                process.terminate()
                results[name] = -1
            except (subprocess.SubprocessError, OSError, ValueError) as e:
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
    """High-performance video editor with comprehensive memory management"""

    def __init__(self):
        self.ffmpeg_path = get("FFMPEG_PATH", "ffmpeg")
        self.temp_dir = get("TEMP_DIR", "/tmp/montage")

        # Initialize memory management
        self.memory_monitor = get_memory_monitor()
        self.adaptive_config = get_adaptive_config()
        self.video_resource_manager = get_video_resource_manager()
        self.ffmpeg_manager = get_ffmpeg_memory_manager()

        # Initialize memory management system if not already done
        if self.memory_monitor and not hasattr(self.memory_monitor, "_monitoring"):
            initialize_memory_management()

        logger.info("VideoEditor initialized with memory management")

    def extract_segments_parallel(
        self, input_file: str, segments: List[VideoSegment]
    ) -> List[str]:
        """Extract multiple segments in parallel using FIFOs"""
        with FFmpegPipeline() as pipeline:
            segment_fifos = []

            for segment in segments:
                # Create FIFO for segment
                fifo_path = pipeline.fifo_manager.create_fifo(
                    f"_seg_{segment.segment_id}"
                )
                segment_fifos.append(fifo_path)

                # Build extraction command
                cmd = [
                    self.ffmpeg_path,
                    "-ss",
                    str(segment.start_time),
                    "-t",
                    str(segment.duration),
                    "-i",
                    input_file,
                    "-c",
                    "copy",  # No re-encoding for speed
                    "-f",
                    "mpegts",  # Use MPEG-TS for streaming
                    "-y",
                    fifo_path,
                ]

                # Start extraction process
                pipeline.add_process(cmd, f"extract_{segment.segment_id}")

            # Return FIFO paths immediately - processes run in background
            return segment_fifos

    def concatenate_segments_fifo(
        self,
        segment_fifos: List[str],
        output_file: str,
        video_codec: str = "libx264",
        audio_codec: str = "aac",
    ) -> None:
        """Concatenate segments from FIFOs without intermediate files"""
        with FFmpegPipeline() as pipeline:
            # Create concat demuxer input
            concat_list = self._create_concat_list(segment_fifos)

            # Build concatenation command
            cmd = [
                self.ffmpeg_path,
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                concat_list,
                "-c:v",
                video_codec,
                "-c:a",
                audio_codec,
                "-preset",
                "fast",  # Balance speed/quality
                "-movflags",
                "+faststart",  # Web optimization
                "-y",
                output_file,
            ]

            # Start concatenation
            pipeline.add_process(cmd, "concatenate")

            # Wait for completion
            results = pipeline.wait_all(timeout=1800)  # 30 minute timeout

            if results.get("concatenate", -1) != 0:
                raise FFmpegError("Concatenation failed")

            # Clean up concat list
            try:
                os.unlink(concat_list)
            except (OSError, FileNotFoundError, PermissionError):
                pass  # Cleanup errors are non-critical

    def _create_concat_list(self, fifos: List[str]) -> str:
        """Create concat demuxer list file"""
        concat_file = os.path.join(self.temp_dir, f"concat_{uuid.uuid4().hex[:8]}.txt")

        with open(concat_file, "w") as f:
            for fifo in fifos:
                f.write(f"file '{fifo}'\n")

        return concat_file

    def process_with_filter_fifo(
        self,
        input_file: str,
        output_file: str,
        filter_complex: str,
        video_codec: str = "libx264",
        audio_codec: str = "aac",
    ) -> None:
        """Process video through filter using FIFO pipeline"""
        with FFmpegPipeline() as pipeline:
            # Create intermediate FIFO
            filtered_fifo = pipeline.fifo_manager.create_fifo("_filtered")

            # First pass: apply filters to FIFO
            filter_cmd = [
                self.ffmpeg_path,
                "-i",
                input_file,
                "-filter_complex",
                filter_complex,
                "-f",
                "mpegts",
                "-c:v",
                "libx264",
                "-preset",
                "ultrafast",  # Fast for intermediate
                "-c:a",
                "pcm_s16le",  # Lossless audio
                "-y",
                filtered_fifo,
            ]

            pipeline.add_process(filter_cmd, "filter")

            # Second pass: encode from FIFO
            encode_cmd = [
                self.ffmpeg_path,
                "-i",
                filtered_fifo,
                "-c:v",
                video_codec,
                "-preset",
                "fast",
                "-c:a",
                audio_codec,
                "-movflags",
                "+faststart",
                "-y",
                output_file,
            ]

            pipeline.add_process(encode_cmd, "encode")

            # Wait for completion
            results = pipeline.wait_all(timeout=1800)

            if any(code != 0 for code in results.values()):
                raise FFmpegError(f"Processing failed: {results}")

    def apply_transitions_fifo(
        self,
        input_file: str,
        output_file: str,
        transition_points: List[float],
        transition_duration: float = 0.5,
    ) -> None:
        """Apply transitions at specified points using xfade filter"""
        if not transition_points:
            # No transitions, just copy
            subprocess.run(
                [self.ffmpeg_path, "-i", input_file, "-c", "copy", "-y", output_file],
                check=True,
            )
            return

        # Build xfade filter chain
        filter_parts = []
        for i, point in enumerate(transition_points):
            offset = point - (transition_duration / 2)
            filter_parts.append(
                f"fade=t=out:st={offset}:d={transition_duration/2}:alpha=1"
            )
            filter_parts.append(
                f"fade=t=in:st={point}:d={transition_duration/2}:alpha=1"
            )

        filter_complex = ",".join(filter_parts)

        # Process with filter
        self.process_with_filter_fifo(input_file, output_file, filter_complex)

    def extract_and_concatenate_efficient(
        self,
        input_file: str,
        segments: List[VideoSegment],
        output_file: str,
        apply_transitions: bool = True,
    ) -> None:
        """Memory-optimized video processing with comprehensive resource management"""
        start_time = time.time()

        if not segments:
            raise VideoProcessingError("No segments provided")

        # Estimate memory requirements
        estimated_memory_mb = estimate_processing_memory(input_file, "transcoding")

        logger.info(
            f"🎬 Processing {len(segments)} segments with MEMORY-OPTIMIZED pipeline..."
            f" (estimated memory: {estimated_memory_mb}MB)"
        )

        # Use memory-constrained operation
        with memory_guard(max_memory_mb=estimated_memory_mb * 2) as monitor:
            try:
                # Check if we should use streaming processing for large files
                should_stream = (
                    self.video_resource_manager
                    and self.video_resource_manager.should_use_streaming(input_file)
                )

                if should_stream:
                    logger.info("🔄 Using streaming processing for large video")
                    success = self._process_large_video_streaming(
                        input_file, segments, output_file, apply_transitions
                    )
                else:
                    logger.info("⚡ Using direct processing")
                    success = self._process_video_direct(
                        input_file, segments, output_file, apply_transitions
                    )

                if not success:
                    raise VideoProcessingError("Video processing failed")

                elapsed = time.time() - start_time
                total_duration = sum(s.duration for s in segments)
                ratio = elapsed / total_duration if total_duration > 0 else 0

                # Get final memory stats
                if monitor:
                    final_stats = monitor.get_current_stats()
                    logger.info(
                        f"✅ Video processing completed: {output_file}"
                        f" (memory usage: {final_stats.process_memory_mb:.0f}MB)"
                    )
                else:
                    logger.info(f"✅ Video processing completed: {output_file}")

                logger.info(
                    f"📊 Processing ratio: {ratio:.2f}x ({elapsed:.1f}s for {total_duration:.1f}s content)"
                )

                # Track performance metrics
                metrics.processing_ratio.labels(stage="editing").observe(ratio)

                if ratio > 1.2:
                    logger.warning(
                        f"Processing ratio {ratio:.2f}x exceeds target of 1.2x"
                    )

            except (
                FFmpegError,
                subprocess.CalledProcessError,
                OSError,
                FileNotFoundError,
                ValueError,
                MemoryError,
            ) as e:
                logger.error(f"❌ Video processing failed: {e}")
                # Force cleanup on error
                cleanup_video_resources()
                raise VideoProcessingError(f"Processing failed: {e}") from e

            finally:
                # Always cleanup resources
                if self.video_resource_manager:
                    cleanup_video_resources()

    def _process_video_direct(
        self,
        input_file: str,
        segments: List[VideoSegment],
        output_file: str,
        apply_transitions: bool,
    ) -> bool:
        """Direct video processing for smaller files"""
        try:
            # Convert VideoSegment objects to video clips format
            video_clips = []
            for i, segment in enumerate(segments):
                video_clips.append(
                    {
                        "start_ms": int(segment.start_time * 1000),
                        "end_ms": int(segment.end_time * 1000),
                        "text": f"Segment {i+1}",
                        "score": 8.0,  # Default score
                    }
                )

            logger.info(
                f"🔧 Converted {len(segments)} VideoSegments to {len(video_clips)} clips"
            )

            # Use intelligent vertical cropping with memory management
            from ..utils.intelligent_crop import create_intelligent_vertical_video

            return create_intelligent_vertical_video(
                video_clips, input_file, output_file
            )

        except Exception as e:
            logger.error(f"Direct processing failed: {e}")
            return False

    def _process_large_video_streaming(
        self,
        input_file: str,
        segments: List[VideoSegment],
        output_file: str,
        apply_transitions: bool,
    ) -> bool:
        """Streaming processing for large videos using memory-safe chunks"""
        try:
            if not self.ffmpeg_manager:
                logger.warning(
                    "FFmpeg manager not available, falling back to direct processing"
                )
                return self._process_video_direct(
                    input_file, segments, output_file, apply_transitions
                )

            # Calculate total video duration for chunking
            total_duration = sum(s.duration for s in segments)

            # Get optimal chunk size based on memory
            if self.video_resource_manager:
                chunk_size_mb = self.video_resource_manager.get_optimal_chunk_size(
                    input_file
                )
                # Convert to time-based chunks (rough estimate: 1MB per second)
                chunk_duration = min(300, max(60, chunk_size_mb))  # 1-5 minutes
            else:
                chunk_duration = 120  # Default 2 minutes

            logger.info(f"🧩 Processing in {chunk_duration}s chunks")

            def process_chunk_func(chunk_input: str) -> str:
                """Process a single chunk"""
                with managed_tempfile(suffix=".mp4") as chunk_output:
                    # Apply processing to chunk (simplified for now)
                    cmd = [
                        self.ffmpeg_path,
                        "-y",
                        "-i",
                        chunk_input,
                        "-c:v",
                        "libx264",
                        "-preset",
                        "fast",
                        "-crf",
                        "23",
                        "-c:a",
                        "aac",
                        "-b:a",
                        "128k",
                        chunk_output,
                    ]

                    with memory_safe_ffmpeg(cmd, chunk_input) as process:
                        result = process.wait()

                    if result == 0:
                        return chunk_output
                    else:
                        raise ValueError(f"Chunk processing failed with code {result}")

            return self.ffmpeg_manager.process_video_chunks(
                input_file, output_file, process_chunk_func, chunk_duration
            )

        except Exception as e:
            logger.error(f"Streaming processing failed: {e}")
            return False


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
        VideoSegment(10, 40, "input.mp4"),  # 30 second segment
        VideoSegment(60, 90, "input.mp4"),  # 30 second segment
        VideoSegment(120, 150, "input.mp4"),  # 30 second segment
    ]

    # Process video efficiently
    editor.extract_and_concatenate_efficient(
        "input.mp4", segments, "output.mp4", apply_transitions=True
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    example_usage()
