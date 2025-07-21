"""Enhanced video processing pipeline with comprehensive error handling"""

import os
import subprocess
import threading
import logging
import time
import signal
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
from contextlib import contextmanager
import uuid
import shutil

# Import error handling
from ..core.exceptions import (
    VideoProcessingError,
    FFmpegError,
    VideoDecodeError,
    VideoEncodeError,
    DiskSpaceError,
    MemoryError as MontageMemoryError,
    ConfigurationError,
    create_error_context,
)
from ..utils.error_handler import (
    with_error_context,
    with_retry,
    safe_execute,
    error_recovery_context,
    handle_ffmpeg_error,
    ErrorCollector,
)

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

    MEMORY_MANAGEMENT_AVAILABLE = True
except ImportError:
    MEMORY_MANAGEMENT_AVAILABLE = False

    # Fallback stubs
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


@dataclass
class VideoSegment:
    """Represents a video segment with error context"""

    start_time: float
    end_time: float
    input_file: str
    segment_id: str = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if not self.segment_id:
            self.segment_id = str(uuid.uuid4())[:8]
        if self.metadata is None:
            self.metadata = {}

        # Validate segment
        if self.start_time < 0:
            raise VideoProcessingError(
                f"Invalid segment start time: {self.start_time}"
            ).with_suggestion("Start time must be >= 0")

        if self.end_time <= self.start_time:
            raise VideoProcessingError(
                f"Invalid segment duration: start={self.start_time}, end={self.end_time}"
            ).with_suggestion("End time must be greater than start time")

    @property
    def duration(self):
        return self.end_time - self.start_time


class FIFOManager:
    """Manages FIFO (named pipe) creation and cleanup with error handling"""

    def __init__(self, temp_dir: Optional[str] = None):
        self.temp_dir = temp_dir or get("TEMP_DIR", "/tmp/montage")
        self.fifos = []
        self._cleanup_registered = False

        # Ensure temp directory exists with proper permissions
        try:
            os.makedirs(self.temp_dir, exist_ok=True, mode=0o755)
        except OSError as e:
            if "Permission denied" in str(e):
                raise ConfigurationError(
                    f"Cannot create temp directory: {self.temp_dir}",
                    config_key="TEMP_DIR",
                ).with_suggestion(
                    "Check directory permissions or set TEMP_DIR to a writable location"
                )
            elif "No space left" in str(e):
                raise DiskSpaceError(
                    "No disk space for temporary files", required_gb=0.1, available_gb=0
                )
            else:
                raise VideoProcessingError(f"Failed to create temp directory: {str(e)}")

    @with_error_context("FIFOManager", "create_fifo")
    def create_fifo(self, name_suffix: str = "") -> str:
        """Create a named pipe with error handling"""
        fifo_name = f"fifo_{uuid.uuid4().hex[:8]}{name_suffix}"
        fifo_path = os.path.join(self.temp_dir, fifo_name)

        try:
            # Check if path already exists
            if os.path.exists(fifo_path):
                logger.warning(f"FIFO already exists, removing: {fifo_path}")
                os.unlink(fifo_path)

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
            if "Permission denied" in str(e):
                raise VideoProcessingError(
                    f"Permission denied creating FIFO in {self.temp_dir}"
                ).with_suggestion("Check directory permissions")
            elif "No space left" in str(e):
                raise DiskSpaceError(
                    "No disk space for FIFO creation", required_gb=0.001, available_gb=0
                )
            else:
                raise VideoProcessingError(f"Failed to create FIFO: {str(e)}")

    def cleanup(self):
        """Remove all created FIFOs with error handling"""
        errors = []
        for fifo in self.fifos:
            try:
                if os.path.exists(fifo):
                    os.unlink(fifo)
                    logger.debug(f"Removed FIFO: {fifo}")
            except OSError as e:
                errors.append(e)
                logger.warning(f"Failed to remove FIFO {fifo}: {e}")

        self.fifos.clear()

        if errors:
            logger.warning(f"Failed to clean up {len(errors)} FIFOs")

    def _signal_handler(self, signum, frame):
        """Clean up FIFOs on signal"""
        logger.info("Received signal, cleaning up FIFOs...")
        self.cleanup()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()


class FFmpegPipeline:
    """Manages FFmpeg processes with comprehensive error handling"""

    def __init__(self):
        self.processes = []
        self.threads = []
        self.fifo_manager = FIFOManager()
        self.error_collector = ErrorCollector()

    @with_error_context("FFmpegPipeline", "add_process")
    def add_process(self, cmd: List[str], name: str = "ffmpeg") -> subprocess.Popen:
        """Add an FFmpeg process with error handling"""
        # Validate FFmpeg is available
        ffmpeg_path = cmd[0] if cmd else "ffmpeg"
        if not shutil.which(ffmpeg_path):
            raise ConfigurationError(
                f"FFmpeg not found: {ffmpeg_path}", config_key="FFMPEG_PATH"
            ).with_suggestion("Install FFmpeg or set FFMPEG_PATH correctly")

        try:
            logger.info(f"Starting {name} process: {' '.join(cmd[:10])}...")

            # Track FFmpeg process
            track_ffmpeg_process_start()

            # Check available disk space before starting
            if MEMORY_MANAGEMENT_AVAILABLE:
                manager = get_video_resource_manager()
                if manager:
                    try:
                        available_gb = manager.get_available_disk_space() / (1024**3)
                        if available_gb < 1.0:  # Less than 1GB
                            raise DiskSpaceError(
                                "Insufficient disk space for video processing",
                                required_gb=1.0,
                                available_gb=available_gb,
                            )
                    except Exception:
                        pass  # Continue if we can't check disk space

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

        except subprocess.CalledProcessError as e:
            track_ffmpeg_process_end()
            raise handle_ffmpeg_error(
                command=cmd,
                return_code=e.returncode,
                stderr=str(e),
                operation=f"Start {name} process",
            )
        except OSError as e:
            track_ffmpeg_process_end()
            if "No such file" in str(e):
                raise ConfigurationError(
                    f"FFmpeg executable not found: {ffmpeg_path}",
                    config_key="FFMPEG_PATH",
                )
            else:
                raise FFmpegError(
                    f"Failed to start {name}: {str(e)}", command=cmd, stderr=""
                )

    def _monitor_stderr(self, process: subprocess.Popen, name: str):
        """Monitor process stderr with error detection"""
        critical_errors = []
        warnings = []

        try:
            for line in iter(process.stderr.readline, b""):
                if not line:
                    break

                line_str = line.decode("utf-8", errors="ignore").strip()

                # Detect critical errors
                if any(
                    err in line_str.lower() for err in ["error", "failed", "invalid"]
                ):
                    critical_errors.append(line_str)
                    logger.error(f"{name}: {line_str}")

                    # Check for specific error patterns
                    if "no space left" in line_str.lower():
                        self.error_collector.collect(
                            process_name=name,
                            error=DiskSpaceError(
                                "FFmpeg: No disk space", required_gb=1.0, available_gb=0
                            ),
                        )
                    elif "permission denied" in line_str.lower():
                        self.error_collector.collect(
                            process_name=name,
                            error=VideoProcessingError(
                                f"FFmpeg: Permission denied"
                            ).with_suggestion("Check file permissions"),
                        )

                # Log warnings
                elif "warning" in line_str.lower():
                    warnings.append(line_str)
                    logger.warning(f"{name}: {line_str}")

                # Progress updates
                elif "frame=" in line_str or "time=" in line_str:
                    logger.debug(f"{name} progress: {line_str}")

        except Exception as e:
            logger.error(f"Error monitoring {name} stderr: {e}")
            self.error_collector.collect(
                process_name=name,
                error=VideoProcessingError(f"Failed to monitor {name}: {str(e)}"),
            )

    @with_error_context("FFmpegPipeline", "wait_all")
    def wait_all(self, timeout: Optional[int] = None) -> Dict[str, int]:
        """Wait for all processes with comprehensive error handling"""
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
                    # Get full error output
                    try:
                        _, stderr = process.communicate(timeout=5)
                        stderr_text = stderr.decode("utf-8", errors="ignore")
                    except:
                        stderr_text = "Could not retrieve error output"

                    # Create appropriate error
                    error = handle_ffmpeg_error(
                        command=["ffmpeg"],  # Command not stored
                        return_code=return_code,
                        stderr=stderr_text,
                        operation=f"{name} process",
                    )

                    self.error_collector.collect(process_name=name, error=error)

                    logger.error(f"{name} failed with code {return_code}")

            except subprocess.TimeoutExpired:
                logger.error(f"{name} timed out after {timeout}s")
                process.terminate()

                # Give it time to terminate gracefully
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    logger.error(f"Force killing {name}")
                    process.kill()

                results[name] = -1

                self.error_collector.collect(
                    process_name=name,
                    error=VideoProcessingError(
                        f"{name} process timed out after {timeout}s"
                    ).with_suggestion(
                        "Consider increasing timeout or processing smaller segments"
                    ),
                )

            except Exception as e:
                logger.error(f"Unexpected error waiting for {name}: {e}")
                results[name] = -2

                self.error_collector.collect(
                    process_name=name,
                    error=VideoProcessingError(f"Failed to wait for {name}: {str(e)}"),
                )
            finally:
                track_ffmpeg_process_end()

        # Check if any errors were collected
        if self.error_collector.has_errors():
            # Raise aggregated error if all processes failed
            failed_count = sum(1 for code in results.values() if code != 0)
            if failed_count == len(results):
                self.error_collector.raise_if_errors("All FFmpeg processes failed")
            elif failed_count > 0:
                logger.warning(f"{failed_count}/{len(results)} processes failed")

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
    """Video editor with comprehensive error handling and recovery"""

    def __init__(self):
        self.ffmpeg_path = get("FFMPEG_PATH", "ffmpeg")
        self.temp_dir = get("TEMP_DIR", "/tmp/montage")

        # Validate FFmpeg availability
        if not shutil.which(self.ffmpeg_path):
            raise ConfigurationError(
                f"FFmpeg not found at: {self.ffmpeg_path}", config_key="FFMPEG_PATH"
            ).with_suggestion("Install FFmpeg or set FFMPEG_PATH environment variable")

        # Initialize memory management if available
        if MEMORY_MANAGEMENT_AVAILABLE:
            self.memory_monitor = get_memory_monitor()
            self.adaptive_config = get_adaptive_config()
            self.video_resource_manager = get_video_resource_manager()
            self.ffmpeg_manager = get_ffmpeg_memory_manager()

            # Initialize if not already done
            if self.memory_monitor and not hasattr(self.memory_monitor, "_monitoring"):
                initialize_memory_management()
        else:
            self.memory_monitor = None
            self.adaptive_config = None
            self.video_resource_manager = None
            self.ffmpeg_manager = None

        logger.info("VideoEditor initialized with error handling")

    @with_error_context("VideoEditor", "extract_segments")
    @with_retry(max_attempts=3, base_delay=2.0)
    def extract_segments_parallel(
        self,
        input_file: str,
        segments: List[VideoSegment],
        job_id: Optional[str] = None,
    ) -> List[str]:
        """Extract multiple segments with error handling and recovery"""
        if not os.path.exists(input_file):
            raise VideoDecodeError("Input video file not found", video_path=input_file)

        if not segments:
            raise VideoProcessingError(
                "No segments provided for extraction"
            ).with_suggestion("Provide at least one video segment")

        # Validate segments
        for segment in segments:
            if segment.input_file != input_file:
                raise VideoProcessingError(
                    f"Segment references different input file: {segment.input_file}"
                ).with_suggestion("All segments must reference the same input file")

        with FFmpegPipeline() as pipeline:
            segment_fifos = []
            error_collector = ErrorCollector()

            for segment in segments:
                with error_collector.collect(segment_id=segment.segment_id):
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

            # Check if too many extractions failed
            if error_collector.has_errors():
                failed_count = len(error_collector.get_errors())
                if failed_count > len(segments) / 2:
                    error_collector.raise_if_errors(
                        f"Too many segment extractions failed ({failed_count}/{len(segments)})"
                    )
                else:
                    logger.warning(
                        f"Some extractions failed ({failed_count}/{len(segments)}), continuing",
                        extra={"job_id": job_id},
                    )

            # Return FIFO paths (including failed ones for error handling)
            return segment_fifos

    @with_error_context("VideoEditor", "concatenate_segments")
    def concatenate_segments_fifo(
        self,
        segment_fifos: List[str],
        output_file: str,
        video_codec: str = "libx264",
        audio_codec: str = "aac",
        job_id: Optional[str] = None,
    ) -> None:
        """Concatenate segments with comprehensive error handling"""
        if not segment_fifos:
            raise VideoProcessingError("No segment FIFOs provided for concatenation")

        # Validate FIFOs exist
        missing_fifos = [f for f in segment_fifos if not os.path.exists(f)]
        if missing_fifos:
            if len(missing_fifos) == len(segment_fifos):
                raise VideoProcessingError(
                    "All segment FIFOs are missing"
                ).with_suggestion("Segment extraction may have failed")
            else:
                logger.warning(
                    f"{len(missing_fifos)} FIFOs missing, continuing with available segments",
                    extra={"job_id": job_id},
                )
                segment_fifos = [f for f in segment_fifos if os.path.exists(f)]

        # Create output directory if needed
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            try:
                os.makedirs(output_dir, exist_ok=True)
            except OSError as e:
                raise VideoEncodeError(
                    f"Cannot create output directory: {output_dir}",
                    output_path=output_file,
                ).with_suggestion("Check directory permissions")

        with FFmpegPipeline() as pipeline:
            concat_list = None
            try:
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
                    "fast",
                    "-movflags",
                    "+faststart",
                    "-y",
                    output_file,
                ]

                # Add quality settings based on codec
                if video_codec == "libx264":
                    cmd.extend(["-crf", "23"])  # Good quality
                elif video_codec == "libx265":
                    cmd.extend(["-crf", "28"])  # Good quality for H.265

                # Start concatenation
                pipeline.add_process(cmd, "concatenate")

                # Wait for completion with timeout
                results = pipeline.wait_all(timeout=1800)  # 30 minute timeout

                if results.get("concatenate", -1) != 0:
                    # Check if output was partially created
                    if os.path.exists(output_file):
                        file_size = os.path.getsize(output_file)
                        if file_size > 0:
                            logger.warning(
                                f"Concatenation failed but partial output exists ({file_size} bytes)",
                                extra={"job_id": job_id},
                            )
                        else:
                            os.unlink(output_file)

                    raise FFmpegError(
                        "Video concatenation failed",
                        command=cmd,
                        stderr="Check FFmpeg logs for details",
                    )

                # Verify output
                if not os.path.exists(output_file):
                    raise VideoEncodeError(
                        "Output file was not created", output_path=output_file
                    )

                file_size = os.path.getsize(output_file)
                if file_size == 0:
                    raise VideoEncodeError(
                        "Output file is empty", output_path=output_file
                    ).with_suggestion("Check if input segments contain valid data")

                logger.info(
                    f"Concatenation completed successfully: {output_file} ({file_size / 1024 / 1024:.1f}MB)",
                    extra={"job_id": job_id},
                )

            finally:
                # Clean up concat list
                if concat_list and os.path.exists(concat_list):
                    try:
                        os.unlink(concat_list)
                    except OSError:
                        pass

    def _create_concat_list(self, fifos: List[str]) -> str:
        """Create concat demuxer list file with error handling"""
        concat_file = os.path.join(self.temp_dir, f"concat_{uuid.uuid4().hex[:8]}.txt")

        try:
            with open(concat_file, "w") as f:
                for fifo in fifos:
                    # Escape single quotes in path
                    escaped_path = fifo.replace("'", "'\\''")
                    f.write(f"file '{escaped_path}'\n")

            return concat_file

        except IOError as e:
            if "Permission denied" in str(e):
                raise VideoProcessingError(
                    f"Permission denied creating concat list in {self.temp_dir}"
                ).with_suggestion("Check directory permissions")
            elif "No space left" in str(e):
                raise DiskSpaceError(
                    "No disk space for concat list", required_gb=0.001, available_gb=0
                )
            else:
                raise VideoProcessingError(f"Failed to create concat list: {str(e)}")

    @with_error_context("VideoEditor", "process_video")
    def extract_and_concatenate_efficient(
        self,
        input_file: str,
        segments: List[VideoSegment],
        output_file: str,
        apply_transitions: bool = True,
        job_id: Optional[str] = None,
    ) -> None:
        """Process video with comprehensive error handling and recovery"""
        start_time = time.time()

        if not segments:
            raise VideoProcessingError("No segments provided")

        # Validate input
        if not os.path.exists(input_file):
            raise VideoDecodeError("Input video not found", video_path=input_file)

        # Check input file size
        input_size = os.path.getsize(input_file)
        if input_size == 0:
            raise VideoDecodeError("Input video file is empty", video_path=input_file)

        # Estimate resource requirements
        estimated_memory_mb = 512  # Default
        if MEMORY_MANAGEMENT_AVAILABLE:
            try:
                estimated_memory_mb = estimate_processing_memory(
                    input_file, "transcoding"
                )
            except Exception as e:
                logger.warning(f"Could not estimate memory: {e}")

        logger.info(
            f"Processing {len(segments)} segments (estimated memory: {estimated_memory_mb}MB)",
            extra={"job_id": job_id, "input_size_mb": input_size / 1024 / 1024},
        )

        # Use memory guard if available
        if MEMORY_MANAGEMENT_AVAILABLE:
            memory_context = memory_guard(max_memory_mb=estimated_memory_mb * 2)
        else:
            from contextlib import nullcontext

            memory_context = nullcontext()

        with memory_context:
            try:
                # Determine processing method based on file size and available resources
                should_stream = False

                if MEMORY_MANAGEMENT_AVAILABLE and self.video_resource_manager:
                    should_stream = self.video_resource_manager.should_use_streaming(
                        input_file
                    )
                else:
                    # Simple heuristic: stream if file > 1GB
                    should_stream = input_size > 1024 * 1024 * 1024

                if should_stream:
                    logger.info(
                        "Using streaming processing for large video",
                        extra={"job_id": job_id},
                    )
                    success = self._process_large_video_streaming(
                        input_file, segments, output_file, apply_transitions, job_id
                    )
                else:
                    logger.info("Using direct processing", extra={"job_id": job_id})
                    success = self._process_video_direct(
                        input_file, segments, output_file, apply_transitions, job_id
                    )

                if not success:
                    raise VideoProcessingError(
                        "Video processing failed without specific error"
                    )

                # Verify output
                if not os.path.exists(output_file):
                    raise VideoEncodeError(
                        "Output file was not created", output_path=output_file
                    )

                output_size = os.path.getsize(output_file)
                if output_size == 0:
                    raise VideoEncodeError(
                        "Output file is empty", output_path=output_file
                    )

                # Calculate performance metrics
                elapsed = time.time() - start_time
                total_duration = sum(s.duration for s in segments)
                ratio = elapsed / total_duration if total_duration > 0 else 0

                logger.info(
                    f"Video processing completed successfully",
                    extra={
                        "job_id": job_id,
                        "output_file": output_file,
                        "output_size_mb": output_size / 1024 / 1024,
                        "processing_time": elapsed,
                        "processing_ratio": ratio,
                    },
                )

                # Track metrics
                metrics.processing_ratio.labels(stage="editing").observe(ratio)

                if ratio > 1.2:
                    logger.warning(
                        f"Processing ratio {ratio:.2f}x exceeds target of 1.2x",
                        extra={"job_id": job_id},
                    )

            except Exception as e:
                # Add context to error
                if isinstance(e, MontageError):
                    e.add_context(
                        job_id=job_id,
                        input_file=input_file,
                        output_file=output_file,
                        segment_count=len(segments),
                    )
                    raise
                else:
                    # Wrap unexpected errors
                    raise VideoProcessingError(
                        f"Unexpected error during video processing: {str(e)}"
                    ).add_context(
                        job_id=job_id, input_file=input_file, output_file=output_file
                    )
            finally:
                # Always cleanup resources
                if MEMORY_MANAGEMENT_AVAILABLE:
                    cleanup_video_resources()

    def _process_video_direct(
        self,
        input_file: str,
        segments: List[VideoSegment],
        output_file: str,
        apply_transitions: bool,
        job_id: Optional[str] = None,
    ) -> bool:
        """Direct video processing with error handling"""
        try:
            # Convert VideoSegment objects to video clips format
            video_clips = []
            for i, segment in enumerate(segments):
                video_clips.append(
                    {
                        "start_ms": int(segment.start_time * 1000),
                        "end_ms": int(segment.end_time * 1000),
                        "text": segment.metadata.get("text", f"Segment {i+1}"),
                        "score": segment.metadata.get("score", 8.0),
                    }
                )

            logger.info(
                f"Converted {len(segments)} segments for processing",
                extra={"job_id": job_id},
            )

            # Use intelligent vertical cropping
            try:
                from ..utils.intelligent_crop import create_intelligent_vertical_video

                return create_intelligent_vertical_video(
                    video_clips, input_file, output_file
                )
            except ImportError:
                logger.warning("Intelligent crop not available, using basic processing")
                # Fallback to basic extraction and concatenation
                fifos = self.extract_segments_parallel(input_file, segments, job_id)
                self.concatenate_segments_fifo(fifos, output_file, job_id=job_id)
                return True

        except Exception as e:
            logger.error(f"Direct processing failed: {e}", extra={"job_id": job_id})
            if isinstance(e, MontageError):
                raise
            else:
                raise VideoProcessingError(
                    f"Direct processing failed: {str(e)}"
                ).add_context(job_id=job_id)

    def _process_large_video_streaming(
        self,
        input_file: str,
        segments: List[VideoSegment],
        output_file: str,
        apply_transitions: bool,
        job_id: Optional[str] = None,
    ) -> bool:
        """Streaming processing for large videos with error handling"""
        if not MEMORY_MANAGEMENT_AVAILABLE:
            logger.warning(
                "Memory management not available, falling back to direct processing",
                extra={"job_id": job_id},
            )
            return self._process_video_direct(
                input_file, segments, output_file, apply_transitions, job_id
            )

        try:
            # Process in chunks to manage memory
            chunk_duration = 120  # 2 minutes per chunk
            processed_chunks = []

            with error_recovery_context(
                "streaming_video_processing",
                cleanup=lambda: self._cleanup_chunks(processed_chunks),
            ):
                # Group segments by time proximity for efficient chunking
                segment_groups = self._group_segments_by_time(segments, chunk_duration)

                logger.info(
                    f"Processing video in {len(segment_groups)} chunks",
                    extra={"job_id": job_id},
                )

                for i, group in enumerate(segment_groups):
                    chunk_output = os.path.join(
                        self.temp_dir, f"chunk_{job_id}_{i:03d}.mp4"
                    )

                    # Process chunk
                    success = self._process_video_direct(
                        input_file, group, chunk_output, False, job_id
                    )

                    if success and os.path.exists(chunk_output):
                        processed_chunks.append(chunk_output)
                    else:
                        logger.warning(
                            f"Chunk {i+1}/{len(segment_groups)} failed",
                            extra={"job_id": job_id},
                        )

                if not processed_chunks:
                    raise VideoProcessingError("No chunks were successfully processed")

                # Concatenate all chunks
                logger.info(
                    f"Concatenating {len(processed_chunks)} chunks",
                    extra={"job_id": job_id},
                )

                self._concatenate_video_files(processed_chunks, output_file, job_id)

                return True

        except Exception as e:
            logger.error(f"Streaming processing failed: {e}", extra={"job_id": job_id})
            if isinstance(e, MontageError):
                raise
            else:
                raise VideoProcessingError(
                    f"Streaming processing failed: {str(e)}"
                ).add_context(job_id=job_id)
        finally:
            # Cleanup temporary chunks
            self._cleanup_chunks(
                processed_chunks if "processed_chunks" in locals() else []
            )

    def _group_segments_by_time(
        self, segments: List[VideoSegment], max_duration: float
    ) -> List[List[VideoSegment]]:
        """Group segments into chunks based on time proximity"""
        if not segments:
            return []

        groups = []
        current_group = []
        current_duration = 0

        for segment in sorted(segments, key=lambda s: s.start_time):
            segment_duration = segment.duration

            # Start new group if adding this segment would exceed max duration
            if current_group and current_duration + segment_duration > max_duration:
                groups.append(current_group)
                current_group = []
                current_duration = 0

            current_group.append(segment)
            current_duration += segment_duration

        if current_group:
            groups.append(current_group)

        return groups

    def _concatenate_video_files(
        self, input_files: List[str], output_file: str, job_id: Optional[str] = None
    ):
        """Concatenate multiple video files"""
        if not input_files:
            raise VideoProcessingError("No input files for concatenation")

        # Create concat list
        concat_list = self._create_concat_list(input_files)

        try:
            cmd = [
                self.ffmpeg_path,
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                concat_list,
                "-c",
                "copy",  # Copy codecs for speed
                "-y",
                output_file,
            ]

            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=600  # 10 minutes
            )

            if result.returncode != 0:
                raise handle_ffmpeg_error(
                    command=cmd,
                    return_code=result.returncode,
                    stderr=result.stderr,
                    operation="Concatenate video files",
                )

        finally:
            if os.path.exists(concat_list):
                os.unlink(concat_list)

    def _cleanup_chunks(self, chunk_files: List[str]):
        """Clean up temporary chunk files"""
        for chunk_file in chunk_files:
            if os.path.exists(chunk_file):
                try:
                    os.unlink(chunk_file)
                except OSError:
                    pass
