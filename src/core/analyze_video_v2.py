#!/usr/bin/env python3
"""
Video analysis module with comprehensive error propagation.
This is an enhanced version of analyze_video.py with proper error handling.
"""
import hashlib
import os
import tempfile
from typing import Dict, List, Any, Optional
from decimal import Decimal

import psycopg2
from faster_whisper import WhisperModel
from deepgram import DeepgramClient, PrerecordedOptions
import ffmpeg

# Import error handling
from ..core.exceptions import (
    TranscriptionError,
    WhisperError,
    DeepgramError,
    FFmpegError,
    VideoDecodeError,
    DatabaseError,
    ConnectionError,
    QueryError,
    ConfigurationError,
    DiskSpaceError,
    MemoryError as MontageMemoryError,
    create_error_context,
)
from ..utils.error_handler import (
    with_error_context,
    with_retry,
    safe_execute,
    handle_ffmpeg_error,
    handle_api_error,
)

# Import centralized logging
try:
    from ..utils.logging_config import get_logger, correlation_context, log_performance
except ImportError:
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).parent.parent))
    from utils.logging_config import get_logger, correlation_context, log_performance

# Configure module logger
logger = get_logger(__name__)

# Import rate limiting
try:
    from .api_wrappers import deepgram_service, graceful_api_call
    from .rate_limiter import Priority

    RATE_LIMITING_AVAILABLE = True
except ImportError:
    RATE_LIMITING_AVAILABLE = False
    logger.warning("Rate limiting not available - using direct API calls")

try:
    from ..config import DATABASE_URL, DEEPGRAM_API_KEY
except ImportError:
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).parent.parent))
    from config import DATABASE_URL, DEEPGRAM_API_KEY

# Fix cache permissions
os.environ["HF_HOME"] = os.path.expanduser("~/.cache/huggingface")
os.environ["TRANSFORMERS_CACHE"] = os.path.expanduser("~/.cache/huggingface")


@with_error_context("VideoAnalysis", "calculate_file_hash")
def sha256_file(path: str) -> str:
    """Calculate SHA256 hash of file for caching"""
    if not os.path.exists(path):
        raise VideoDecodeError(
            f"Video file not found: {path}", video_path=path
        ).with_suggestion("Check if the file path is correct")

    try:
        hash_sha256 = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()
    except (IOError, OSError) as e:
        if "Permission denied" in str(e):
            raise VideoDecodeError(
                f"Permission denied reading video file", video_path=path
            ).with_suggestion("Check file permissions")
        elif "No space left" in str(e):
            raise DiskSpaceError(
                "No disk space available for video processing",
                required_gb=0.1,  # Minimal space needed
                available_gb=0,
            )
        else:
            raise VideoDecodeError(
                f"Failed to read video file: {str(e)}", video_path=path
            )


@with_error_context("VideoAnalysis", "extract_audio")
@with_retry(max_attempts=3, base_delay=1.0)
def extract_audio_to_wav(video_path: str, job_id: Optional[str] = None) -> str:
    """Extract audio from video to WAV format with proper error handling"""
    if not os.path.exists(video_path):
        raise VideoDecodeError(
            f"Video file not found for audio extraction", video_path=video_path
        )

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        output_path = tmp.name

    try:
        # Get video info first to detect issues early
        probe = ffmpeg.probe(video_path)

        # Check if video has audio stream
        audio_streams = [s for s in probe["streams"] if s["codec_type"] == "audio"]
        if not audio_streams:
            raise VideoDecodeError(
                "Video file has no audio stream", video_path=video_path
            ).with_suggestion(
                "This video appears to be silent - check if audio is expected"
            )

        # Extract audio
        stream = ffmpeg.input(video_path)
        stream = ffmpeg.output(
            stream, output_path, acodec="pcm_s16le", ac=1, ar="16000"
        )
        stream = ffmpeg.overwrite_output(stream)

        # Run FFmpeg with proper error capture
        out, err = ffmpeg.run(stream, capture_stdout=True, capture_stderr=True)

        if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
            raise FFmpegError(
                "FFmpeg produced empty or no output",
                command=ffmpeg.compile(stream),
                stderr=err.decode("utf-8") if err else "",
            )

        return output_path

    except ffmpeg.Error as e:
        # Clean up on error
        if os.path.exists(output_path):
            try:
                os.unlink(output_path)
            except OSError:
                pass

        # Convert FFmpeg error with context
        stderr = (
            e.stderr.decode("utf-8") if hasattr(e, "stderr") and e.stderr else str(e)
        )
        raise handle_ffmpeg_error(
            command=["ffmpeg", "-i", video_path, "-acodec", "pcm_s16le", output_path],
            return_code=1,
            stderr=stderr,
            operation="Audio extraction",
        )
    except Exception as e:
        # Clean up on any error
        if os.path.exists(output_path):
            try:
                os.unlink(output_path)
            except OSError:
                pass
        raise


@with_error_context("Transcription", "whisper_transcribe")
def transcribe_whisper(
    wav_path: str, job_id: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Faster-whisper transcription with proper error handling"""
    if not os.path.exists(wav_path):
        raise WhisperError(
            "Audio file not found for transcription", audio_path=wav_path
        )

    try:
        # Set cache directory to avoid permission issues
        cache_dir = os.path.join(tempfile.gettempdir(), "whisper_cache")
        os.makedirs(cache_dir, exist_ok=True)

        # Load model with error handling
        try:
            model = WhisperModel(
                "base", device="cpu", compute_type="int8", download_root=cache_dir
            )
        except Exception as e:
            if "CUDA" in str(e) or "GPU" in str(e):
                raise WhisperError(
                    "GPU/CUDA error loading Whisper model", audio_path=wav_path
                ).with_suggestion("Falling back to CPU mode")
            elif "download" in str(e).lower():
                raise WhisperError(
                    "Failed to download Whisper model", audio_path=wav_path
                ).with_suggestion("Check internet connection")
            else:
                raise WhisperError(
                    f"Failed to load Whisper model: {str(e)}", audio_path=wav_path
                )

        # Transcribe with word-level timestamps
        segments, info = model.transcribe(wav_path, word_timestamps=True)

        words = []
        for segment in segments:
            if hasattr(segment, "words") and segment.words:
                for word in segment.words:
                    words.append(
                        {
                            "word": word.word,
                            "start": word.start,
                            "end": word.end,
                            "confidence": word.probability,
                        }
                    )
            else:
                # Fallback: use segment-level data
                segment_words = segment.text.split()
                duration = segment.end - segment.start
                word_duration = duration / len(segment_words) if segment_words else 0

                for i, word in enumerate(segment_words):
                    words.append(
                        {
                            "word": word,
                            "start": segment.start + i * word_duration,
                            "end": segment.start + (i + 1) * word_duration,
                            "confidence": 0.8,  # Default confidence
                        }
                    )

        logger.info(
            "Faster-whisper transcription completed",
            extra={
                "word_count": len(words),
                "model": "base",
                "device": "cpu",
                "job_id": job_id,
            },
        )
        return words

    except MemoryError as e:
        raise MontageMemoryError(
            "Out of memory during Whisper transcription",
            required_mb=2048,  # Whisper base model typically needs ~2GB
            available_mb=0,  # Unknown
        ).with_suggestion("Try processing shorter audio segments")
    except Exception as e:
        if "whisper" not in str(e).lower():
            # Not a Whisper-specific error, wrap it
            raise WhisperError(
                f"Unexpected error during transcription: {str(e)}", audio_path=wav_path
            )
        raise


try:
    from ..core.cost import priced
except ImportError:
    from core.cost import priced


@priced("deepgram.nova-2", Decimal("0.0003"))  # $0.03 per 100 seconds
@with_error_context("Transcription", "deepgram_transcribe")
@with_retry(max_attempts=3, base_delay=2.0, exponential_base=2.0)
def transcribe_deepgram(wav_path: str, job_id: str = "default") -> List[Dict[str, Any]]:
    """Deepgram transcription with comprehensive error handling"""
    if not DEEPGRAM_API_KEY:
        raise ConfigurationError(
            "Deepgram API key not configured", config_key="DEEPGRAM_API_KEY"
        ).with_suggestion("Set DEEPGRAM_API_KEY environment variable")

    # Use rate-limited service if available
    if RATE_LIMITING_AVAILABLE:
        try:
            return graceful_api_call(
                service_func=deepgram_service.transcribe_audio,
                fallback_func=_transcribe_deepgram_direct,
                service_name="deepgram",
                audio_path=wav_path,
                job_id=job_id,
            )
        except Exception as e:
            logger.warning(f"Rate-limited Deepgram failed: {e}, trying direct call")
            # Fall through to direct call

    # Fallback to direct API call
    return _transcribe_deepgram_direct(wav_path, job_id)


def _transcribe_deepgram_direct(
    audio_path: str, job_id: str = "default"
) -> List[Dict[str, Any]]:
    """Direct Deepgram transcription with error handling"""
    wav_path = audio_path  # For compatibility

    if not os.path.exists(wav_path):
        raise DeepgramError(
            "Audio file not found for Deepgram transcription"
        ).add_context(audio_path=wav_path)

    try:
        # Check file size
        file_size = os.path.getsize(wav_path)
        file_size_mb = file_size / (1024 * 1024)

        if file_size_mb > 100:  # Deepgram has file size limits
            raise DeepgramError(
                f"Audio file too large for Deepgram: {file_size_mb:.1f}MB"
            ).with_suggestion("Use chunked transcription for large files")

        # Get duration for chunking decision
        from ..utils.ffmpeg_utils import get_video_info

        info = get_video_info(wav_path)
        duration = info.get("duration", 0)

        if duration > 600:  # 10 minutes - use chunking
            logger.info(
                "Using chunked Deepgram transcription for large file",
                extra={
                    "duration_seconds": duration,
                    "method": "chunked",
                    "job_id": job_id,
                },
            )
            return transcribe_deepgram_chunked(wav_path, job_id, duration)

        logger.info(
            "Starting direct Deepgram transcription",
            extra={"duration_seconds": duration, "method": "direct", "job_id": job_id},
        )

        try:
            deepgram = DeepgramClient(DEEPGRAM_API_KEY)
        except Exception as e:
            raise AuthenticationError(
                "Failed to initialize Deepgram client", service="deepgram"
            ).with_suggestion("Verify your Deepgram API key is valid")

        with open(wav_path, "rb") as audio:
            buffer_data = audio.read()

        payload = {"buffer": buffer_data}
        options = PrerecordedOptions(
            model="nova-2",
            language="en-US",
            smart_format=True,
            punctuate=True,
            diarize=True,
            utterances=True,
        )

        # Make API call with proper error handling
        try:
            response = deepgram.listen.rest.v("1").transcribe_file(payload, options)
        except Exception as e:
            # Check for specific error types
            error_str = str(e)
            status_code = None

            if hasattr(e, "status_code"):
                status_code = e.status_code
            elif "429" in error_str:
                status_code = 429
            elif "401" in error_str or "403" in error_str:
                status_code = 401
            elif "500" in error_str or "502" in error_str or "503" in error_str:
                status_code = 500

            raise handle_api_error(
                service="deepgram",
                error=e,
                status_code=status_code,
                response_body=error_str,
            )

        # Process response
        words = []
        if hasattr(response, "results") and response.results:
            channels = response.results.channels
            if channels and len(channels) > 0:
                channel = channels[0]
                if channel.alternatives and len(channel.alternatives) > 0:
                    alternative = channel.alternatives[0]
                    if hasattr(alternative, "words") and alternative.words:
                        for word in alternative.words:
                            word_data = {
                                "word": word.word,
                                "start": word.start,
                                "end": word.end,
                                "confidence": (
                                    word.confidence
                                    if hasattr(word, "confidence")
                                    else 0.8
                                ),
                            }

                            # Add speaker if diarization is available
                            if hasattr(word, "speaker"):
                                word_data["speaker"] = f"SPEAKER_{word.speaker:02d}"

                            words.append(word_data)

        if not words:
            raise DeepgramError(
                "Deepgram returned empty transcription"
            ).with_suggestion("Check if the audio file contains speech")

        logger.info(
            "Deepgram transcription completed",
            extra={"word_count": len(words), "model": "nova-2", "job_id": job_id},
        )
        return words

    except Exception as e:
        # Re-raise if already a MontageError
        if isinstance(e, MontageError):
            raise

        # Wrap unexpected errors
        raise DeepgramError(
            f"Unexpected error during Deepgram transcription: {str(e)}"
        ).add_context(audio_path=wav_path, job_id=job_id)


@with_error_context("Transcription", "deepgram_chunked")
def transcribe_deepgram_chunked(
    wav_path: str, job_id: str, duration: float
) -> List[Dict[str, Any]]:
    """Chunked Deepgram transcription with comprehensive error handling"""
    import tempfile
    import subprocess

    chunk_duration = 300  # 5 minutes per chunk
    chunks_needed = int(duration / chunk_duration) + 1
    all_words = []
    errors = []

    logger.info(
        "Starting chunked Deepgram transcription",
        extra={
            "total_chunks": chunks_needed,
            "chunk_duration": chunk_duration,
            "total_duration": duration,
            "job_id": job_id,
        },
    )

    for chunk_idx in range(chunks_needed):
        start_time = chunk_idx * chunk_duration
        end_time = min(start_time + chunk_duration, duration)

        if start_time >= duration:
            break

        logger.debug(
            "Processing audio chunk",
            extra={
                "chunk_index": chunk_idx + 1,
                "total_chunks": chunks_needed,
                "start_time": start_time,
                "end_time": end_time,
                "job_id": job_id,
            },
        )

        # Extract chunk to temporary file
        chunk_file = None
        try:
            chunk_file = tempfile.mktemp(suffix=f"_chunk_{chunk_idx}.wav")

            # Use FFmpeg to extract audio segment
            cmd = [
                "ffmpeg",
                "-y",
                "-i",
                wav_path,
                "-ss",
                str(start_time),
                "-t",
                str(end_time - start_time),
                "-ar",
                "16000",
                "-ac",
                "1",
                "-c:a",
                "pcm_s16le",
                chunk_file,
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)

            if result.returncode != 0:
                error = handle_ffmpeg_error(
                    command=cmd,
                    return_code=result.returncode,
                    stderr=result.stderr,
                    operation=f"Extract audio chunk {chunk_idx + 1}",
                )
                errors.append(error)
                logger.warning(
                    f"Failed to extract chunk {chunk_idx + 1}: {error.message}"
                )
                continue

            if not os.path.exists(chunk_file) or os.path.getsize(chunk_file) == 0:
                error = FFmpegError(
                    f"Empty audio chunk {chunk_idx + 1}",
                    command=cmd,
                    stderr="No output produced",
                )
                errors.append(error)
                continue

            # Transcribe chunk
            try:
                chunk_words = _transcribe_deepgram_direct(
                    chunk_file, f"{job_id}_chunk_{chunk_idx}"
                )

                # Adjust timestamps
                for word in chunk_words:
                    word["start"] += start_time
                    word["end"] += start_time
                    all_words.append(word)

                logger.debug(
                    "Audio chunk processing completed",
                    extra={
                        "chunk_index": chunk_idx + 1,
                        "chunk_words": len(chunk_words),
                        "total_words": len(all_words),
                        "job_id": job_id,
                    },
                )

            except Exception as e:
                error = handle_error(e)
                error.add_context(chunk_index=chunk_idx + 1, chunk_start=start_time)
                errors.append(error)
                logger.warning(
                    f"Chunk {chunk_idx + 1} transcription failed: {error.message}"
                )

        finally:
            # Clean up temporary file
            if chunk_file and os.path.exists(chunk_file):
                try:
                    os.unlink(chunk_file)
                except OSError:
                    pass

    # Check if we got any successful transcriptions
    if not all_words and errors:
        # All chunks failed
        from ..utils.error_handler import aggregate_errors

        raise aggregate_errors(errors)
    elif not all_words:
        raise DeepgramError("No words transcribed from any chunk").with_suggestion(
            "Check if the audio contains speech"
        )

    logger.info(
        "Deepgram chunked transcription completed",
        extra={
            "total_words": len(all_words),
            "total_chunks": chunks_needed,
            "failed_chunks": len(errors),
            "job_id": job_id,
        },
    )

    return all_words


# Import the rest of the functions from the original module
from .analyze_video import (
    deduplicate_words,
    final_text_cleanup,
    rover_merge,
    diarize_speakers,
    simple_speaker_segments,
    align_speakers,
)


@with_error_context("Database", "get_cached_transcript")
def get_cached_transcript(file_hash: str) -> Optional[str]:
    """Get cached transcript with proper error handling"""
    try:
        # Skip cache if no database available
        if not DATABASE_URL or DATABASE_URL.startswith("sqlite"):
            return None

        conn = psycopg2.connect(DATABASE_URL)
        cursor = conn.cursor()

        cursor.execute(
            "SELECT transcript FROM transcript_cache WHERE sha256 = %s", (file_hash,)
        )

        result = cursor.fetchone()
        cursor.close()
        conn.close()

        if result:
            logger.info(
                "Retrieved cached transcript",
                extra={"cache_hit": True, "file_hash": file_hash[:8]},
            )
            return result[0]
        return None

    except psycopg2.OperationalError as e:
        # Connection errors - don't fail the whole operation
        logger.warning(f"Database connection failed for cache lookup: {e}")
        return None
    except psycopg2.DatabaseError as e:
        # Query errors - log but continue
        logger.warning(f"Database query failed for cache lookup: {e}")
        return None
    except Exception as e:
        # Unexpected errors - log but continue
        logger.warning(f"Unexpected error during cache lookup: {e}")
        return None


@with_error_context("Database", "cache_transcript")
def cache_transcript(file_hash: str, transcript: str, job_id: Optional[str] = None):
    """Cache transcript with proper error handling"""
    try:
        # Skip cache if no database available
        if not DATABASE_URL or DATABASE_URL.startswith("sqlite"):
            return

        conn = psycopg2.connect(DATABASE_URL)
        cursor = conn.cursor()

        cursor.execute(
            "INSERT INTO transcript_cache (sha256, transcript) VALUES (%s, %s) "
            "ON CONFLICT (sha256) DO UPDATE SET transcript = %s",
            (file_hash, transcript, transcript),
        )

        conn.commit()
        cursor.close()
        conn.close()

        logger.info(
            "Transcript cached successfully",
            extra={"file_hash": file_hash[:8], "job_id": job_id},
        )

    except psycopg2.OperationalError as e:
        # Connection errors - log but don't fail
        logger.warning(f"Failed to cache transcript due to connection error: {e}")
    except psycopg2.DatabaseError as e:
        # Query errors - log but don't fail
        logger.warning(f"Failed to cache transcript due to database error: {e}")
    except Exception as e:
        # Unexpected errors - log but don't fail
        logger.warning(f"Failed to cache transcript due to unexpected error: {e}")


@with_error_context("VideoAnalysis", "analyze_video")
def analyze_video(video_path: str, job_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Video analysis with comprehensive error handling and propagation.

    Args:
        video_path: Path to the video file
        job_id: Optional job ID for tracking

    Returns:
        Dict with analysis results

    Raises:
        VideoDecodeError: If video file is invalid or corrupted
        TranscriptionError: If transcription fails
        DiskSpaceError: If insufficient disk space
        MemoryError: If insufficient memory
    """
    if not os.path.exists(video_path):
        raise VideoDecodeError(
            "Video file not found", video_path=video_path
        ).with_suggestion("Check if the file path is correct")

    # Import memory management (with fallback)
    try:
        from ..utils.memory_manager import memory_guard, get_memory_monitor
        from ..utils.resource_manager import estimate_processing_memory

        memory_management_available = True
    except ImportError:
        memory_management_available = False
        logger.warning("Memory management not available - using basic processing")

    logger.info(
        "Starting video analysis with error handling",
        extra={"video_path": os.path.basename(video_path), "job_id": job_id},
    )

    # Estimate memory requirements
    estimated_memory = 1024  # Default 1GB
    if memory_management_available:
        try:
            estimated_memory = estimate_processing_memory(video_path, "analysis")
        except Exception as e:
            logger.warning(f"Could not estimate memory requirements: {e}")

    # Use memory guard if available
    if memory_management_available:
        memory_context = memory_guard(max_memory_mb=estimated_memory)
    else:
        from contextlib import nullcontext

        memory_context = nullcontext()

    with memory_context:
        # Calculate file hash for caching
        file_hash = sha256_file(video_path)

        # Check cache first
        cached = get_cached_transcript(file_hash)
        if cached:
            import json

            logger.info(
                "Using cached transcript",
                extra={"file_hash": file_hash[:8], "job_id": job_id},
            )
            try:
                return json.loads(cached)
            except json.JSONDecodeError as e:
                logger.warning(f"Invalid cached transcript, regenerating: {e}")

        # Check file size and memory pressure for chunked processing
        file_size_mb = os.path.getsize(video_path) / (1024 * 1024)
        should_use_chunked = file_size_mb > 500  # Files larger than 500MB

        if memory_management_available:
            monitor = get_memory_monitor()
            if monitor:
                stats = monitor.get_current_stats()
                # Use chunked processing if memory pressure is high
                should_use_chunked = (
                    should_use_chunked
                    or stats.pressure_level.value in ["high", "critical"]
                )

        try:
            if should_use_chunked:
                logger.info(
                    f"Using chunked analysis for large video ({file_size_mb:.0f}MB)"
                )
                return _analyze_video_chunked(video_path, file_hash, job_id)
            else:
                logger.info(f"Using direct analysis for video ({file_size_mb:.0f}MB)")
                return _analyze_video_direct(video_path, file_hash, job_id)
        except Exception as e:
            # Add video context to any error
            if isinstance(e, MontageError):
                e.add_context(
                    video_path=video_path, file_hash=file_hash[:8], job_id=job_id
                )
            raise


def _analyze_video_chunked(
    video_path: str, file_hash: str, job_id: Optional[str] = None
) -> Dict[str, Any]:
    """Memory-safe chunked video analysis with error handling"""
    logger.info("Starting chunked video analysis", extra={"job_id": job_id})

    from ..utils.ffmpeg_utils import get_video_info
    from ..utils.error_handler import ErrorCollector

    try:
        info = get_video_info(video_path)
    except Exception as e:
        raise VideoDecodeError(
            f"Failed to get video info: {str(e)}", video_path=video_path
        )

    duration = info.get("duration", 0)
    if duration <= 0:
        raise VideoDecodeError(
            "Could not determine video duration", video_path=video_path
        ).with_suggestion("Video file may be corrupted")

    # Process in 10-minute chunks
    chunk_duration = 600
    chunks_needed = max(1, int(duration / chunk_duration) + 1)

    logger.info(
        f"Processing {duration:.0f}s video in {chunks_needed} chunks",
        extra={"job_id": job_id},
    )

    all_words = []
    all_turns = []
    error_collector = ErrorCollector()

    for chunk_idx in range(chunks_needed):
        start_time = chunk_idx * chunk_duration
        end_time = min(start_time + chunk_duration, duration)

        if start_time >= duration:
            break

        with error_collector.collect(chunk_index=chunk_idx, start_time=start_time):
            logger.info(
                f"Processing chunk {chunk_idx + 1}/{chunks_needed}: {start_time:.0f}s - {end_time:.0f}s",
                extra={"job_id": job_id},
            )

            # Extract audio chunk
            chunk_audio_path = extract_audio_chunk(video_path, start_time, end_time)

            try:
                # Process chunk
                chunk_result = _analyze_audio_chunk(
                    chunk_audio_path, start_time, job_id
                )

                # Adjust timestamps and merge results
                for word in chunk_result.get("words", []):
                    word["start"] += start_time
                    word["end"] += start_time
                    all_words.append(word)

                for turn in chunk_result.get("speaker_turns", []):
                    turn["start"] += start_time
                    turn["end"] += start_time
                    all_turns.append(turn)

            finally:
                # Clean up chunk audio
                if os.path.exists(chunk_audio_path):
                    try:
                        os.unlink(chunk_audio_path)
                    except OSError:
                        pass

            # Force garbage collection between chunks
            import gc

            gc.collect()

    # Check if too many chunks failed
    if error_collector.has_errors():
        failed_count = len(error_collector.get_errors())
        if failed_count > chunks_needed / 2:
            error_collector.raise_if_errors(
                f"Too many chunks failed ({failed_count}/{chunks_needed})"
            )
        else:
            logger.warning(
                f"Some chunks failed ({failed_count}/{chunks_needed}), continuing with partial results",
                extra={"job_id": job_id},
            )

    if not all_words:
        raise TranscriptionError("No words transcribed from any chunk").with_suggestion(
            "Check if the video contains speech"
        )

    # Create final result
    transcript = " ".join([word["word"] for word in all_words])

    result = {
        "sha": file_hash,
        "words": all_words,
        "transcript": transcript,
        "speaker_turns": all_turns,
    }

    # Cache result
    import json

    cache_transcript(file_hash, json.dumps(result), job_id)

    logger.info(
        f"Chunked analysis completed: {len(all_words)} words, {len(all_turns)} speaker turns",
        extra={"job_id": job_id},
    )
    return result


def extract_audio_chunk(video_path: str, start_time: float, end_time: float) -> str:
    """Extract audio chunk with error handling"""
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        output_path = tmp.name

    try:
        duration = end_time - start_time
        stream = ffmpeg.input(video_path, ss=start_time, t=duration)
        stream = ffmpeg.output(
            stream, output_path, acodec="pcm_s16le", ac=1, ar="16000"
        )
        stream = ffmpeg.overwrite_output(stream)

        out, err = ffmpeg.run(stream, capture_stdout=True, capture_stderr=True)

        if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
            raise FFmpegError(
                f"Failed to extract audio chunk",
                command=ffmpeg.compile(stream),
                stderr=err.decode("utf-8") if err else "",
            )

        return output_path

    except ffmpeg.Error as e:
        if os.path.exists(output_path):
            try:
                os.unlink(output_path)
            except OSError:
                pass

        stderr = (
            e.stderr.decode("utf-8") if hasattr(e, "stderr") and e.stderr else str(e)
        )
        raise handle_ffmpeg_error(
            command=ffmpeg.compile(stream) if "stream" in locals() else [],
            return_code=1,
            stderr=stderr,
            operation=f"Extract audio chunk ({start_time:.0f}s-{end_time:.0f}s)",
        )


def _analyze_audio_chunk(
    wav_path: str, time_offset: float = 0.0, job_id: Optional[str] = None
) -> Dict[str, Any]:
    """Analyze a single audio chunk with error handling"""
    from ..utils.error_handler import ErrorCollector

    error_collector = ErrorCollector()
    words = []

    # Try Whisper first
    with error_collector.collect(engine="whisper"):
        words = transcribe_whisper(wav_path, job_id)

    # If Whisper failed or returned no words, try Deepgram
    if not words:
        with error_collector.collect(engine="deepgram"):
            words = transcribe_deepgram(wav_path, job_id)

    # If both failed, check errors
    if not words and error_collector.has_errors():
        error_collector.raise_if_errors("All transcription engines failed")
    elif not words:
        logger.warning("No words transcribed from chunk", extra={"job_id": job_id})
        words = []

    # Simple speaker segmentation for chunks
    turns = safe_execute(
        lambda: simple_speaker_segments(wav_path), default=[], log_errors=True
    )

    # Align speakers with words
    if words and turns:
        words = align_speakers(words, turns)

    return {"words": words, "speaker_turns": turns}


def _analyze_video_direct(
    video_path: str, file_hash: str, job_id: Optional[str] = None
) -> Dict[str, Any]:
    """Direct video analysis with comprehensive error handling"""
    # Extract audio with error handling
    wav_path = extract_audio_to_wav(video_path, job_id)

    try:
        from ..utils.error_handler import ErrorCollector

        # Transcribe with both engines, collecting errors
        error_collector = ErrorCollector()
        fw_words = []
        dg_words = []

        # Try Whisper
        with error_collector.collect(engine="whisper"):
            fw_words = transcribe_whisper(wav_path, job_id)
            fw_words = deduplicate_words(fw_words)

        # Try Deepgram
        with error_collector.collect(engine="deepgram"):
            dg_words = transcribe_deepgram(wav_path, job_id)
            dg_words = deduplicate_words(dg_words)

        # Merge transcriptions if we have any
        if fw_words or dg_words:
            merged_words = rover_merge(fw_words, dg_words)
        else:
            # Both failed - check if we should raise errors
            if error_collector.has_errors():
                error_collector.raise_if_errors("All transcription engines failed")

            # No errors but no words - create minimal transcript
            logger.warning("No words transcribed - using fallback minimal transcript")
            merged_words = [
                {
                    "word": "[No speech detected]",
                    "start": 0.0,
                    "end": 1.0,
                    "confidence": 0.0,
                }
            ]

        # Continue with speaker detection
        deepgram_has_speakers = any("speaker" in word for word in dg_words)

        if deepgram_has_speakers:
            logger.info("Using Deepgram speaker diarization", extra={"job_id": job_id})
            # Extract speaker turns from Deepgram words
            turns = _extract_speaker_turns_from_words(merged_words)
        else:
            # Fallback to simple segmentation
            logger.info("Using fallback speaker segmentation", extra={"job_id": job_id})
            turns = safe_execute(
                lambda: diarize_speakers(wav_path),
                default=[{"speaker": "SPEAKER_00", "start": 0.0, "end": 1.0}],
                log_errors=True,
            )

            # Align speakers with words
            merged_words = align_speakers(merged_words, turns)

        # Create full transcript
        transcript = " ".join([word["word"] for word in merged_words])
        transcript = final_text_cleanup(transcript)

        # Prepare result
        result = {
            "sha": file_hash,
            "words": merged_words,
            "transcript": transcript,
            "speaker_turns": turns,
        }

        # Cache result
        import json

        cache_transcript(file_hash, json.dumps(result), job_id)

        logger.info(
            "Video analysis completed successfully",
            extra={
                "total_words": len(merged_words),
                "speaker_turns": len(turns),
                "file_hash": file_hash[:8],
                "transcript_length": len(transcript),
                "job_id": job_id,
            },
        )
        return result

    finally:
        # Clean up temp audio file
        if os.path.exists(wav_path):
            try:
                os.unlink(wav_path)
            except OSError as e:
                logger.warning(f"Failed to clean up temp audio file: {e}")


def _extract_speaker_turns_from_words(
    words: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Extract speaker turns from words with speaker labels"""
    if not words:
        return []

    turns = []
    current_speaker = None
    current_turn = None

    for word in words:
        speaker = word.get("speaker", "SPEAKER_00")

        if speaker != current_speaker:
            # Save previous turn
            if current_turn:
                turns.append(current_turn)

            # Start new turn
            current_speaker = speaker
            current_turn = {
                "speaker": speaker,
                "start": word["start"],
                "end": word["end"],
            }
        else:
            # Extend current turn
            if current_turn:
                current_turn["end"] = word["end"]

    # Save final turn
    if current_turn:
        turns.append(current_turn)

    # Merge very short turns
    MIN_TURN_DURATION = 2.0
    MERGE_THRESHOLD = 0.5

    merged_turns = []
    for turn in turns:
        duration = turn["end"] - turn["start"]

        if merged_turns and duration < MIN_TURN_DURATION:
            last_turn = merged_turns[-1]
            gap = turn["start"] - last_turn["end"]

            if gap < MERGE_THRESHOLD:
                # Merge with previous turn
                last_turn["end"] = turn["end"]
                if (
                    turn["speaker"] != last_turn["speaker"]
                    and duration > (last_turn["end"] - turn["start"]) / 2
                ):
                    last_turn["speaker"] = turn["speaker"]
                continue

        merged_turns.append(turn)

    return merged_turns
