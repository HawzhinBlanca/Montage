"""Base test utilities and helper classes for Montage tests."""

import os
import json
import time
import tempfile
import subprocess
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch
from contextlib import contextmanager
import hashlib

import numpy as np
import cv2
from pydub import AudioSegment
from pydub.generators import Sine


class VideoTestUtils:
    """Utilities for video testing."""

    @staticmethod
    def create_test_video(
        output_path: str,
        duration: float = 5.0,
        width: int = 1920,
        height: int = 1080,
        fps: float = 30.0,
        codec: str = "libx264",
        audio: bool = True,
        color: Tuple[int, int, int] = (0, 255, 0),
    ) -> Dict[str, Any]:
        """Create a test video with specific parameters."""
        # Create video frames
        frames = []
        num_frames = int(duration * fps)

        for i in range(num_frames):
            frame = np.full((height, width, 3), color, dtype=np.uint8)
            # Add frame number text
            cv2.putText(
                frame,
                f"Frame {i}",
                (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2,
            )
            frames.append(frame)

        # Write video using OpenCV
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        for frame in frames:
            out.write(frame)
        out.release()

        # Add audio if requested
        if audio:
            # Create audio track
            audio_duration_ms = int(duration * 1000)
            tone = Sine(440).to_audio_segment(duration=audio_duration_ms)

            # Save audio to temp file
            audio_temp = tempfile.mktemp(suffix=".wav")
            tone.export(audio_temp, format="wav")

            # Combine audio and video using ffmpeg
            output_temp = tempfile.mktemp(suffix=".mp4")
            cmd = [
                "ffmpeg",
                "-y",
                "-i",
                output_path,
                "-i",
                audio_temp,
                "-c:v",
                "copy",
                "-c:a",
                "aac",
                "-shortest",
                output_temp,
            ]
            subprocess.run(cmd, check=True, capture_output=True)

            # Replace original with audio version
            os.replace(output_temp, output_path)
            os.unlink(audio_temp)

        return {
            "path": output_path,
            "duration": duration,
            "width": width,
            "height": height,
            "fps": fps,
            "codec": codec,
            "has_audio": audio,
            "num_frames": num_frames,
        }

    @staticmethod
    def verify_video_properties(video_path: str) -> Dict[str, Any]:
        """Verify video properties using OpenCV."""
        cap = cv2.VideoCapture(video_path)

        props = {
            "exists": os.path.exists(video_path),
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "fps": cap.get(cv2.CAP_PROP_FPS),
            "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            "duration": cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS),
        }

        cap.release()
        return props

    @staticmethod
    def extract_frame(video_path: str, frame_number: int) -> np.ndarray:
        """Extract a specific frame from video."""
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()
        cap.release()

        if not ret:
            raise ValueError(f"Could not extract frame {frame_number}")

        return frame


class AudioTestUtils:
    """Utilities for audio testing."""

    @staticmethod
    def create_test_audio(
        output_path: str,
        duration: float = 5.0,
        frequency: float = 440.0,
        sample_rate: int = 44100,
        channels: int = 2,
    ) -> Dict[str, Any]:
        """Create a test audio file."""
        duration_ms = int(duration * 1000)

        # Generate tone
        tone = Sine(frequency).to_audio_segment(
            duration=duration_ms, volume=-20.0  # dB
        )

        # Convert to stereo if needed
        if channels == 2 and tone.channels == 1:
            tone = tone.set_channels(2)

        # Export
        tone.export(output_path, format="mp3", bitrate="192k")

        return {
            "path": output_path,
            "duration": duration,
            "frequency": frequency,
            "sample_rate": sample_rate,
            "channels": channels,
        }

    @staticmethod
    def create_speech_audio(
        output_path: str, text: str = "This is a test speech", duration: float = 5.0
    ) -> Dict[str, Any]:
        """Create a mock speech audio file."""
        # For testing, we'll create a modulated tone to simulate speech
        duration_ms = int(duration * 1000)

        # Create base tone
        base_tone = Sine(200).to_audio_segment(duration=duration_ms)

        # Add some variation to simulate speech patterns
        segments = []
        segment_duration = 100  # ms

        for i in range(0, duration_ms, segment_duration):
            freq = 200 + (i % 300)  # Vary frequency
            segment = Sine(freq).to_audio_segment(duration=segment_duration)
            segments.append(segment)

        speech = sum(segments)
        speech = speech.set_frame_rate(44100).set_channels(1)
        speech.export(output_path, format="wav")

        return {"path": output_path, "text": text, "duration": duration}


class MockAPIResponses:
    """Mock responses for external APIs."""

    @staticmethod
    def whisper_transcription(text: str = None, segments: List[Dict] = None):
        """Create a mock Whisper transcription response."""
        if text is None:
            text = "This is a test transcription with multiple segments."

        if segments is None:
            segments = [
                {"text": "This is a test", "start": 0.0, "end": 2.0},
                {"text": "transcription with", "start": 2.0, "end": 4.0},
                {"text": "multiple segments.", "start": 4.0, "end": 6.0},
            ]

        mock_segments = []
        for seg in segments:
            mock_seg = Mock()
            mock_seg.text = seg["text"]
            mock_seg.start = seg["start"]
            mock_seg.end = seg["end"]
            mock_segments.append(mock_seg)

        info = Mock()
        info.language = "en"
        info.duration = segments[-1]["end"] if segments else 6.0

        return (mock_segments, info)

    @staticmethod
    def deepgram_transcription(text: str = None, words: List[Dict] = None):
        """Create a mock Deepgram transcription response."""
        if text is None:
            text = "This is a Deepgram transcription"

        if words is None:
            words = [
                {"word": "This", "start": 0.0, "end": 0.5},
                {"word": "is", "start": 0.5, "end": 0.7},
                {"word": "a", "start": 0.7, "end": 0.8},
                {"word": "Deepgram", "start": 0.8, "end": 1.3},
                {"word": "transcription", "start": 1.3, "end": 2.0},
            ]

        response = Mock()
        response.results.channels = [Mock()]
        response.results.channels[0].alternatives = [Mock()]
        response.results.channels[0].alternatives[0].transcript = text

        mock_words = []
        for w in words:
            mock_word = Mock()
            mock_word.word = w["word"]
            mock_word.start = w["start"]
            mock_word.end = w["end"]
            mock_words.append(mock_word)

        response.results.channels[0].alternatives[0].words = mock_words
        return response

    @staticmethod
    def claude_highlights(highlights: List[Dict] = None):
        """Create a mock Claude highlights response."""
        if highlights is None:
            highlights = [
                {
                    "slug": "introduction",
                    "title": "Opening Introduction",
                    "start_ms": 0,
                    "end_ms": 30000,
                },
                {
                    "slug": "main-topic",
                    "title": "Main Topic Discussion",
                    "start_ms": 30000,
                    "end_ms": 90000,
                },
                {
                    "slug": "conclusion",
                    "title": "Closing Remarks",
                    "start_ms": 90000,
                    "end_ms": 120000,
                },
            ]

        return {"highlights": highlights}


class DatabaseTestUtils:
    """Utilities for database testing."""

    @staticmethod
    def create_test_schema(connection):
        """Create test database schema."""
        schema = """
        CREATE TABLE IF NOT EXISTS video_job (
            id SERIAL PRIMARY KEY,
            video_path TEXT NOT NULL,
            status TEXT DEFAULT 'pending',
            priority INTEGER DEFAULT 5,
            metadata JSONB DEFAULT '{}',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        CREATE TABLE IF NOT EXISTS transcript_cache (
            id SERIAL PRIMARY KEY,
            video_hash TEXT UNIQUE NOT NULL,
            transcript JSONB NOT NULL,
            metadata JSONB DEFAULT '{}',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        CREATE TABLE IF NOT EXISTS api_cost_log (
            id SERIAL PRIMARY KEY,
            service TEXT NOT NULL,
            model TEXT,
            cost_usd DECIMAL(10, 6),
            tokens_used INTEGER,
            job_id INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        CREATE TABLE IF NOT EXISTS job_checkpoint (
            id SERIAL PRIMARY KEY,
            job_id INTEGER NOT NULL,
            stage TEXT NOT NULL,
            data JSONB,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """

        with connection.cursor() as cur:
            cur.execute(schema)
        connection.commit()

    @staticmethod
    def insert_test_job(connection, **kwargs) -> int:
        """Insert a test job and return its ID."""
        data = {
            "video_path": kwargs.get("video_path", "/test/video.mp4"),
            "status": kwargs.get("status", "pending"),
            "priority": kwargs.get("priority", 5),
            "metadata": json.dumps(kwargs.get("metadata", {})),
        }

        with connection.cursor() as cur:
            cur.execute(
                """
                INSERT INTO video_job (video_path, status, priority, metadata)
                VALUES (%(video_path)s, %(status)s, %(priority)s, %(metadata)s)
                RETURNING id
            """,
                data,
            )
            job_id = cur.fetchone()[0]

        connection.commit()
        return job_id


class PerformanceTestUtils:
    """Utilities for performance testing."""

    @staticmethod
    @contextmanager
    def measure_time(name: str = "Operation"):
        """Context manager to measure execution time."""
        start = time.time()
        yield
        duration = time.time() - start
        print(f"{name} took {duration:.3f} seconds")

    @staticmethod
    @contextmanager
    def measure_memory():
        """Context manager to measure memory usage."""
        import psutil

        process = psutil.Process()

        start_memory = process.memory_info().rss / 1024 / 1024  # MB
        yield
        end_memory = process.memory_info().rss / 1024 / 1024  # MB

        print(
            f"Memory usage: {start_memory:.1f}MB -> {end_memory:.1f}MB "
            f"(+{end_memory - start_memory:.1f}MB)"
        )

    @staticmethod
    def generate_large_video_metadata(num_segments: int = 10000) -> Dict[str, Any]:
        """Generate metadata for a large video with many segments."""
        segments = []
        current_time = 0.0

        for i in range(num_segments):
            duration = np.random.uniform(0.5, 3.0)
            segments.append(
                {
                    "text": f"Segment {i} text content",
                    "start": current_time,
                    "end": current_time + duration,
                    "speaker": f"SPEAKER_{i % 5:02d}",
                }
            )
            current_time += duration

        return {"segments": segments, "duration": current_time, "language": "en"}


class IntegrationTestBase:
    """Base class for integration tests."""

    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp(prefix="montage_test_")
        self.test_outputs = []

    def teardown_method(self):
        """Clean up test environment."""
        # Clean up test outputs
        for path in self.test_outputs:
            if os.path.exists(path):
                os.unlink(path)

        # Clean up temp directory
        if os.path.exists(self.temp_dir):
            import shutil

            shutil.rmtree(self.temp_dir)

    def create_test_output_path(self, filename: str) -> str:
        """Create a test output path that will be cleaned up."""
        path = os.path.join(self.temp_dir, filename)
        self.test_outputs.append(path)
        return path

    @contextmanager
    def mock_external_apis(self):
        """Mock all external API calls."""
        with patch("faster_whisper.WhisperModel") as mock_whisper, patch(
            "deepgram.DeepgramClient"
        ) as mock_deepgram, patch("anthropic.Anthropic") as mock_anthropic, patch(
            "openai.OpenAI"
        ) as mock_openai:

            # Configure mocks
            mock_whisper.return_value.transcribe.return_value = (
                MockAPIResponses.whisper_transcription()
            )

            mock_deepgram.return_value.listen.prerecorded.v.return_value = (
                MockAPIResponses.deepgram_transcription()
            )

            mock_anthropic.return_value.messages.create.return_value = Mock(
                content=[Mock(text=json.dumps(MockAPIResponses.claude_highlights()))],
                usage=Mock(input_tokens=100, output_tokens=50),
            )

            yield {
                "whisper": mock_whisper,
                "deepgram": mock_deepgram,
                "anthropic": mock_anthropic,
                "openai": mock_openai,
            }


def assert_video_properties(video_path: str, expected: Dict[str, Any]):
    """Assert that a video has expected properties."""
    props = VideoTestUtils.verify_video_properties(video_path)

    for key, expected_value in expected.items():
        actual_value = props.get(key)
        if isinstance(expected_value, float):
            # Allow small differences for floating point values
            assert (
                abs(actual_value - expected_value) < 0.1
            ), f"{key}: expected {expected_value}, got {actual_value}"
        else:
            assert (
                actual_value == expected_value
            ), f"{key}: expected {expected_value}, got {actual_value}"


def create_mock_process(returncode: int = 0, stdout: bytes = b"", stderr: bytes = b""):
    """Create a mock subprocess."""
    process = Mock()
    process.returncode = returncode
    process.stdout = stdout
    process.stderr = stderr
    process.poll.return_value = returncode
    process.communicate.return_value = (stdout, stderr)
    process.wait.return_value = returncode
    return process
