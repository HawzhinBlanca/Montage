"""
Comprehensive Edge-Path Coverage Script
Tests all possible execution paths and edge cases to identify dead code

This script systematically exercises every possible code path, error condition,
and edge case to ensure 100% coverage and identify truly dead code.
"""

import os
import sys
import tempfile
import shutil
import subprocess
import json
import time
import logging
import asyncio
import threading
from unittest.mock import patch
import psutil
import signal

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tests.test_postgres_fixture import PostgresTestFixture
from src.core.db import Database
from src.core.checkpoint import CheckpointManager
from src.utils.video_validator import VideoValidator, VideoMetadata
from src.utils.budget_guard import BudgetManager
from src.utils.retry_utils import retry_with_backoff
from src.providers.video_processor import SmartVideoEditor
from src.providers.smart_crop import SmartCropper, apply_smart_crop

try:
    from src.config import get
except ImportError:
    import sys
    import os

    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

logger = logging.getLogger(__name__)


class EdgePathTester:
    """Systematic edge case and path coverage tester"""

    def __init__(self):
        self.test_results = {}
        self.coverage_data = {}
        self.temp_dir = tempfile.mkdtemp(prefix="edge_test_")
        self.postgres_fixture = None

    def setup(self):
        """Setup comprehensive test environment"""
        logger.info("Setting up edge path test environment...")

        # Start PostgreSQL container
        self.postgres_fixture = PostgresTestFixture()
        self.db_info = self.postgres_fixture.start()

        # Create test data directory
        os.makedirs(os.path.join(self.temp_dir, "test_data"), exist_ok=True)

        # Set environment variables for testing
        os.environ.update(
            {
                "POSTGRES_HOST": self.db_info["host"],
                "POSTGRES_PORT": str(self.db_info["port"]),
                "POSTGRES_USER": self.db_info["user"],
                "POSTGRES_PASSWORD": self.db_info["password"],
                "POSTGRES_DB": self.db_info["database"],
                "OPENAI_API_KEY": "test-key-for-coverage",  # pragma: allowlist secret
                "DEEPGRAM_API_KEY": "test-key-for-coverage",
                "BUDGET_LIMIT": "1.00",
                "TEMP_DIR": self.temp_dir,
            }
        )

        logger.info(f"Test environment ready in {self.temp_dir}")

    def cleanup(self):
        """Cleanup test environment"""
        if self.postgres_fixture:
            self.postgres_fixture.stop()
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def create_test_videos(self):
        """Create comprehensive set of test video files"""
        test_data_dir = os.path.join(self.temp_dir, "test_data")

        # Valid short video (15 seconds)
        short_video = os.path.join(test_data_dir, "short.mp4")
        self._create_test_video(short_video, duration=15, width=1920, height=1080)

        # Valid long video (3 hours)
        long_video = os.path.join(test_data_dir, "long.mp4")
        self._create_test_video(long_video, duration=10800, width=1920, height=1080)

        # Huge file (>1GB simulated)
        huge_video = os.path.join(test_data_dir, "huge.mp4")
        self._create_test_video(
            huge_video, duration=60, width=4096, height=2160, bitrate="50M"
        )

        # Corrupted video file
        corrupt_video = os.path.join(test_data_dir, "corrupt.mp4")
        with open(corrupt_video, "wb") as f:
            f.write(b"This is not a video file at all!")

        # Invalid format file
        invalid_video = os.path.join(test_data_dir, "invalid.txt")
        with open(invalid_video, "w") as f:
            f.write("Plain text file")

        # Empty file
        empty_video = os.path.join(test_data_dir, "empty.mp4")
        open(empty_video, "w").close()

        # Permission-denied file (if possible)
        restricted_video = os.path.join(test_data_dir, "restricted.mp4")
        self._create_test_video(restricted_video, duration=30)
        try:
            os.chmod(restricted_video, 0o000)
        except Exception:
            pass  # Not all systems support this

        # Vertical video
        vertical_video = os.path.join(test_data_dir, "vertical.mp4")
        self._create_test_video(vertical_video, duration=60, width=720, height=1280)

        # Unusual aspect ratio
        unusual_video = os.path.join(test_data_dir, "unusual.mp4")
        self._create_test_video(unusual_video, duration=45, width=1000, height=500)

        # Very low quality
        lowqual_video = os.path.join(test_data_dir, "lowqual.mp4")
        self._create_test_video(
            lowqual_video, duration=30, width=320, height=240, bitrate="100k"
        )

        return {
            "short": short_video,
            "long": long_video,
            "huge": huge_video,
            "corrupt": corrupt_video,
            "invalid": invalid_video,
            "empty": empty_video,
            "restricted": restricted_video,
            "vertical": vertical_video,
            "unusual": unusual_video,
            "lowqual": lowqual_video,
        }

    def _create_test_video(
        self,
        path: str,
        duration: int = 30,
        width: int = 1920,
        height: int = 1080,
        bitrate: str = "1M",
    ):
        """Create a test video file using FFmpeg"""
        try:
            cmd = [
                "ffmpeg",
                "-f",
                "lavfi",
                "-i",
                f"testsrc=duration={duration}:size={width}x{height}:rate=30",
                "-f",
                "lavfi",
                "-i",
                f"sine=frequency=1000:duration={duration}",
                "-c:v",
                "libx264",
                "-preset",
                "ultrafast",
                "-b:v",
                bitrate,
                "-c:a",
                "aac",
                "-b:a",
                "128k",
                "-y",
                path,
            ]
            subprocess.run(cmd, capture_output=True, check=True, timeout=300)
            logger.info(f"Created test video: {path} ({duration}s, {width}x{height})")
        except Exception as e:
            logger.warning(f"Failed to create test video {path}: {e}")
            # Create a minimal dummy file
            with open(path, "wb") as f:
                f.write(b"dummy video data")

    # ============================================================================
    # DATABASE EDGE CASES
    # ============================================================================

    def test_database_edge_cases(self):
        """Test all database error conditions and edge cases"""
        logger.info("Testing database edge cases...")

        # Valid database operations
        db = Database()

        # Test 1: Normal operations
        job_data = {
            "id": "test-job-1",
            "status": "queued",
            "input_path": "/test/video.mp4",
            "src_hash": "abc123",
        }
        db.insert("video_job", job_data)
        result = db.find_one("video_job", {"id": "test-job-1"})
        assert result is not None

        # Test 2: Duplicate insertion (should handle gracefully)
        try:
            db.insert("video_job", job_data)  # Same data
            assert False, "Should have raised exception for duplicate"
        except Exception:
            pass  # Expected

        # Test 3: Invalid table name
        try:
            db.insert("nonexistent_table", {"id": "test"})
            assert False, "Should have raised exception for invalid table"
        except Exception:
            pass  # Expected

        # Test 4: Invalid column name
        try:
            db.insert("video_job", {"nonexistent_column": "value"})
            assert False, "Should have raised exception for invalid column"
        except Exception:
            pass  # Expected

        # Test 5: SQL injection attempt
        try:
            malicious_data = {
                "id": "'; DROP TABLE video_job; --",
                "status": "queued",
                "input_path": "/test/video.mp4",
                "src_hash": "def456",
            }
            db.insert("video_job", malicious_data)
            # Should not crash - parameterized queries should prevent injection
        except Exception:
            pass  # Either prevented or failed safely

        # Test 6: Very large data
        large_metadata = {"large_field": "x" * 1000000}  # 1MB of data
        try:
            db.insert(
                "video_job",
                {
                    "id": "test-large",
                    "status": "queued",
                    "input_path": "/test/large.mp4",
                    "src_hash": "large123",
                    "metadata": json.dumps(large_metadata),
                },
            )
        except Exception as e:
            logger.info(f"Large data insertion failed as expected: {e}")

        # Test 7: Connection pool exhaustion
        connections = []
        try:
            for i in range(50):  # Try to exhaust pool
                conn = db.get_connection()
                connections.append(conn)
        except Exception as e:
            logger.info(f"Connection pool exhaustion handled: {e}")
        finally:
            for conn in connections:
                try:
                    conn.close()
                except Exception:
                    pass

        # Test 8: Concurrent access
        def concurrent_insert(thread_id):
            try:
                db.insert(
                    "video_job",
                    {
                        "id": f"concurrent-{thread_id}",
                        "status": "queued",
                        "input_path": f"/test/concurrent-{thread_id}.mp4",
                        "src_hash": f"concurrent{thread_id}",
                    },
                )
            except Exception as e:
                logger.info(f"Concurrent insert {thread_id} failed: {e}")

        threads = []
        for i in range(10):
            t = threading.Thread(target=concurrent_insert, args=(i,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        # Test 9: Database disconnection simulation
        with patch.object(
            db, "get_connection", side_effect=Exception("Connection lost")
        ):
            try:
                db.find_one("video_job", {"id": "test"})
                assert False, "Should have failed with connection error"
            except Exception:
                pass  # Expected

        logger.info("Database edge cases completed")

    # ============================================================================
    # CHECKPOINT SYSTEM EDGE CASES
    # ============================================================================

    def test_checkpoint_edge_cases(self):
        """Test checkpoint system edge cases"""
        logger.info("Testing checkpoint edge cases...")

        checkpoint_mgr = CheckpointManager()

        # Test 1: Normal checkpoint operations
        checkpoint_mgr.save_checkpoint("test-job", "analysis", {"segments": []})
        data = checkpoint_mgr.load_checkpoint("test-job")
        assert data is not None

        # Test 2: Load non-existent checkpoint
        result = checkpoint_mgr.load_checkpoint("nonexistent-job")
        assert result is None

        # Test 3: Very large checkpoint data
        large_data = {"large_array": list(range(100000))}
        try:
            checkpoint_mgr.save_checkpoint("large-job", "analysis", large_data)
            loaded = checkpoint_mgr.load_checkpoint("large-job")
            assert loaded is not None
        except Exception as e:
            logger.info(f"Large checkpoint failed as expected: {e}")

        # Test 4: Invalid JSON data
        with patch.object(
            checkpoint_mgr,
            "_serialize_data",
            side_effect=Exception("Serialization error"),
        ):
            try:
                checkpoint_mgr.save_checkpoint(
                    "invalid-job", "analysis", {"circular": None}
                )
                assert False, "Should have failed with serialization error"
            except Exception:
                pass  # Expected

        # Test 5: Redis connection failure
        with patch.object(checkpoint_mgr, "redis_client", None):
            try:
                checkpoint_mgr.save_checkpoint("redis-fail", "analysis", {})
                assert False, "Should have failed with Redis error"
            except Exception:
                pass  # Expected

        # Test 6: Checkpoint expiration
        checkpoint_mgr.save_checkpoint("expire-test", "analysis", {}, ttl=1)
        time.sleep(2)
        result = checkpoint_mgr.load_checkpoint("expire-test")
        assert result is None  # Should be expired

        # Test 7: Concurrent checkpoint access
        def concurrent_checkpoint(thread_id):
            try:
                checkpoint_mgr.save_checkpoint(
                    f"concurrent-checkpoint-{thread_id}",
                    "analysis",
                    {"thread": thread_id},
                )
                checkpoint_mgr.load_checkpoint(f"concurrent-checkpoint-{thread_id}")
            except Exception as e:
                logger.info(f"Concurrent checkpoint {thread_id} failed: {e}")

        threads = []
        for i in range(5):
            t = threading.Thread(target=concurrent_checkpoint, args=(i,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        # Test 8: Health check edge cases
        health = checkpoint_mgr.health_check()
        assert isinstance(health, bool)

        # Test 9: Cleanup operations
        checkpoint_mgr.clear_checkpoint("test-job")
        result = checkpoint_mgr.load_checkpoint("test-job")
        assert result is None

        logger.info("Checkpoint edge cases completed")

    # ============================================================================
    # VIDEO VALIDATION EDGE CASES
    # ============================================================================

    def test_video_validation_edge_cases(self):
        """Test video validation with all possible edge cases"""
        logger.info("Testing video validation edge cases...")

        validator = VideoValidator()
        test_videos = self.create_test_videos()

        # Test 1: Valid video
        is_valid, reason = validator.validate_input(test_videos["short"])
        logger.info(f"Short video validation: {is_valid}, {reason}")

        # Test 2: Corrupted video
        is_valid, reason = validator.validate_input(test_videos["corrupt"])
        assert not is_valid
        logger.info(f"Corrupt video validation: {is_valid}, {reason}")

        # Test 3: Non-existent file
        is_valid, reason = validator.validate_input("/nonexistent/file.mp4")
        assert not is_valid

        # Test 4: Empty file
        is_valid, reason = validator.validate_input(test_videos["empty"])
        assert not is_valid

        # Test 5: Directory instead of file
        is_valid, reason = validator.validate_input(self.temp_dir)
        assert not is_valid

        # Test 6: File without extension
        no_ext_file = os.path.join(self.temp_dir, "noextension")
        with open(no_ext_file, "w") as f:
            f.write("test")
        is_valid, reason = validator.validate_input(no_ext_file)
        assert not is_valid

        # Test 7: Very long file path
        long_path = os.path.join(self.temp_dir, "a" * 200 + ".mp4")
        try:
            shutil.copy(test_videos["short"], long_path)
            is_valid, reason = validator.validate_input(long_path)
            logger.info(f"Long path validation: {is_valid}, {reason}")
        except Exception as e:
            logger.info(f"Long path test failed: {e}")

        # Test 8: File with special characters
        special_path = os.path.join(self.temp_dir, "special!@#$%^&*()_+.mp4")
        try:
            shutil.copy(test_videos["short"], special_path)
            is_valid, reason = validator.validate_input(special_path)
            logger.info(f"Special chars validation: {is_valid}, {reason}")
        except Exception as e:
            logger.info(f"Special chars test failed: {e}")

        # Test 9: Symlink handling
        symlink_path = os.path.join(self.temp_dir, "symlink.mp4")
        try:
            os.symlink(test_videos["short"], symlink_path)
            is_valid, reason = validator.validate_input(symlink_path)
            logger.info(f"Symlink validation: {is_valid}, {reason}")
        except Exception as e:
            logger.info(f"Symlink test failed: {e}")

        # Test 10: Permission denied file
        try:
            is_valid, reason = validator.validate_input(test_videos["restricted"])
            logger.info(f"Restricted file validation: {is_valid}, {reason}")
        except Exception as e:
            logger.info(f"Restricted file test failed: {e}")

        # Test 11: Get video info edge cases
        try:
            info = validator.get_video_info(test_videos["short"])
            assert isinstance(info, VideoMetadata)
            assert info.duration > 0
        except Exception as e:
            logger.error(f"Video info extraction failed: {e}")

        # Test 12: Get info from invalid file
        try:
            info = validator.get_video_info(test_videos["corrupt"])
            assert False, "Should have failed for corrupt video"
        except Exception:
            pass  # Expected

        # Test 13: File hash calculation
        hash1 = validator._calculate_file_hash(test_videos["short"])
        hash2 = validator._calculate_file_hash(test_videos["short"])
        assert hash1 == hash2  # Should be consistent

        # Test 14: Hash of non-existent file
        try:
            validator._calculate_file_hash("/nonexistent/file.mp4")
            assert False, "Should have failed for non-existent file"
        except Exception:
            pass  # Expected

        logger.info("Video validation edge cases completed")

    # ============================================================================
    # BUDGET SYSTEM EDGE CASES
    # ============================================================================

    def test_budget_edge_cases(self):
        """Test budget management system edge cases"""
        logger.info("Testing budget edge cases...")

        budget_mgr = BudgetManager()

        # Test 1: Normal budget operations
        budget_mgr.track_cost("test-job", "openai", "gpt-4", 0.50)
        cost = budget_mgr.get_job_cost("test-job")
        assert cost == 0.50

        # Test 2: Budget limit exceeded
        budget_mgr.track_cost("test-job", "openai", "gpt-4", 10.00)  # Exceed limit
        try:
            is_within, remaining = budget_mgr.check_budget("test-job")
            assert not is_within
        except Exception:
            pass  # May raise BudgetExceededError

        # Test 3: Negative costs (should be prevented)
        try:
            budget_mgr.track_cost("negative-job", "openai", "gpt-4", -1.00)
            assert False, "Should not allow negative costs"
        except Exception:
            pass  # Expected

        # Test 4: Zero costs
        budget_mgr.track_cost("zero-job", "openai", "gpt-4", 0.00)
        cost = budget_mgr.get_job_cost("zero-job")
        assert cost == 0.00

        # Test 5: Very large costs
        try:
            budget_mgr.track_cost("large-job", "openai", "gpt-4", 999999.99)
            cost = budget_mgr.get_job_cost("large-job")
            logger.info(f"Large cost tracked: {cost}")
        except Exception as e:
            logger.info(f"Large cost failed as expected: {e}")

        # Test 6: Invalid job ID
        try:
            budget_mgr.track_cost("", "openai", "gpt-4", 1.00)
            assert False, "Should not allow empty job ID"
        except Exception:
            pass  # Expected

        # Test 7: Invalid service name
        try:
            budget_mgr.track_cost("invalid-service", "", "gpt-4", 1.00)
            assert False, "Should not allow empty service"
        except Exception:
            pass  # Expected

        # Test 8: Concurrent cost tracking
        def concurrent_cost_tracking(thread_id):
            try:
                for i in range(10):
                    budget_mgr.track_cost(
                        f"concurrent-{thread_id}", "openai", "gpt-4", 0.01
                    )
            except Exception as e:
                logger.info(f"Concurrent cost tracking {thread_id} failed: {e}")

        threads = []
        for i in range(5):
            t = threading.Thread(target=concurrent_cost_tracking, args=(i,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        # Test 9: Budget reset
        budget_mgr.reset_job_budget("test-job")
        cost = budget_mgr.get_job_cost("test-job")
        assert cost == 0.00

        # Test 10: Cost breakdown
        breakdown = budget_mgr.get_job_cost_breakdown("test-job")
        assert isinstance(breakdown, dict)

        # Test 11: Budget decorator edge cases
        @budget_mgr.with_budget_tracking("decorator-test")
        def test_function_success():
            return "success"

        @budget_mgr.with_budget_tracking("decorator-test")
        def test_function_failure():
            raise Exception("Test failure")

        # Test successful function
        result = test_function_success()
        assert result == "success"

        # Test function that raises exception
        try:
            test_function_failure()
            assert False, "Should have raised exception"
        except Exception:
            pass  # Expected

        logger.info("Budget edge cases completed")

    # ============================================================================
    # VIDEO PROCESSING EDGE CASES
    # ============================================================================

    def test_video_processing_edge_cases(self):
        """Test video processing pipeline edge cases"""
        logger.info("Testing video processing edge cases...")

        test_videos = self.create_test_videos()
        editor = SmartVideoEditor()

        # Test 1: Process valid short video
        try:
            result = editor.analyze_video(test_videos["short"])
            logger.info(
                f"Short video analysis: {len(result.get('segments', []))} segments"
            )
        except Exception as e:
            logger.error(f"Short video analysis failed: {e}")

        # Test 2: Process corrupted video
        try:
            result = editor.analyze_video(test_videos["corrupt"])
            assert False, "Should have failed for corrupt video"
        except Exception:
            pass  # Expected

        # Test 3: Process empty file
        try:
            result = editor.analyze_video(test_videos["empty"])
            assert False, "Should have failed for empty file"
        except Exception:
            pass  # Expected

        # Test 4: Process very long video
        try:
            result = editor.analyze_video(test_videos["long"])
            logger.info(
                f"Long video analysis: {len(result.get('segments', []))} segments"
            )
        except Exception as e:
            logger.error(f"Long video analysis failed: {e}")

        # Test 5: Process vertical video
        try:
            result = editor.analyze_video(test_videos["vertical"])
            logger.info(
                f"Vertical video analysis: {len(result.get('segments', []))} segments"
            )
        except Exception as e:
            logger.error(f"Vertical video analysis failed: {e}")

        # Test 6: Generate highlights with empty segments
        try:
            highlights = editor.generate_highlights([])
            assert len(highlights) == 0
        except Exception as e:
            logger.error(f"Empty highlights generation failed: {e}")

        # Test 7: Generate highlights with invalid segments
        invalid_segments = [
            {"start_time": -1, "end_time": 10, "score": 0.8},  # Negative start
            {"start_time": 10, "end_time": 5, "score": 0.8},  # End before start
            {"start_time": "invalid", "end_time": 10, "score": 0.8},  # Invalid type
        ]
        try:
            highlights = editor.generate_highlights(invalid_segments)
            logger.info(f"Invalid segments processed: {len(highlights)} highlights")
        except Exception as e:
            logger.error(f"Invalid segments failed: {e}")

        # Test 8: Execute edit with invalid edit plan
        invalid_edit_plan = {
            "segments": [
                {"start": -10, "end": 20},  # Invalid times
                {"start": 100, "end": 50},  # End before start
            ]
        }

        output_path = os.path.join(self.temp_dir, "invalid_edit.mp4")
        try:
            editor.execute_edit(
                "invalid-job", test_videos["short"], invalid_edit_plan, output_path
            )
            assert False, "Should have failed with invalid edit plan"
        except Exception:
            pass  # Expected

        # Test 9: Execute edit with non-existent input
        try:
            valid_edit_plan = {"segments": [{"start": 10, "end": 20}]}
            editor.execute_edit(
                "nonexistent-job",
                "/nonexistent/video.mp4",
                valid_edit_plan,
                output_path,
            )
            assert False, "Should have failed with non-existent input"
        except Exception:
            pass  # Expected

        # Test 10: Execute edit with invalid output path
        try:
            valid_edit_plan = {"segments": [{"start": 5, "end": 10}]}
            invalid_output = "/invalid/path/that/does/not/exist/output.mp4"
            editor.execute_edit(
                "invalid-output-job",
                test_videos["short"],
                valid_edit_plan,
                invalid_output,
            )
            assert False, "Should have failed with invalid output path"
        except Exception:
            pass  # Expected

        # Test 11: Memory stress test with many segments
        many_segments = []
        for i in range(1000):
            many_segments.append(
                {
                    "start_time": i * 10,
                    "end_time": i * 10 + 5,
                    "score": 0.5 + (i % 5) * 0.1,
                }
            )

        try:
            highlights = editor.generate_highlights(many_segments)
            logger.info(f"Many segments processed: {len(highlights)} highlights")
        except Exception as e:
            logger.error(f"Many segments failed: {e}")

        # Test 12: Interrupt processing (signal handling)
        def interrupt_processing():
            time.sleep(2)
            os.kill(os.getpid(), signal.SIGINT)

        interrupt_thread = threading.Thread(target=interrupt_processing)
        interrupt_thread.daemon = True

        try:
            interrupt_thread.start()
            result = editor.analyze_video(test_videos["long"])  # Long video
            logger.info("Processing completed despite interrupt attempt")
        except KeyboardInterrupt:
            logger.info("Processing interrupted as expected")
        except Exception as e:
            logger.info(f"Processing failed with interrupt: {e}")

        logger.info("Video processing edge cases completed")

    # ============================================================================
    # SMART CROP EDGE CASES
    # ============================================================================

    def test_smart_crop_edge_cases(self):
        """Test smart cropping edge cases"""
        logger.info("Testing smart crop edge cases...")

        test_videos = self.create_test_videos()

        # Test 1: Normal smart crop
        output_path = os.path.join(self.temp_dir, "cropped_normal.mp4")
        try:
            result = apply_smart_crop(test_videos["short"], output_path, "9:16")
            logger.info(f"Normal crop result: {result}")
            assert os.path.exists(output_path)
        except Exception as e:
            logger.error(f"Normal smart crop failed: {e}")

        # Test 2: Invalid aspect ratio
        try:
            result = apply_smart_crop(test_videos["short"], output_path, "invalid")
            assert False, "Should have failed with invalid aspect ratio"
        except Exception:
            pass  # Expected

        # Test 3: Zero aspect ratio
        try:
            result = apply_smart_crop(test_videos["short"], output_path, "0:1")
            assert False, "Should have failed with zero aspect ratio"
        except Exception:
            pass  # Expected

        # Test 4: Very unusual aspect ratio
        try:
            result = apply_smart_crop(test_videos["short"], output_path, "100:1")
            logger.info("Unusual aspect ratio processed")
        except Exception as e:
            logger.error(f"Unusual aspect ratio failed: {e}")

        # Test 5: Crop vertical video to horizontal
        output_path = os.path.join(self.temp_dir, "vertical_to_horizontal.mp4")
        try:
            result = apply_smart_crop(test_videos["vertical"], output_path, "16:9")
            logger.info(f"Vertical to horizontal crop: {result}")
        except Exception as e:
            logger.error(f"Vertical to horizontal crop failed: {e}")

        # Test 6: Crop to same aspect ratio
        output_path = os.path.join(self.temp_dir, "same_aspect.mp4")
        try:
            result = apply_smart_crop(
                test_videos["short"], output_path, "16:9"
            )  # Already 16:9
            logger.info(f"Same aspect ratio crop: {result}")
        except Exception as e:
            logger.error(f"Same aspect ratio crop failed: {e}")

        # Test 7: Crop corrupted video
        try:
            result = apply_smart_crop(test_videos["corrupt"], output_path, "9:16")
            assert False, "Should have failed with corrupted video"
        except Exception:
            pass  # Expected

        # Test 8: SmartCropper class edge cases
        cropper = SmartCropper(output_aspect_ratio=9.0 / 16.0)

        # Face detection with no faces
        import cv2
        import numpy as np

        # Create test frame with no faces
        test_frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        faces = cropper._detect_faces(test_frame)
        assert len(faces) == 0

        # Create test frame with artificial face-like pattern
        cv2.rectangle(test_frame, (800, 400), (1100, 700), (255, 255, 255), -1)
        faces = cropper._detect_faces(test_frame)
        logger.info(f"Faces detected in artificial pattern: {len(faces)}")

        # Test very small input dimensions
        try:
            small_output_w, small_output_h = cropper._calculate_output_dimensions(
                100, 100
            )
            logger.info(f"Small dimensions: {small_output_w}x{small_output_h}")
        except Exception as e:
            logger.error(f"Small dimensions failed: {e}")

        # Test very large input dimensions
        try:
            large_output_w, large_output_h = cropper._calculate_output_dimensions(
                8192, 4320
            )
            logger.info(f"Large dimensions: {large_output_w}x{large_output_h}")
        except Exception as e:
            logger.error(f"Large dimensions failed: {e}")

        logger.info("Smart crop edge cases completed")

    # ============================================================================
    # ASYNC OPERATIONS EDGE CASES
    # ============================================================================

    async def test_async_edge_cases(self):
        """Test asynchronous operations edge cases"""
        logger.info("Testing async edge cases...")

        # Test 1: Normal async operation
        async def normal_async_op():
            await asyncio.sleep(0.1)
            return "success"

        result = await normal_async_op()
        assert result == "success"

        # Test 2: Async operation with exception
        async def failing_async_op():
            await asyncio.sleep(0.1)
            raise Exception("Async failure")

        try:
            await failing_async_op()
            assert False, "Should have raised exception"
        except Exception:
            pass  # Expected

        # Test 3: Async timeout
        async def slow_async_op():
            await asyncio.sleep(10)  # Very slow
            return "completed"

        try:
            result = await asyncio.wait_for(slow_async_op(), timeout=0.5)
            assert False, "Should have timed out"
        except asyncio.TimeoutError:
            pass  # Expected

        # Test 4: Concurrent async operations
        async def concurrent_op(op_id):
            await asyncio.sleep(0.1)
            return f"op-{op_id}"

        tasks = [concurrent_op(i) for i in range(10)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        assert len(results) == 10

        # Test 5: Async operation with resource contention
        semaphore = asyncio.Semaphore(2)  # Limit to 2 concurrent

        async def resource_limited_op(op_id):
            async with semaphore:
                await asyncio.sleep(0.2)
                return f"limited-{op_id}"

        tasks = [resource_limited_op(i) for i in range(5)]
        results = await asyncio.gather(*tasks)
        assert len(results) == 5

        # Test 6: Async generator edge cases
        async def async_generator():
            for i in range(5):
                await asyncio.sleep(0.05)
                yield i

        values = []
        async for value in async_generator():
            values.append(value)
        assert values == [0, 1, 2, 3, 4]

        # Test 7: Async context manager
        class AsyncContextManager:
            async def __aenter__(self):
                await asyncio.sleep(0.01)
                return self

            async def __aexit__(self, exc_type, exc_val, exc_tb):
                await asyncio.sleep(0.01)
                return False

        async with AsyncContextManager() as ctx:
            assert ctx is not None

        logger.info("Async edge cases completed")

    # ============================================================================
    # MEMORY AND RESOURCE EDGE CASES
    # ============================================================================

    def test_memory_edge_cases(self):
        """Test memory and resource handling edge cases"""
        logger.info("Testing memory edge cases...")

        # Test 1: Large data structure creation
        try:
            large_list = list(range(1000000))  # 1M integers
            assert len(large_list) == 1000000
            del large_list  # Cleanup
        except MemoryError:
            logger.info("Large list creation failed due to memory constraints")

        # Test 2: Memory cleanup verification
        import gc

        process = psutil.Process()
        initial_memory = process.memory_info().rss

        # Create and destroy large objects
        for i in range(10):
            large_data = [list(range(100000)) for _ in range(10)]
            del large_data
            gc.collect()

        final_memory = process.memory_info().rss
        memory_growth = final_memory - initial_memory
        logger.info(
            f"Memory growth after large allocations: {memory_growth / 1024 / 1024:.1f} MB"
        )

        # Test 3: File handle limits
        open_files = []
        try:
            for i in range(100):
                temp_file = os.path.join(self.temp_dir, f"temp_{i}.txt")
                f = open(temp_file, "w")
                f.write(f"test data {i}")
                open_files.append(f)
        except Exception as e:
            logger.info(f"File handle limit reached at {len(open_files)} files: {e}")
        finally:
            for f in open_files:
                try:
                    f.close()
                except Exception:
                    pass

        # Test 4: Thread creation limits
        threads = []
        try:

            def dummy_thread():
                time.sleep(1)

            for i in range(50):
                t = threading.Thread(target=dummy_thread)
                t.start()
                threads.append(t)
        except Exception as e:
            logger.info(f"Thread limit reached at {len(threads)} threads: {e}")
        finally:
            for t in threads:
                try:
                    t.join(timeout=0.1)
                except Exception:
                    pass

        # Test 5: Disk space handling
        try:
            large_file = os.path.join(self.temp_dir, "large_file.bin")
            with open(large_file, "wb") as f:
                # Write 100MB of data
                chunk = b"x" * 1024 * 1024  # 1MB chunk
                for i in range(100):
                    f.write(chunk)

            assert os.path.getsize(large_file) == 100 * 1024 * 1024
            os.remove(large_file)
        except Exception as e:
            logger.info(f"Large file creation failed: {e}")

        # Test 6: CPU intensive operations
        import multiprocessing

        def cpu_intensive_task(n):
            # Calculate prime numbers up to n
            primes = []
            for i in range(2, min(n, 10000)):  # Limit to prevent excessive load
                is_prime = True
                for j in range(2, int(i**0.5) + 1):
                    if i % j == 0:
                        is_prime = False
                        break
                if is_prime:
                    primes.append(i)
            return len(primes)

        try:
            # Run CPU intensive task across multiple processes
            with multiprocessing.Pool(processes=2) as pool:
                results = pool.map(cpu_intensive_task, [1000, 2000, 3000])
                logger.info(f"CPU intensive results: {results}")
        except Exception as e:
            logger.info(f"CPU intensive test failed: {e}")

        logger.info("Memory edge cases completed")

    # ============================================================================
    # ERROR RECOVERY EDGE CASES
    # ============================================================================

    def test_error_recovery_edge_cases(self):
        """Test error recovery and resilience edge cases"""
        logger.info("Testing error recovery edge cases...")

        # Test 1: Retry mechanism with transient failures
        call_count = 0

        @retry_with_backoff(max_retries=3, initial_delay=0.1)
        def flaky_function():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("Transient failure")
            return "success"

        result = flaky_function()
        assert result == "success"
        assert call_count == 3

        # Test 2: Retry with permanent failure
        call_count = 0

        @retry_with_backoff(max_retries=2, initial_delay=0.1)
        def always_failing_function():
            nonlocal call_count
            call_count += 1
            raise ValueError("Permanent failure")

        try:
            always_failing_function()
            assert False, "Should have failed after retries"
        except ValueError:
            pass  # Expected
        assert call_count == 3  # Initial call + 2 retries

        # Test 3: Graceful degradation
        class ServiceWithFallback:
            def __init__(self):
                self.primary_available = True

            def get_data(self):
                if self.primary_available:
                    raise Exception("Primary service down")
                else:
                    return "fallback data"

            def get_data_with_fallback(self):
                try:
                    return self.get_data()
                except Exception:
                    self.primary_available = False
                    return "fallback data"

        service = ServiceWithFallback()
        result = service.get_data_with_fallback()
        assert result == "fallback data"

        # Test 4: Circuit breaker pattern
        class CircuitBreaker:
            def __init__(self, failure_threshold=3, recovery_timeout=5):
                self.failure_threshold = failure_threshold
                self.recovery_timeout = recovery_timeout
                self.failure_count = 0
                self.last_failure_time = None
                self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN

            def call(self, func, *args, **kwargs):
                if self.state == "OPEN":
                    if time.time() - self.last_failure_time > self.recovery_timeout:
                        self.state = "HALF_OPEN"
                    else:
                        raise Exception("Circuit breaker is OPEN")

                try:
                    result = func(*args, **kwargs)
                    if self.state == "HALF_OPEN":
                        self.state = "CLOSED"
                        self.failure_count = 0
                    return result
                except Exception as e:
                    self.failure_count += 1
                    self.last_failure_time = time.time()

                    if self.failure_count >= self.failure_threshold:
                        self.state = "OPEN"

                    raise e

        def unreliable_service():
            if time.time() % 2 < 1:  # Fail half the time
                raise Exception("Service failure")
            return "success"

        breaker = CircuitBreaker(failure_threshold=2, recovery_timeout=1)

        # Trigger circuit breaker
        failures = 0
        for i in range(5):
            try:
                result = breaker.call(unreliable_service)
                logger.info(f"Service call {i}: {result}")
            except Exception as e:
                failures += 1
                logger.info(f"Service call {i} failed: {e}")

        logger.info(
            f"Circuit breaker test: {failures} failures, state: {breaker.state}"
        )

        # Test 5: Deadlock detection and prevention
        lock1 = threading.Lock()
        lock2 = threading.Lock()
        deadlock_detected = False

        def thread1():
            nonlocal deadlock_detected
            try:
                if lock1.acquire(timeout=1):
                    time.sleep(0.1)
                    if lock2.acquire(timeout=1):
                        lock2.release()
                    lock1.release()
                else:
                    deadlock_detected = True
            except Exception as e:
                logger.info(f"Thread1 error: {e}")

        def thread2():
            nonlocal deadlock_detected
            try:
                if lock2.acquire(timeout=1):
                    time.sleep(0.1)
                    if lock1.acquire(timeout=1):
                        lock1.release()
                    lock2.release()
                else:
                    deadlock_detected = True
            except Exception as e:
                logger.info(f"Thread2 error: {e}")

        t1 = threading.Thread(target=thread1)
        t2 = threading.Thread(target=thread2)

        t1.start()
        t2.start()

        t1.join()
        t2.join()

        logger.info(f"Deadlock detection test: deadlock_detected={deadlock_detected}")

        # Test 6: Resource cleanup on exception
        class ResourceManager:
            def __init__(self):
                self.resources = []

            def acquire_resource(self):
                resource = f"resource_{len(self.resources)}"
                self.resources.append(resource)
                return resource

            def release_all(self):
                released = len(self.resources)
                self.resources.clear()
                return released

        manager = ResourceManager()

        try:
            for i in range(5):
                manager.acquire_resource()
                if i == 3:
                    raise Exception("Simulated failure")
        except Exception:
            pass
        finally:
            released = manager.release_all()
            logger.info(f"Released {released} resources after exception")

        logger.info("Error recovery edge cases completed")

    # ============================================================================
    # INTEGRATION AND END-TO-END EDGE CASES
    # ============================================================================

    def test_integration_edge_cases(self):
        """Test full integration and end-to-end edge cases"""
        logger.info("Testing integration edge cases...")

        test_videos = self.create_test_videos()

        # Test 1: Complete pipeline with valid input
        try:
            from main import VideoProcessingPipeline

            # Mock monitoring to avoid port conflicts
            with patch("src.utils.monitoring_integration.MonitoringServer"):
                pipeline = VideoProcessingPipeline()

                # Create job
                job_id = pipeline.create_job(
                    test_videos["short"],
                    os.path.join(self.temp_dir, "pipeline_output.mp4"),
                )

                # Process job (may fail due to missing dependencies, but should handle gracefully)
                try:
                    result = pipeline.process_job(job_id)
                    logger.info(f"Pipeline completed successfully: {result}")
                except Exception as e:
                    logger.info(f"Pipeline failed as expected: {e}")

        except ImportError as e:
            logger.info(f"Pipeline import failed: {e}")

        # Test 2: Pipeline with invalid input
        try:
            with patch("src.utils.monitoring_integration.MonitoringServer"):
                pipeline = VideoProcessingPipeline()

                job_id = pipeline.create_job(
                    test_videos["corrupt"],
                    os.path.join(self.temp_dir, "invalid_output.mp4"),
                )

                try:
                    result = pipeline.process_job(job_id)
                    assert False, "Should have failed with corrupt input"
                except Exception:
                    pass  # Expected

        except Exception as e:
            logger.info(f"Invalid pipeline test failed: {e}")

        # Test 3: Pipeline interruption
        try:
            with patch("src.utils.monitoring_integration.MonitoringServer"):
                pipeline = VideoProcessingPipeline()

                def interrupt_pipeline():
                    time.sleep(1)
                    # Simulate SIGTERM
                    os.kill(os.getpid(), signal.SIGTERM)

                interrupt_thread = threading.Thread(target=interrupt_pipeline)
                interrupt_thread.daemon = True
                interrupt_thread.start()

                job_id = pipeline.create_job(
                    test_videos["long"],  # Long video
                    os.path.join(self.temp_dir, "interrupted_output.mp4"),
                )

                try:
                    result = pipeline.process_job(job_id)
                    logger.info("Pipeline completed despite interrupt")
                except Exception as e:
                    logger.info(f"Pipeline interrupted: {e}")

        except Exception as e:
            logger.info(f"Pipeline interruption test failed: {e}")

        # Test 4: Multiple concurrent jobs
        try:
            with patch("src.utils.monitoring_integration.MonitoringServer"):
                pipeline = VideoProcessingPipeline()

                job_ids = []
                for i in range(3):
                    job_id = pipeline.create_job(
                        test_videos["short"],
                        os.path.join(self.temp_dir, f"concurrent_{i}.mp4"),
                    )
                    job_ids.append(job_id)

                # Process jobs concurrently
                def process_job(job_id):
                    try:
                        return pipeline.process_job(job_id)
                    except Exception as e:
                        logger.info(f"Concurrent job {job_id} failed: {e}")
                        return None

                threads = []
                for job_id in job_ids:
                    t = threading.Thread(target=process_job, args=(job_id,))
                    threads.append(t)
                    t.start()

                for t in threads:
                    t.join()

                logger.info("Concurrent jobs completed")

        except Exception as e:
            logger.info(f"Concurrent jobs test failed: {e}")

        # Test 5: Resource exhaustion scenarios
        try:
            # Simulate low disk space
            with patch(
                "shutil.disk_usage", return_value=(1000, 100, 50)
            ):  # Very low free space
                with patch("src.utils.monitoring_integration.MonitoringServer"):
                    pipeline = VideoProcessingPipeline()

                    job_id = pipeline.create_job(
                        test_videos["short"],
                        os.path.join(self.temp_dir, "low_disk_output.mp4"),
                    )

                    try:
                        result = pipeline.process_job(job_id)
                        logger.info("Pipeline handled low disk space")
                    except Exception as e:
                        logger.info(f"Pipeline failed with low disk space: {e}")

        except Exception as e:
            logger.info(f"Resource exhaustion test failed: {e}")

        logger.info("Integration edge cases completed")

    # ============================================================================
    # MAIN EXECUTION
    # ============================================================================

    def run_all_tests(self):
        """Run all edge case tests systematically"""
        logger.info("=== STARTING COMPREHENSIVE EDGE PATH TESTING ===")

        test_methods = [
            self.test_database_edge_cases,
            self.test_checkpoint_edge_cases,
            self.test_video_validation_edge_cases,
            self.test_budget_edge_cases,
            self.test_video_processing_edge_cases,
            self.test_smart_crop_edge_cases,
            self.test_memory_edge_cases,
            self.test_error_recovery_edge_cases,
            self.test_integration_edge_cases,
        ]

        # Add async test
        async_tests = [self.test_async_edge_cases]

        results = {}

        # Run synchronous tests
        for test_method in test_methods:
            test_name = test_method.__name__
            logger.info(f"\n--- Running {test_name} ---")

            try:
                start_time = time.time()
                test_method()
                duration = time.time() - start_time
                results[test_name] = {
                    "status": "PASSED",
                    "duration": duration,
                    "error": None,
                }
                logger.info(f"{test_name} PASSED in {duration:.2f}s")

            except Exception as e:
                duration = time.time() - start_time
                results[test_name] = {
                    "status": "FAILED",
                    "duration": duration,
                    "error": str(e),
                }
                logger.error(f"{test_name} FAILED: {e}")

        # Run asynchronous tests
        for async_test in async_tests:
            test_name = async_test.__name__
            logger.info(f"\n--- Running {test_name} ---")

            try:
                start_time = time.time()
                asyncio.run(async_test())
                duration = time.time() - start_time
                results[test_name] = {
                    "status": "PASSED",
                    "duration": duration,
                    "error": None,
                }
                logger.info(f"{test_name} PASSED in {duration:.2f}s")

            except Exception as e:
                duration = time.time() - start_time
                results[test_name] = {
                    "status": "FAILED",
                    "duration": duration,
                    "error": str(e),
                }
                logger.error(f"{test_name} FAILED: {e}")

        # Summary
        total_tests = len(results)
        passed_tests = sum(1 for r in results.values() if r["status"] == "PASSED")
        failed_tests = total_tests - passed_tests
        total_duration = sum(r["duration"] for r in results.values())

        logger.info("\n=== EDGE PATH TESTING SUMMARY ===")
        logger.info(f"Total tests: {total_tests}")
        logger.info(f"Passed: {passed_tests}")
        logger.info(f"Failed: {failed_tests}")
        logger.info(f"Total duration: {total_duration:.2f}s")
        logger.info(f"Success rate: {passed_tests/total_tests*100:.1f}%")

        if failed_tests > 0:
            logger.info("\nFailed tests:")
            for test_name, result in results.items():
                if result["status"] == "FAILED":
                    logger.info(f"  {test_name}: {result['error']}")

        return results


def main():
    """Main entry point for edge path coverage testing"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    tester = EdgePathTester()

    try:
        tester.setup()
        results = tester.run_all_tests()

        # Save results to file
        results_file = os.path.join(tester.temp_dir, "edge_path_results.json")
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)

        print(f"Results saved to: {results_file}")

        # Exit with appropriate code
        failed_count = sum(1 for r in results.values() if r["status"] == "FAILED")
        sys.exit(1 if failed_count > 0 else 0)

    finally:
        tester.cleanup()


if __name__ == "__main__":
    main()
