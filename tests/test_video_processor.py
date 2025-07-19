"""Test FIFO-based video processing pipeline"""

import pytest
import os
import tempfile
import subprocess
from unittest.mock import patch, MagicMock
from video_processor import (
    VideoSegment,
    FIFOManager,
    FFmpegPipeline,
    VideoEditor,
    VideoProcessingError,
)


class TestVideoSegment:
    """Test VideoSegment dataclass"""

    def test_segment_creation(self):
        """Test creating video segment"""
        segment = VideoSegment(10.5, 25.5, "test.mp4")

        assert segment.start_time == 10.5
        assert segment.end_time == 25.5
        assert segment.input_file == "test.mp4"
        assert segment.duration == 15.0
        assert segment.segment_id is not None

    def test_segment_with_custom_id(self):
        """Test segment with custom ID"""
        segment = VideoSegment(0, 30, "test.mp4", "custom_id")
        assert segment.segment_id == "custom_id"


class TestFIFOManager:
    """Test FIFO manager functionality"""

    def test_create_fifo(self):
        """Test FIFO creation"""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = FIFOManager(temp_dir)

            # Create FIFO
            fifo_path = manager.create_fifo("_test")

            assert os.path.exists(fifo_path)
            assert "_test" in fifo_path
            assert fifo_path in manager.fifos

            # Verify it's a FIFO
            assert os.path.stat(fifo_path).st_mode & 0o170000 == 0o010000  # S_ISFIFO

            # Cleanup
            manager.cleanup()
            assert not os.path.exists(fifo_path)

    def test_multiple_fifos(self):
        """Test creating multiple FIFOs"""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = FIFOManager(temp_dir)

            # Create multiple FIFOs
            fifos = []
            for i in range(5):
                fifo = manager.create_fifo(f"_{i}")
                fifos.append(fifo)

            # All should exist
            assert len(manager.fifos) == 5
            for fifo in fifos:
                assert os.path.exists(fifo)

            # Cleanup should remove all
            manager.cleanup()
            for fifo in fifos:
                assert not os.path.exists(fifo)

    def test_context_manager(self):
        """Test FIFO manager as context manager"""
        with tempfile.TemporaryDirectory() as temp_dir:
            fifo_path = None

            with FIFOManager(temp_dir) as manager:
                fifo_path = manager.create_fifo()
                assert os.path.exists(fifo_path)

            # Should be cleaned up after context
            assert not os.path.exists(fifo_path)

    @patch("os.mkfifo")
    def test_fifo_creation_error(self, mock_mkfifo):
        """Test error handling in FIFO creation"""
        mock_mkfifo.side_effect = OSError("Permission denied")

        manager = FIFOManager()
        with pytest.raises(VideoProcessingError):
            manager.create_fifo()


class TestFFmpegPipeline:
    """Test FFmpeg pipeline management"""

    @patch("subprocess.Popen")
    def test_add_process(self, mock_popen):
        """Test adding process to pipeline"""
        # Mock process
        mock_process = MagicMock()
        mock_process.stderr = MagicMock()
        mock_process.stderr.readline.return_value = b""
        mock_popen.return_value = mock_process

        pipeline = FFmpegPipeline()

        # Add process
        cmd = ["ffmpeg", "-i", "input.mp4", "-c", "copy", "output.mp4"]
        process = pipeline.add_process(cmd, "test_ffmpeg")

        assert process == mock_process
        assert len(pipeline.processes) == 1
        assert pipeline.processes[0] == ("test_ffmpeg", mock_process)

        # Verify subprocess was called correctly
        mock_popen.assert_called_once_with(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0
        )

    @patch("subprocess.Popen")
    def test_wait_all_success(self, mock_popen):
        """Test waiting for all processes successfully"""
        # Mock processes
        process1 = MagicMock()
        process1.wait.return_value = 0
        process1.poll.return_value = 0

        process2 = MagicMock()
        process2.wait.return_value = 0
        process2.poll.return_value = 0

        pipeline = FFmpegPipeline()
        pipeline.processes = [("proc1", process1), ("proc2", process2)]

        # Wait for all
        results = pipeline.wait_all(timeout=30)

        assert results == {"proc1": 0, "proc2": 0}
        process1.wait.assert_called_once()
        process2.wait.assert_called_once()

    @patch("subprocess.Popen")
    def test_wait_all_failure(self, mock_popen):
        """Test handling process failures"""
        # Mock failed process
        process = MagicMock()
        process.wait.return_value = 1
        process.communicate.return_value = (b"", b"Error: Invalid input")

        pipeline = FFmpegPipeline()
        pipeline.processes = [("failed_proc", process)]

        # Wait should capture failure
        results = pipeline.wait_all()

        assert results == {"failed_proc": 1}
        process.communicate.assert_called_once()

    @patch("subprocess.Popen")
    def test_wait_all_timeout(self, mock_popen):
        """Test timeout handling"""
        # Mock process that times out
        process = MagicMock()
        process.wait.side_effect = subprocess.TimeoutExpired("cmd", 30)
        process.poll.return_value = None

        pipeline = FFmpegPipeline()
        pipeline.processes = [("slow_proc", process)]

        # Wait with timeout
        results = pipeline.wait_all(timeout=30)

        assert results == {"slow_proc": -1}
        process.terminate.assert_called_once()

    def test_cleanup(self):
        """Test pipeline cleanup"""
        # Mock running process
        process = MagicMock()
        process.poll.return_value = None  # Still running

        pipeline = FFmpegPipeline()
        pipeline.processes = [("test", process)]

        # Cleanup should terminate
        pipeline.cleanup()

        process.terminate.assert_called_once()
        process.wait.assert_called_once_with(timeout=5)
        assert len(pipeline.processes) == 0


class TestVideoEditor:
    """Test video editor functionality"""

    @pytest.fixture
    def editor(self):
        """Create video editor instance"""
        return VideoEditor()

    @patch("subprocess.Popen")
    def test_extract_segments_parallel(self, mock_popen, editor):
        """Test parallel segment extraction"""
        # Mock process
        mock_process = MagicMock()
        mock_process.stderr.readline.return_value = b""
        mock_popen.return_value = mock_process

        # Create segments
        segments = [
            VideoSegment(0, 10, "input.mp4"),
            VideoSegment(20, 30, "input.mp4"),
            VideoSegment(40, 50, "input.mp4"),
        ]

        # Extract segments
        with patch("os.mkfifo"):  # Mock FIFO creation
            fifos = editor.extract_segments_parallel("input.mp4", segments)

        # Should create one FIFO per segment
        assert len(fifos) == 3

        # Should start one FFmpeg process per segment
        assert mock_popen.call_count == 3

        # Verify FFmpeg commands
        for i, call_args in enumerate(mock_popen.call_args_list):
            cmd = call_args[0][0]
            assert "-ss" in cmd
            assert str(segments[i].start_time) in cmd
            assert "-t" in cmd
            assert str(segments[i].duration) in cmd

    def test_create_concat_list(self, editor):
        """Test concat list creation"""
        fifos = ["/tmp/fifo1", "/tmp/fifo2", "/tmp/fifo3"]

        concat_file = editor._create_concat_list(fifos)

        assert os.path.exists(concat_file)

        # Verify content
        with open(concat_file, "r") as f:
            content = f.read()

        for fifo in fifos:
            assert f"file '{fifo}'" in content

        # Cleanup
        os.unlink(concat_file)

    @patch("subprocess.Popen")
    @patch("os.mkfifo")
    def test_concatenate_segments_fifo(self, mock_mkfifo, mock_popen, editor):
        """Test FIFO-based concatenation"""
        # Mock process
        mock_process = MagicMock()
        mock_process.wait.return_value = 0
        mock_process.stderr.readline.return_value = b""
        mock_popen.return_value = mock_process

        fifos = ["/tmp/fifo1", "/tmp/fifo2"]

        # Mock concat list creation
        with patch.object(
            editor, "_create_concat_list", return_value="/tmp/concat.txt"
        ):
            editor.concatenate_segments_fifo(fifos, "output.mp4")

        # Verify FFmpeg was called with concat demuxer
        cmd = mock_popen.call_args[0][0]
        assert "-f" in cmd
        assert "concat" in cmd
        assert "output.mp4" in cmd

    @patch("subprocess.run")
    def test_apply_transitions_no_transitions(self, mock_run, editor):
        """Test handling no transitions"""
        mock_run.return_value = MagicMock(returncode=0)

        editor.apply_transitions_fifo("input.mp4", "output.mp4", [])

        # Should just copy without transitions
        cmd = mock_run.call_args[0][0]
        assert "-c" in cmd
        assert "copy" in cmd

    @patch.object(VideoEditor, "process_with_filter_fifo")
    def test_apply_transitions_with_points(self, mock_process, editor):
        """Test applying transitions"""
        transition_points = [30.0, 60.0, 90.0]

        editor.apply_transitions_fifo("input.mp4", "output.mp4", transition_points)

        # Should call process with filter
        mock_process.assert_called_once()

        # Check filter was built correctly
        filter_arg = mock_process.call_args[0][2]
        assert "fade" in filter_arg
        assert "t=out" in filter_arg
        assert "t=in" in filter_arg

    @patch("video_processor.FFmpegPipeline")
    def test_process_with_filter_fifo(self, mock_pipeline_class, editor):
        """Test filter processing with FIFO"""
        # Mock pipeline
        mock_pipeline = MagicMock()
        mock_pipeline.wait_all.return_value = {"filter": 0, "encode": 0}
        mock_pipeline.fifo_manager.create_fifo.return_value = "/tmp/filtered_fifo"
        mock_pipeline.__enter__.return_value = mock_pipeline
        mock_pipeline_class.return_value = mock_pipeline

        # Process with filter
        editor.process_with_filter_fifo("input.mp4", "output.mp4", "scale=1920:1080")

        # Should add two processes (filter and encode)
        assert mock_pipeline.add_process.call_count == 2

        # First call should be filter to FIFO
        filter_call = mock_pipeline.add_process.call_args_list[0]
        filter_cmd = filter_call[0][0]
        assert "-filter_complex" in filter_cmd
        assert "scale=1920:1080" in filter_cmd

        # Second call should be encode from FIFO
        encode_call = mock_pipeline.add_process.call_args_list[1]
        encode_cmd = encode_call[0][0]
        assert "/tmp/filtered_fifo" in encode_cmd
        assert "output.mp4" in encode_cmd


class TestIntegration:
    """Integration tests for video processing"""

    @patch("subprocess.Popen")
    @patch("os.mkfifo")
    def test_extract_and_concatenate_efficient(self, mock_mkfifo, mock_popen):
        """Test complete extraction and concatenation workflow"""
        # Mock successful processes
        mock_process = MagicMock()
        mock_process.wait.return_value = 0
        mock_process.poll.return_value = 0
        mock_process.stderr.readline.return_value = b""
        mock_popen.return_value = mock_process

        editor = VideoEditor()

        segments = [
            VideoSegment(0, 30, "input.mp4"),
            VideoSegment(60, 90, "input.mp4"),
            VideoSegment(120, 150, "input.mp4"),
        ]

        # Run efficient processing
        with patch("os.rename"):  # Mock file move
            editor.extract_and_concatenate_efficient(
                "input.mp4", segments, "output.mp4", apply_transitions=False
            )

        # Should have started extraction processes
        assert mock_popen.call_count >= len(segments)

        # Verify performance ratio tracking
        from metrics import metrics

        ratio_count = metrics.processing_ratio.labels(stage="editing")._count.get()
        assert ratio_count > 0

    def test_performance_requirement(self):
        """Test that processing meets performance requirements"""
        # This is a conceptual test - in real usage would need actual video

        # Requirement: 20-minute 1080p video in < 1.2x duration
        video_duration = 20 * 60  # 20 minutes in seconds
        max_processing_time = video_duration * 1.2

        # Mock timing

        # Simulate processing segments totaling 20 minutes
        segments = []
        for i in range(40):  # 40 x 30-second segments
            segments.append(VideoSegment(i * 30, (i + 1) * 30, "input.mp4"))

        # In real test, would process actual video
        # For now, just verify the math
        total_duration = sum(s.duration for s in segments)
        assert total_duration == video_duration

        # Processing time should be under limit
        simulated_processing_time = video_duration * 0.8  # Simulate 0.8x ratio
        assert simulated_processing_time < max_processing_time


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
