"""Test concat demuxer-based video editor"""

import pytest
import os
import tempfile
from unittest.mock import patch, MagicMock
from concat_editor import (
    EditSegment,
    ConcatEditor,
    create_edit_segments
)


class TestEditSegment:
    """Test EditSegment dataclass"""
    
    def test_segment_creation(self):
        """Test creating edit segment"""
        segment = EditSegment("video.mp4", 10.0, 25.0)
        
        assert segment.source_file == "video.mp4"
        assert segment.start_time == 10.0
        assert segment.end_time == 25.0
        assert segment.duration == 15.0
        assert segment.transition_type == "fade"
        assert segment.transition_duration == 0.5
        assert segment.segment_id is not None
    
    def test_custom_transition(self):
        """Test segment with custom transition"""
        segment = EditSegment(
            "video.mp4", 
            0, 30,
            transition_type="dissolve",
            transition_duration=1.0
        )
        
        assert segment.transition_type == "dissolve"
        assert segment.transition_duration == 1.0


class TestConcatEditor:
    """Test concat editor functionality"""
    
    @pytest.fixture
    def editor(self):
        """Create concat editor instance"""
        with tempfile.TemporaryDirectory() as temp_dir:
            editor = ConcatEditor()
            editor.temp_dir = temp_dir
            yield editor
    
    def test_create_concat_list(self, editor):
        """Test concat list file creation"""
        files = [
            "/tmp/segment1.ts",
            "/tmp/segment2.ts", 
            "/tmp/segment3.ts"
        ]
        
        concat_list = editor._create_concat_list(files)
        
        assert os.path.exists(concat_list)
        
        # Verify content
        with open(concat_list, 'r') as f:
            content = f.read()
        
        for file_path in files:
            assert f"file '{file_path}'" in content
        
        # Cleanup
        os.unlink(concat_list)
    
    @patch('subprocess.run')
    def test_simple_concat(self, mock_run, editor):
        """Test simple concatenation without transitions"""
        mock_run.return_value = MagicMock(returncode=0)
        
        editor._simple_concat(
            "concat.txt",
            "output.mp4",
            "libx264",
            "aac"
        )
        
        # Verify FFmpeg command
        cmd = mock_run.call_args[0][0]
        assert '-f' in cmd
        assert 'concat' in cmd
        assert '-safe' in cmd
        assert '0' in cmd
        assert 'output.mp4' in cmd
    
    def test_build_xfade_filter(self, editor):
        """Test xfade filter generation"""
        segments = [
            EditSegment("input.mp4", 0, 10, transition_duration=0.5),
            EditSegment("input.mp4", 10, 20, transition_duration=0.5),
            EditSegment("input.mp4", 20, 30, transition_duration=0.5)
        ]
        
        filter_str = editor._build_xfade_filter(segments)
        
        # Check filter components
        assert '[0:v]xfadeall' in filter_str
        assert 'transitions=2' in filter_str  # 3 segments = 2 transitions
        assert '[v]' in filter_str
        assert '[a]' in filter_str
        
        # Verify length constraint
        assert len(filter_str) < 300
    
    def test_filter_length_constraint(self, editor):
        """Test filter length verification for many segments"""
        # Create 50 segments
        segments = []
        for i in range(50):
            segments.append(
                EditSegment("input.mp4", i * 10, (i + 1) * 10, transition_duration=0.5)
            )
        
        # Verify filter length
        is_valid, length = editor.verify_filter_length(segments)
        
        # Should meet constraint (< 300 chars)
        assert is_valid, f"Filter too long: {length} chars"
        assert length < 300
    
    def test_filter_length_single_segment(self, editor):
        """Test filter length for single segment"""
        segments = [EditSegment("input.mp4", 0, 30)]
        
        is_valid, length = editor.verify_filter_length(segments)
        
        assert is_valid
        assert length == 0  # No filter needed for single segment
    
    @patch('concat_editor.FFmpegPipeline')
    def test_extract_segments_to_files(self, mock_pipeline_class, editor):
        """Test parallel segment extraction"""
        # Mock pipeline
        mock_pipeline = MagicMock()
        mock_pipeline.wait_all.return_value = {
            'extract_000': 0,
            'extract_001': 0,
            'extract_002': 0
        }
        mock_pipeline.__enter__.return_value = mock_pipeline
        mock_pipeline_class.return_value = mock_pipeline
        
        segments = [
            EditSegment("input.mp4", 0, 10),
            EditSegment("input.mp4", 20, 30),
            EditSegment("input.mp4", 40, 50)
        ]
        
        temp_files = editor._extract_segments_to_files(segments)
        
        # Should create one file per segment
        assert len(temp_files) == 3
        
        # Should add extraction processes
        assert mock_pipeline.add_process.call_count == 3
        
        # Verify extraction commands
        for i, call_args in enumerate(mock_pipeline.add_process.call_args_list):
            cmd = call_args[0][0]
            assert '-ss' in cmd
            assert str(segments[i].start_time) in cmd
            assert '-t' in cmd
            assert str(segments[i].duration) in cmd
            assert '-f' in cmd
            assert 'mpegts' in cmd
    
    @patch('concat_editor.FFmpegPipeline')
    def test_extract_segments_failure(self, mock_pipeline_class, editor):
        """Test handling of extraction failures"""
        # Mock pipeline with failure
        mock_pipeline = MagicMock()
        mock_pipeline.wait_all.return_value = {
            'extract_000': 0,
            'extract_001': 1,  # Failed
            'extract_002': 0
        }
        mock_pipeline.__enter__.return_value = mock_pipeline
        mock_pipeline_class.return_value = mock_pipeline
        
        segments = [
            EditSegment("input.mp4", 0, 10),
            EditSegment("input.mp4", 20, 30),
            EditSegment("input.mp4", 40, 50)
        ]
        
        with pytest.raises(Exception) as exc_info:
            editor._extract_segments_to_files(segments)
        
        assert "extraction failed" in str(exc_info.value).lower()
    
    @patch('subprocess.run')
    def test_concat_with_transitions_small(self, mock_run, editor):
        """Test concatenation with transitions for small segment count"""
        mock_run.return_value = MagicMock(returncode=0)
        
        segments = [
            EditSegment("input.mp4", 0, 10),
            EditSegment("input.mp4", 10, 20),
            EditSegment("input.mp4", 20, 30)
        ]
        
        editor._concat_with_transitions(
            "concat.txt",
            "output.mp4", 
            segments,
            "libx264",
            "aac"
        )
        
        # Should use filter_complex
        cmd = mock_run.call_args[0][0]
        assert '-filter_complex' in cmd
        assert 'xfadeall' in cmd[cmd.index('-filter_complex') + 1]
    
    @patch('concat_editor.ConcatEditor._concat_with_batch_transitions')
    def test_concat_with_transitions_large(self, mock_batch, editor):
        """Test fallback to batch transitions for many segments"""
        # Create many segments
        segments = [EditSegment("input.mp4", i * 10, (i + 1) * 10) for i in range(20)]
        
        editor._concat_with_transitions(
            "concat.txt",
            "output.mp4",
            segments,
            "libx264",
            "aac"
        )
        
        # Should fall back to batch approach
        mock_batch.assert_called_once()
    
    def test_cleanup_temp_files(self, editor):
        """Test temporary file cleanup"""
        # Create temp files
        temp_files = []
        for i in range(3):
            fd, path = tempfile.mkstemp(dir=editor.temp_dir)
            os.close(fd)
            temp_files.append(path)
            assert os.path.exists(path)
        
        # Cleanup
        editor._cleanup_temp_files(temp_files)
        
        # All files should be removed
        for path in temp_files:
            assert not os.path.exists(path)
    
    @patch('concat_editor.ConcatEditor._extract_segments_to_files')
    @patch('concat_editor.ConcatEditor._simple_concat')
    def test_execute_edit_no_transitions(self, mock_concat, mock_extract, editor):
        """Test complete edit execution without transitions"""
        # Mock methods
        temp_files = ["/tmp/seg1.ts", "/tmp/seg2.ts"]
        mock_extract.return_value = temp_files
        
        segments = [
            EditSegment("input.mp4", 0, 30),
            EditSegment("input.mp4", 60, 90)
        ]
        
        with patch('os.unlink'):  # Mock cleanup
            result = editor.execute_edit(
                segments,
                "output.mp4",
                apply_transitions=False
            )
        
        # Verify result
        assert result['segments_processed'] == 2
        assert result['total_duration'] == 60
        assert result['output_file'] == "output.mp4"
        assert 'processing_time' in result
        assert 'processing_ratio' in result
        
        # Verify methods called
        mock_extract.assert_called_once_with(segments)
        mock_concat.assert_called_once()


class TestHelperFunctions:
    """Test helper functions"""
    
    def test_create_edit_segments(self):
        """Test converting highlights to edit segments"""
        highlights = [
            {
                'start_time': 10,
                'end_time': 40,
                'transition': 'dissolve',
                'transition_duration': 1.0
            },
            {
                'start_time': 60,
                'end_time': 90
                # Use defaults
            }
        ]
        
        segments = create_edit_segments(highlights, "source.mp4")
        
        assert len(segments) == 2
        
        # First segment
        assert segments[0].source_file == "source.mp4"
        assert segments[0].start_time == 10
        assert segments[0].end_time == 40
        assert segments[0].transition_type == "dissolve"
        assert segments[0].transition_duration == 1.0
        
        # Second segment (defaults)
        assert segments[1].transition_type == "fade"
        assert segments[1].transition_duration == 0.5


class TestIntegration:
    """Integration tests"""
    
    @patch('subprocess.run')
    @patch('concat_editor.FFmpegPipeline')
    def test_full_edit_workflow(self, mock_pipeline_class, mock_run):
        """Test complete editing workflow"""
        # Mock successful execution
        mock_run.return_value = MagicMock(returncode=0)
        
        # Mock pipeline
        mock_pipeline = MagicMock()
        mock_pipeline.wait_all.return_value = {
            'extract_000': 0,
            'extract_001': 0
        }
        mock_pipeline.__enter__.return_value = mock_pipeline
        mock_pipeline_class.return_value = mock_pipeline
        
        editor = ConcatEditor()
        
        segments = [
            EditSegment("input.mp4", 0, 30),
            EditSegment("input.mp4", 60, 90)
        ]
        
        with patch('os.unlink'):  # Mock cleanup
            result = editor.execute_edit(
                segments,
                "final.mp4",
                apply_transitions=True
            )
        
        # Verify workflow
        assert result['segments_processed'] == 2
        assert mock_pipeline.add_process.call_count == 2  # Two extractions
        assert mock_run.called  # Concatenation executed


if __name__ == "__main__":
    pytest.main([__file__, "-v"])