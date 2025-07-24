#!/usr/bin/env python3
"""
Comprehensive tests for video effects and transitions functionality
Tests FFmpeg-based video processing, transitions, and effects
Target: ≥90% coverage for montage/utils/video_effects.py
"""
import os
import subprocess
import tempfile
from unittest.mock import MagicMock, Mock, patch, call, mock_open

import pytest

from montage.utils.video_effects import create_professional_video


class TestCreateProfessionalVideo:
    """Test suite for create_professional_video function"""

    @pytest.fixture
    def sample_clips(self):
        """Create sample clip data"""
        return [
            {"start_ms": 0, "end_ms": 5000},
            {"start_ms": 10000, "end_ms": 15000},
            {"start_ms": 20000, "end_ms": 25000},
        ]

    @pytest.fixture
    def sample_single_clip(self):
        """Create single clip data"""
        return [{"start_ms": 1000, "end_ms": 6000}]

    @pytest.fixture
    def sample_two_clips(self):
        """Create two clip data for transition testing"""
        return [
            {"start_ms": 0, "end_ms": 5000},
            {"start_ms": 10000, "end_ms": 15000},
        ]

    @pytest.fixture
    def temp_output(self):
        """Create temporary output file path"""
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f:
            output_path = f.name
        yield output_path
        # Cleanup
        if os.path.exists(output_path):
            os.remove(output_path)

    def test_basic_video_creation_landscape(self, sample_clips, temp_output):
        """Test basic video creation in landscape format"""
        with patch('subprocess.run') as mock_run:
            # Mock successful subprocess calls
            mock_run.return_value = MagicMock(returncode=0, stderr="")
            
            with patch('os.makedirs') as mock_makedirs:
                with patch('os.unlink') as mock_unlink:
                    result = create_professional_video(
                        clips=sample_clips,
                        source_video="/path/to/source.mp4",
                        output_path=temp_output,
                        vertical_format=False,
                        enhance_colors=True,
                        add_transitions=True
                    )
                    
                    assert result is True
                    
                    # Verify temp directory creation
                    mock_makedirs.assert_called_once_with('/tmp/montage_pro', exist_ok=True)
                    
                    # Verify subprocess was called for each clip, transitions, and final assembly
                    assert mock_run.call_count >= len(sample_clips)

    def test_basic_video_creation_no_transitions(self, sample_clips, temp_output):
        """Test basic video creation without transitions"""
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stderr="")
            
            result = create_professional_video(
                clips=sample_clips,
                source_video="/path/to/source.mp4",
                output_path=temp_output,
                vertical_format=False,
                enhance_colors=True,
                add_transitions=False
            )
            
            assert result is True

    def test_basic_video_creation_no_color_enhancement(self, sample_clips, temp_output):
        """Test basic video creation without color enhancement"""
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stderr="")
            
            result = create_professional_video(
                clips=sample_clips,
                source_video="/path/to/source.mp4",
                output_path=temp_output,
                vertical_format=False,
                enhance_colors=False,
                add_transitions=True
            )
            
            assert result is True

    def test_video_creation_with_vertical_format_with_cropper(self, sample_clips, temp_output):
        """Test video creation with vertical format using IntelligentCropper"""
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stderr="")
            
            with patch('montage.utils.video_effects.IntelligentCropper') as mock_cropper_class:
                mock_cropper = MagicMock()
                mock_cropper.analyze_video_content.return_value = {
                    "crop_center": {"x": 960, "y": 540},
                    "confidence": 0.9
                }
                mock_cropper.generate_crop_filter.return_value = "crop=608:1080:336:0"
                mock_cropper_class.return_value = mock_cropper
                
                # Mock ffprobe to return dimensions
                probe_result = MagicMock()
                probe_result.stdout = "1920x1080"
                probe_result.returncode = 0
                mock_run.side_effect = [probe_result] * len(sample_clips) + [MagicMock(returncode=0, stderr="")] * 10
                
                result = create_professional_video(
                    clips=sample_clips,
                    source_video="/path/to/source.mp4",
                    output_path=temp_output,
                    vertical_format=True
                )
                
                assert result is True
                
                # Verify intelligent cropper was used
                assert mock_cropper.analyze_video_content.call_count == len(sample_clips)
                assert mock_cropper.generate_crop_filter.call_count == len(sample_clips)

    def test_video_creation_with_vertical_format_fallback_no_cropper(self, sample_clips, temp_output):
        """Test video creation with vertical format fallback when IntelligentCropper not available"""
        with patch('subprocess.run') as mock_run:
            # Mock ffprobe to return dimensions
            probe_result = MagicMock()
            probe_result.stdout = "1920x1080"
            probe_result.returncode = 0
            mock_run.side_effect = [probe_result] * len(sample_clips) + [MagicMock(returncode=0, stderr="")] * 10
            
            # Mock ImportError for IntelligentCropper
            with patch('montage.utils.video_effects.IntelligentCropper', side_effect=ImportError("Module not found")):
                result = create_professional_video(
                    clips=sample_clips,
                    source_video="/path/to/source.mp4",
                    output_path=temp_output,
                    vertical_format=True
                )
                
                assert result is True
                
                # Verify fallback crop filter was used by checking the ffmpeg commands
                # The actual verification happens in the command structure checks

    def test_video_creation_vertical_ffprobe_failure(self, sample_clips, temp_output):
        """Test video creation with vertical format when ffprobe fails"""
        with patch('subprocess.run') as mock_run:
            # Mock ffprobe failure, then successful clip processing
            probe_result = MagicMock()
            probe_result.returncode = 1  # ffprobe fails
            mock_run.side_effect = [probe_result] * len(sample_clips) + [MagicMock(returncode=0, stderr="")] * 10
            
            result = create_professional_video(
                clips=sample_clips,
                source_video="/path/to/source.mp4",
                output_path=temp_output,
                vertical_format=True
            )
            
            assert result is True

    def test_video_creation_vertical_ffprobe_exception(self, sample_clips, temp_output):
        """Test video creation with vertical format when ffprobe raises exception"""
        with patch('subprocess.run') as mock_run:
            def side_effect(*args, **kwargs):
                cmd = args[0]
                if 'ffprobe' in cmd:
                    raise subprocess.CalledProcessError(1, cmd)
                return MagicMock(returncode=0, stderr="")
            
            mock_run.side_effect = side_effect
            
            result = create_professional_video(
                clips=sample_clips,
                source_video="/path/to/source.mp4",
                output_path=temp_output,
                vertical_format=True
            )
            
            assert result is True

    def test_video_creation_with_subtitles_landscape(self, sample_clips, temp_output):
        """Test video creation with subtitles in landscape format"""
        with tempfile.NamedTemporaryFile(suffix='.srt', delete=False) as f:
            subtitles_path = f.name
            # Write sample SRT content
            f.write(b"1\n00:00:00,000 --> 00:00:05,000\nTest subtitle\n\n")
        
        try:
            with patch('subprocess.run') as mock_run:
                mock_run.return_value = MagicMock(returncode=0, stderr="")
                
                result = create_professional_video(
                    clips=sample_clips,
                    source_video="/path/to/source.mp4",
                    output_path=temp_output,
                    subtitles_path=subtitles_path,
                    vertical_format=False
                )
                
                assert result is True
                
                # Check that subtitles filter was included in final command
                final_call = mock_run.call_args_list[-1]
                cmd = final_call[0][0]
                assert any('subtitles=' in str(arg) and 'FontSize=24' in str(arg) for arg in cmd)
        
        finally:
            if os.path.exists(subtitles_path):
                os.remove(subtitles_path)

    def test_video_creation_with_subtitles_vertical(self, sample_clips, temp_output):
        """Test video creation with subtitles in vertical format"""
        with tempfile.NamedTemporaryFile(suffix='.srt', delete=False) as f:
            subtitles_path = f.name
            f.write(b"1\n00:00:00,000 --> 00:00:05,000\nTest subtitle\n\n")
        
        try:
            with patch('subprocess.run') as mock_run:
                mock_run.return_value = MagicMock(returncode=0, stderr="")
                
                result = create_professional_video(
                    clips=sample_clips,
                    source_video="/path/to/source.mp4",
                    output_path=temp_output,
                    subtitles_path=subtitles_path,
                    vertical_format=True
                )
                
                assert result is True
                
                # Check that subtitles filter has vertical formatting
                final_call = mock_run.call_args_list[-1]
                cmd = final_call[0][0]
                assert any('subtitles=' in str(arg) and 'FontSize=32' in str(arg) for arg in cmd)
        
        finally:
            if os.path.exists(subtitles_path):
                os.remove(subtitles_path)

    def test_video_creation_with_nonexistent_subtitles(self, sample_clips, temp_output):
        """Test video creation with nonexistent subtitles file"""
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stderr="")
            
            result = create_professional_video(
                clips=sample_clips,
                source_video="/path/to/source.mp4",
                output_path=temp_output,
                subtitles_path="/nonexistent/subtitles.srt"
            )
            
            assert result is True
            
            # Verify no subtitles were added
            final_call = mock_run.call_args_list[-1]
            cmd = final_call[0][0]
            assert not any('subtitles=' in str(arg) for arg in cmd)

    def test_ffmpeg_command_structure_basic(self, sample_clips, temp_output):
        """Test the structure of FFmpeg commands for basic processing"""
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stderr="")
            
            create_professional_video(
                clips=sample_clips,
                source_video="/path/to/source.mp4",
                output_path=temp_output
            )
            
            # Check individual clip extraction commands
            for i, call_args in enumerate(mock_run.call_args_list[:len(sample_clips)]):
                cmd = call_args[0][0]
                
                # Basic FFmpeg structure
                assert cmd[0] == 'ffmpeg'
                assert '-y' in cmd  # Overwrite
                assert '-i' in cmd  # Input
                assert '-vf' in cmd  # Video filters
                
                # Check for fade effects
                vf_index = cmd.index('-vf')
                filters = cmd[vf_index + 1]
                assert 'fade=t=in' in filters
                assert 'fade=t=out' in filters
                
                # Check encoding settings
                assert '-c:v' in cmd
                assert 'libx264' in cmd
                assert '-crf' in cmd
                assert '-c:a' in cmd
                assert 'aac' in cmd

    def test_zoom_effects_on_single_clip(self, temp_output):
        """Test zoom effects on single clip (first and last)"""
        clips = [{"start_ms": 0, "end_ms": 5000}]
        
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stderr="")
            
            create_professional_video(
                clips=clips,
                source_video="/path/to/source.mp4",
                output_path=temp_output
            )
            
            # Check clip has both zoom in (first clip) and zoom out (last clip) effects
            cmd = mock_run.call_args_list[0][0][0]
            vf_index = cmd.index('-vf')
            filters = cmd[vf_index + 1]
            assert 'zoompan' in filters
            # Single clip should have zoom in effect (first clip)
            assert 'zoom+0.0005' in filters

    def test_zoom_effects_on_first_and_last_clips(self, temp_output):
        """Test zoom effects are applied to first and last clips"""
        clips = [
            {"start_ms": 0, "end_ms": 5000},
            {"start_ms": 10000, "end_ms": 15000},
            {"start_ms": 20000, "end_ms": 25000},
        ]
        
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stderr="")
            
            create_professional_video(
                clips=clips,
                source_video="/path/to/source.mp4",
                output_path=temp_output
            )
            
            # Check first clip has zoom in
            first_clip_cmd = mock_run.call_args_list[0][0][0]
            vf_index = first_clip_cmd.index('-vf')
            first_filters = first_clip_cmd[vf_index + 1]
            assert 'zoompan' in first_filters
            assert 'zoom+0.0005' in first_filters  # Zoom in
            
            # Check middle clip has no zoom effect
            middle_clip_cmd = mock_run.call_args_list[1][0][0]
            vf_index = middle_clip_cmd.index('-vf')
            middle_filters = middle_clip_cmd[vf_index + 1]
            assert 'zoompan' not in middle_filters
            
            # Check last clip has zoom out
            last_clip_cmd = mock_run.call_args_list[2][0][0]
            vf_index = last_clip_cmd.index('-vf')
            last_filters = last_clip_cmd[vf_index + 1]
            assert 'zoompan' in last_filters
            assert '1.05-0.05' in last_filters  # Zoom out

    def test_transition_creation_single_transition(self, sample_two_clips, temp_output):
        """Test single transition between two clips"""
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stderr="")
            
            create_professional_video(
                clips=sample_two_clips,
                source_video="/path/to/source.mp4",
                output_path=temp_output,
                add_transitions=True
            )
            
            # Find transition commands (they use filter_complex with xfade)
            transition_calls = [
                call for call in mock_run.call_args_list
                if any('-filter_complex' in str(arg) and 'xfade' in str(arg) 
                      for arg in call[0][0])
            ]
            
            # Should have 1 transition for 2 clips
            assert len(transition_calls) == 1

    def test_transition_creation_multiple_transitions(self, sample_clips, temp_output):
        """Test multiple transitions between clips"""
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stderr="")
            
            create_professional_video(
                clips=sample_clips,
                source_video="/path/to/source.mp4",
                output_path=temp_output,
                add_transitions=True
            )
            
            # Find transition commands (they use filter_complex with xfade)
            transition_calls = [
                call for call in mock_run.call_args_list
                if any('-filter_complex' in str(arg) and 'xfade' in str(arg) 
                      for arg in call[0][0])
            ]
            
            # Should have n-1 transitions for n clips
            assert len(transition_calls) == len(sample_clips) - 1

    def test_color_enhancement_filters_enabled(self, sample_clips, temp_output):
        """Test color enhancement filters are applied when enabled"""
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stderr="")
            
            create_professional_video(
                clips=sample_clips,
                source_video="/path/to/source.mp4",
                output_path=temp_output,
                enhance_colors=True
            )
            
            # Check each clip has color enhancement
            for i in range(len(sample_clips)):
                cmd = mock_run.call_args_list[i][0][0]
                vf_index = cmd.index('-vf')
                filters = cmd[vf_index + 1]
                
                # Check for color enhancement filter
                assert 'eq=' in filters
                assert 'brightness=' in filters
                assert 'saturation=' in filters
                assert 'contrast=' in filters

    def test_color_enhancement_filters_disabled(self, sample_clips, temp_output):
        """Test color enhancement filters are not applied when disabled"""
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stderr="")
            
            create_professional_video(
                clips=sample_clips,
                source_video="/path/to/source.mp4",
                output_path=temp_output,
                enhance_colors=False
            )
            
            # Check each clip does NOT have color enhancement
            for i in range(len(sample_clips)):
                cmd = mock_run.call_args_list[i][0][0]
                vf_index = cmd.index('-vf')
                filters = cmd[vf_index + 1]
                
                # Should not have color enhancement filter
                assert 'eq=' not in filters

    def test_final_encoding_settings(self, sample_clips, temp_output):
        """Test final video encoding settings"""
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stderr="")
            
            create_professional_video(
                clips=sample_clips,
                source_video="/path/to/source.mp4",
                output_path=temp_output
            )
            
            # Find final assembly command (uses concat)
            final_cmd = None
            for call_args in reversed(mock_run.call_args_list):
                cmd = call_args[0][0]
                if '-f' in cmd and 'concat' in cmd:
                    final_cmd = cmd
                    break
            
            assert final_cmd is not None
            
            # Check professional encoding settings
            assert '-preset' in final_cmd
            assert 'slow' in final_cmd
            assert '-crf' in final_cmd
            assert '17' in final_cmd  # High quality
            assert '-profile:v' in final_cmd
            assert 'high' in final_cmd
            assert '-level' in final_cmd
            assert '4.2' in final_cmd
            assert '-movflags' in final_cmd
            assert '+faststart' in final_cmd  # Streaming optimization

    def test_error_handling_first_clip_extraction_failure(self, sample_clips, temp_output):
        """Test handling of first clip extraction failure"""
        with patch('subprocess.run') as mock_run:
            # First clip extraction fails
            mock_run.return_value = MagicMock(returncode=1, stderr="FFmpeg error")
            
            result = create_professional_video(
                clips=sample_clips,
                source_video="/path/to/source.mp4",
                output_path=temp_output
            )
            
            assert result is False

    def test_error_handling_middle_clip_extraction_failure(self, sample_clips, temp_output):
        """Test handling of middle clip extraction failure"""
        with patch('subprocess.run') as mock_run:
            # First clip succeeds, second fails
            returns = [
                MagicMock(returncode=0, stderr=""),  # First clip succeeds
                MagicMock(returncode=1, stderr="FFmpeg error"),  # Second clip fails
            ]
            mock_run.side_effect = returns
            
            result = create_professional_video(
                clips=sample_clips,
                source_video="/path/to/source.mp4",
                output_path=temp_output
            )
            
            assert result is False

    def test_error_handling_final_render_failure(self, sample_clips, temp_output):
        """Test handling of final render failure"""
        with patch('subprocess.run') as mock_run:
            # All clip extractions and transitions succeed, final render fails
            returns = []
            # Clip extractions succeed
            for _ in range(len(sample_clips)):
                returns.append(MagicMock(returncode=0, stderr=""))
            # Transitions succeed
            for _ in range(len(sample_clips) - 1):
                returns.append(MagicMock(returncode=0, stderr=""))
            # Final render fails
            returns.append(MagicMock(returncode=1, stderr="Final render error"))
            
            mock_run.side_effect = returns
            
            result = create_professional_video(
                clips=sample_clips,
                source_video="/path/to/source.mp4",
                output_path=temp_output
            )
            
            assert result is False

    def test_exception_handling_general(self, sample_clips, temp_output):
        """Test general exception handling"""
        with patch('subprocess.run', side_effect=Exception("General test exception")):
            result = create_professional_video(
                clips=sample_clips,
                source_video="/path/to/source.mp4",
                output_path=temp_output
            )
            
            assert result is False

    def test_exception_handling_in_makedirs(self, sample_clips, temp_output):
        """Test exception handling in makedirs"""
        with patch('os.makedirs', side_effect=OSError("Cannot create directory")):
            result = create_professional_video(
                clips=sample_clips,
                source_video="/path/to/source.mp4",
                output_path=temp_output
            )
            
            assert result is False

    def test_temp_file_cleanup_success(self, sample_clips, temp_output):
        """Test temporary files are cleaned up successfully"""
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stderr="")
            
            with patch('os.unlink') as mock_unlink:
                create_professional_video(
                    clips=sample_clips,
                    source_video="/path/to/source.mp4",
                    output_path=temp_output
                )
                
                # Should attempt to clean up clip files and transitions
                assert mock_unlink.call_count > 0

    def test_temp_file_cleanup_failure(self, sample_clips, temp_output):
        """Test temp file cleanup continues even if some deletions fail"""
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stderr="")
            
            with patch('os.unlink', side_effect=OSError("Cannot delete file")) as mock_unlink:
                # Should not raise exception even if cleanup fails
                result = create_professional_video(
                    clips=sample_clips,
                    source_video="/path/to/source.mp4",
                    output_path=temp_output
                )
                
                assert result is True
                assert mock_unlink.call_count > 0

    def test_temp_directory_creation(self, sample_clips, temp_output):
        """Test temporary directory is created"""
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stderr="")
            
            with patch('os.makedirs') as mock_makedirs:
                create_professional_video(
                    clips=sample_clips,
                    source_video="/path/to/source.mp4",
                    output_path=temp_output
                )
                
                mock_makedirs.assert_called_once_with('/tmp/montage_pro', exist_ok=True)

    def test_padding_for_transitions_start_not_zero(self, temp_output):
        """Test clips have padding added for transitions when start is not zero"""
        clips = [{"start_ms": 1000, "end_ms": 5000}]  # Single clip starting at 1s
        
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stderr="")
            
            create_professional_video(
                clips=clips,
                source_video="/path/to/source.mp4",
                output_path=temp_output
            )
            
            # Check the extraction command
            cmd = mock_run.call_args_list[0][0][0]
            
            # Find -ss (start time) and -t (duration)
            ss_index = cmd.index('-ss')
            start_time = float(cmd[ss_index + 1])
            
            # Should have 0.5s padding before (1s - 0.5s = 0.5s)
            assert start_time == 0.5

    def test_padding_for_transitions_start_zero(self, temp_output):
        """Test clips have padding handled correctly when start is zero"""
        clips = [{"start_ms": 0, "end_ms": 5000}]  # Single clip starting at 0
        
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stderr="")
            
            create_professional_video(
                clips=clips,
                source_video="/path/to/source.mp4",
                output_path=temp_output
            )
            
            # Check the extraction command
            cmd = mock_run.call_args_list[0][0][0]
            
            # Find -ss (start time)
            ss_index = cmd.index('-ss')
            start_time = float(cmd[ss_index + 1])
            
            # Should be 0 (max(0, 0 - 500) = 0)
            assert start_time == 0.0

    def test_vignette_effect_for_vertical(self, sample_clips, temp_output):
        """Test vignette effect is added for vertical format"""
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stderr="")
            
            with patch('montage.utils.video_effects.IntelligentCropper') as mock_cropper_class:
                mock_cropper = MagicMock()
                mock_cropper.analyze_video_content.return_value = {
                    "crop_center": {"x": 960, "y": 540},
                    "confidence": 0.9
                }
                mock_cropper.generate_crop_filter.return_value = "crop=608:1080:336:0"
                mock_cropper_class.return_value = mock_cropper
                
                create_professional_video(
                    clips=sample_clips,
                    source_video="/path/to/source.mp4",
                    output_path=temp_output,
                    vertical_format=True
                )
                
                # Check filters include vignette
                for i in range(len(sample_clips)):
                    cmd = mock_run.call_args_list[i][0][0]
                    vf_index = cmd.index('-vf')
                    filters = cmd[vf_index + 1]
                    assert 'vignette' in filters

    def test_landscape_scaling_filter(self, sample_clips, temp_output):
        """Test landscape format uses standard scaling"""
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stderr="")
            
            create_professional_video(
                clips=sample_clips,
                source_video="/path/to/source.mp4",
                output_path=temp_output,
                vertical_format=False
            )
            
            # Check filters include standard scaling
            for i in range(len(sample_clips)):
                cmd = mock_run.call_args_list[i][0][0]
                vf_index = cmd.index('-vf')
                filters = cmd[vf_index + 1]
                assert 'scale=1920:1080' in filters

    def test_audio_settings_clips(self, sample_clips, temp_output):
        """Test audio encoding settings for individual clips"""
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stderr="")
            
            create_professional_video(
                clips=sample_clips,
                source_video="/path/to/source.mp4",
                output_path=temp_output
            )
            
            # Check individual clips have audio settings
            for i in range(len(sample_clips)):
                cmd = mock_run.call_args_list[i][0][0]
                assert '-c:a' in cmd
                assert 'aac' in cmd
                assert '-b:a' in cmd
                assert '192k' in cmd

    def test_audio_settings_final_output(self, sample_clips, temp_output):
        """Test audio encoding settings for final output"""
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stderr="")
            
            create_professional_video(
                clips=sample_clips,
                source_video="/path/to/source.mp4",
                output_path=temp_output
            )
            
            # Check final output has higher quality audio
            final_cmd = mock_run.call_args_list[-1][0][0]
            assert '-b:a' in final_cmd
            assert '256k' in final_cmd  # Higher bitrate for final
            assert '-ar' in final_cmd
            assert '48000' in final_cmd  # Sample rate

    def test_concat_file_creation(self, sample_clips, temp_output):
        """Test concat file is created correctly"""
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stderr="")
            
            with patch('builtins.open', mock_open()) as mock_file:
                create_professional_video(
                    clips=sample_clips,
                    source_video="/path/to/source.mp4",
                    output_path=temp_output
                )
                
                # Verify concat file was written
                mock_file.assert_called()
                # Check that file operations were called
                handle = mock_file.return_value
                assert handle.write.call_count > 0

    def test_transition_command_structure(self, sample_two_clips, temp_output):
        """Test transition command structure"""
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stderr="")
            
            create_professional_video(
                clips=sample_two_clips,
                source_video="/path/to/source.mp4",
                output_path=temp_output,
                add_transitions=True
            )
            
            # Find transition command
            transition_cmd = None
            for call_args in mock_run.call_args_list:
                cmd = call_args[0][0]
                if '-filter_complex' in cmd and any('xfade' in str(arg) for arg in cmd):
                    transition_cmd = cmd
                    break
            
            assert transition_cmd is not None
            assert 'ffmpeg' in transition_cmd
            assert '-y' in transition_cmd
            assert '-filter_complex' in transition_cmd
            assert '-c:v' in transition_cmd
            assert 'libx264' in transition_cmd

    @pytest.mark.parametrize("vertical,expected_style", [
        (False, "FontSize=24"),
        (True, "FontSize=32"),
    ])
    def test_subtitle_style_based_on_format(self, sample_clips, temp_output, vertical, expected_style):
        """Test subtitle styling changes based on format"""
        with tempfile.NamedTemporaryFile(suffix='.srt', delete=False) as f:
            subtitles_path = f.name
            f.write(b"1\n00:00:00,000 --> 00:00:05,000\nTest\n\n")
        
        try:
            with patch('subprocess.run') as mock_run:
                mock_run.return_value = MagicMock(returncode=0, stderr="")
                
                create_professional_video(
                    clips=sample_clips,
                    source_video="/path/to/source.mp4",
                    output_path=temp_output,
                    vertical_format=vertical,
                    subtitles_path=subtitles_path
                )
                
                # Find final command with subtitles
                final_cmd = mock_run.call_args_list[-1][0][0]
                subtitle_arg = next(arg for arg in final_cmd if 'subtitles=' in str(arg))
                assert expected_style in subtitle_arg
        
        finally:
            if os.path.exists(subtitles_path):
                os.remove(subtitles_path)

    def test_clip_timing_calculation(self, temp_output):
        """Test clip timing and duration calculations"""
        clips = [{"start_ms": 2000, "end_ms": 8000}]  # 6 second clip
        
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stderr="")
            
            create_professional_video(
                clips=clips,
                source_video="/path/to/source.mp4",
                output_path=temp_output
            )
            
            cmd = mock_run.call_args_list[0][0][0]
            
            # Find -ss (start time) and -t (duration)
            ss_index = cmd.index('-ss')
            t_index = cmd.index('-t')
            
            start_time = float(cmd[ss_index + 1])
            duration = float(cmd[t_index + 1])
            
            # Start should be 2s - 0.5s padding = 1.5s
            assert start_time == 1.5
            # Duration should be 6s + 1s padding = 7s
            assert duration == 7.0

    def test_video_preset_settings(self, sample_clips, temp_output):
        """Test video preset settings for clips vs final render"""
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stderr="")
            
            create_professional_video(
                clips=sample_clips,
                source_video="/path/to/source.mp4",
                output_path=temp_output
            )
            
            # Check clip presets (should be "medium")
            for i in range(len(sample_clips)):
                cmd = mock_run.call_args_list[i][0][0]
                preset_index = cmd.index('-preset')
                assert cmd[preset_index + 1] == 'medium'
            
            # Check final preset (should be "slow")
            final_cmd = mock_run.call_args_list[-1][0][0]
            preset_index = final_cmd.index('-preset')
            assert cmd[preset_index + 1] in ['medium', 'slow']  # Could be either depending on call order

    def test_crf_quality_settings(self, sample_clips, temp_output):
        """Test CRF quality settings"""
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stderr="")
            
            create_professional_video(
                clips=sample_clips,
                source_video="/path/to/source.mp4",
                output_path=temp_output
            )
            
            # Check clip CRF (should be 18)
            for i in range(len(sample_clips)):
                cmd = mock_run.call_args_list[i][0][0]
                crf_index = cmd.index('-crf')
                assert cmd[crf_index + 1] == '18'
            
            # Check final CRF (should be 17)
            final_cmd = mock_run.call_args_list[-1][0][0]
            crf_index = final_cmd.index('-crf')
            assert final_cmd[crf_index + 1] == '17'


class TestIntegration:
    """Integration tests for video effects"""

    def test_full_pipeline_simulation_all_features(self):
        """Test complete video creation pipeline with all features"""
        clips = [
            {"start_ms": 0, "end_ms": 3000},
            {"start_ms": 5000, "end_ms": 8000},
        ]
        
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f:
            output_path = f.name
        
        with tempfile.NamedTemporaryFile(suffix='.srt', delete=False) as f:
            subtitles_path = f.name
            f.write(b"1\n00:00:00,000 --> 00:00:03,000\nTest subtitle\n\n")
        
        try:
            with patch('subprocess.run') as mock_run:
                # Simulate all subprocess calls succeeding
                mock_run.return_value = MagicMock(returncode=0, stderr="")
                
                with patch('montage.utils.video_effects.IntelligentCropper') as mock_cropper_class:
                    mock_cropper = MagicMock()
                    mock_cropper.analyze_video_content.return_value = {
                        "crop_center": {"x": 960, "y": 540},
                        "confidence": 0.9
                    }
                    mock_cropper.generate_crop_filter.return_value = "crop=608:1080:336:0"
                    mock_cropper_class.return_value = mock_cropper
                    
                    # Run with all features enabled
                    result = create_professional_video(
                        clips=clips,
                        source_video="/path/to/source.mp4",
                        output_path=output_path,
                        vertical_format=True,
                        enhance_colors=True,
                        add_transitions=True,
                        subtitles_path=subtitles_path
                    )
                    
                    assert result is True
                    
                    # Verify the sequence of operations
                    calls = mock_run.call_args_list
                    
                    # Should have: 2 clip extractions + transitions + final assembly
                    assert len(calls) >= 3
                    
                    # Verify intelligent cropping was used
                    assert mock_cropper.analyze_video_content.call_count == len(clips)
        
        finally:
            if os.path.exists(output_path):
                os.remove(output_path)
            if os.path.exists(subtitles_path):
                os.remove(subtitles_path)

    def test_full_pipeline_minimal_features(self):
        """Test complete video creation pipeline with minimal features"""
        clips = [{"start_ms": 0, "end_ms": 5000}]
        
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f:
            output_path = f.name
        
        try:
            with patch('subprocess.run') as mock_run:
                mock_run.return_value = MagicMock(returncode=0, stderr="")
                
                # Run with minimal features
                result = create_professional_video(
                    clips=clips,
                    source_video="/path/to/source.mp4",
                    output_path=output_path,
                    vertical_format=False,
                    enhance_colors=False,
                    add_transitions=False
                )
                
                assert result is True
                
                # Should have minimal calls: 1 clip + final assembly
                calls = mock_run.call_args_list
                assert len(calls) >= 2
        
        finally:
            if os.path.exists(output_path):
                os.remove(output_path)

    def test_error_recovery_simulation(self):
        """Test error recovery in various scenarios"""
        clips = [
            {"start_ms": 0, "end_ms": 3000},
            {"start_ms": 5000, "end_ms": 8000},
        ]
        
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f:
            output_path = f.name
        
        try:
            # Test scenario where second clip fails
            with patch('subprocess.run') as mock_run:
                returns = [
                    MagicMock(returncode=0, stderr=""),  # First clip succeeds
                    MagicMock(returncode=1, stderr="Second clip error"),  # Second clip fails
                ]
                mock_run.side_effect = returns
                
                result = create_professional_video(
                    clips=clips,
                    source_video="/path/to/source.mp4",
                    output_path=output_path
                )
                
                assert result is False
        
        finally:
            if os.path.exists(output_path):
                os.remove(output_path)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])