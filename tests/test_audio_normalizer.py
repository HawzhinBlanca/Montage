"""Test two-pass audio normalization"""

import pytest
import tempfile
from unittest.mock import patch, MagicMock
import subprocess
from audio_normalizer import (
    AudioNormalizer,
    LoudnessStats,
    NormalizationTarget,
    AudioNormalizationError,
    normalize_video_audio,
)


class TestLoudnessStats:
    """Test LoudnessStats dataclass"""

    def test_from_json(self):
        """Test creating stats from JSON"""
        json_data = {
            "input_i": "-23.5",
            "input_tp": "-3.2",
            "input_lra": "8.7",
            "input_thresh": "-33.5",
            "target_offset": "7.5",
        }

        stats = LoudnessStats.from_json(json_data)

        assert stats.input_i == -23.5
        assert stats.input_tp == -3.2
        assert stats.input_lra == 8.7
        assert stats.input_thresh == -33.5
        assert stats.target_offset == 7.5

    def test_from_json_missing_values(self):
        """Test with missing values uses defaults"""
        json_data = {"input_i": "-20.0"}

        stats = LoudnessStats.from_json(json_data)

        assert stats.input_i == -20.0
        assert stats.input_tp == -70.0  # Default
        assert stats.input_lra == 0.0  # Default


class TestNormalizationTarget:
    """Test NormalizationTarget dataclass"""

    def test_default_values(self):
        """Test default target values"""
        target = NormalizationTarget()

        assert target.integrated == -16.0
        assert target.true_peak == -1.0
        assert target.lra == 7.0

    def test_to_filter_params(self):
        """Test converting to filter parameters"""
        target = NormalizationTarget(integrated=-18.0, true_peak=-2.0, lra=10.0)

        params = target.to_filter_params()

        assert params == "I=-18.0:TP=-2.0:LRA=10.0"


class TestAudioNormalizer:
    """Test audio normalizer functionality"""

    @pytest.fixture
    def normalizer(self):
        """Create normalizer instance"""
        with tempfile.TemporaryDirectory() as temp_dir:
            normalizer = AudioNormalizer()
            normalizer.temp_dir = temp_dir
            yield normalizer

    @patch("subprocess.run")
    def test_analyze_loudness(self, mock_run, normalizer):
        """Test loudness analysis (pass 1)"""
        # Mock FFmpeg output with loudness JSON
        mock_output = """
        [Parsed_loudnorm_0 @ 0x7f8b8c004080] 
        {
            "input_i" : "-23.54",
            "input_tp" : "-3.21",
            "input_lra" : "8.70",
            "input_thresh" : "-33.70",
            "target_offset" : "7.54"
        }
        """

        mock_run.return_value = MagicMock(returncode=0, stderr=mock_output, stdout="")

        stats = normalizer._analyze_loudness("input.mp4")

        # Verify command
        cmd = mock_run.call_args[0][0]
        assert "-af" in cmd
        assert "loudnorm=print_format=json" in cmd
        assert "-f" in cmd
        assert "null" in cmd

        # Verify parsed stats
        assert stats.input_i == -23.54
        assert stats.input_tp == -3.21
        assert stats.input_lra == 8.70

    @patch("subprocess.run")
    def test_analyze_loudness_parse_error(self, mock_run, normalizer):
        """Test handling of parse errors"""
        # Mock output without valid JSON
        mock_run.return_value = MagicMock(
            returncode=0, stderr="No JSON data here", stdout=""
        )

        with pytest.raises(AudioNormalizationError) as exc_info:
            normalizer._analyze_loudness("input.mp4")

        assert "Could not parse" in str(exc_info.value)

    @patch("subprocess.run")
    def test_apply_normalization(self, mock_run, normalizer):
        """Test normalization application (pass 2)"""
        mock_run.return_value = MagicMock(returncode=0)

        stats = LoudnessStats(
            input_i=-23.5,
            input_tp=-3.2,
            input_lra=8.7,
            input_thresh=-33.5,
            target_offset=7.5,
        )

        target = NormalizationTarget()

        normalizer._apply_normalization("input.mp4", "output.mp4", stats, target)

        # Verify command
        cmd = mock_run.call_args[0][0]
        assert "-af" in cmd

        # Check filter includes measured values
        filter_idx = cmd.index("-af") + 1
        filter_str = cmd[filter_idx]
        assert "measured_I=-23.5" in filter_str
        assert "measured_TP=-3.2" in filter_str
        assert "measured_LRA=8.7" in filter_str
        assert "I=-16.0" in filter_str  # Target

        # Check video copy and audio encoding
        assert "-c:v" in cmd
        assert "copy" in cmd
        assert "-c:a" in cmd
        assert "aac" in cmd

    @patch("audio_normalizer.AudioNormalizer._apply_normalization")
    @patch("audio_normalizer.AudioNormalizer._verify_normalization")
    @patch("audio_normalizer.AudioNormalizer._analyze_loudness")
    def test_normalize_audio_complete(
        self, mock_analyze, mock_verify, mock_apply, normalizer
    ):
        """Test complete normalization workflow"""
        # Mock analysis results
        mock_analyze.return_value = LoudnessStats(
            input_i=-23.5,
            input_tp=-3.2,
            input_lra=8.7,
            input_thresh=-33.5,
            target_offset=7.5,
        )

        # Mock verification
        mock_verify.return_value = {
            "loudness": -16.1,
            "true_peak": -1.1,
            "lra": 7.2,
            "spread_lu": 0.8,
        }

        result = normalizer.normalize_audio("input.mp4", "output.mp4")

        # Verify workflow
        mock_analyze.assert_called_once_with("input.mp4")
        mock_apply.assert_called_once()
        mock_verify.assert_called_once_with("output.mp4")

        # Check result
        assert result["input_loudness"] == -23.5
        assert result["output_loudness"] == -16.1
        assert result["adjustment_db"] == -16.0 - (-23.5)  # 7.5 dB
        assert result["spread_lu"] == 0.8

    @patch("subprocess.run")
    def test_normalize_segments(self, mock_run, normalizer):
        """Test normalizing multiple segments"""
        # Mock loudness analysis for 3 segments
        analysis_outputs = [
            '{"input_i": "-20.0", "input_tp": "-2.0", "input_lra": "7.0", "input_thresh": "-30.0", "target_offset": "4.0"}',
            '{"input_i": "-25.0", "input_tp": "-4.0", "input_lra": "9.0", "input_thresh": "-35.0", "target_offset": "9.0"}',
            '{"input_i": "-22.0", "input_tp": "-3.0", "input_lra": "8.0", "input_thresh": "-32.0", "target_offset": "6.0"}',
        ]

        # Setup mock to return different outputs for analysis calls
        call_count = 0

        def side_effect(*args, **kwargs):
            nonlocal call_count
            if "loudnorm=print_format=json" in args[0]:
                # Analysis call
                output = analysis_outputs[call_count % 3]
                call_count += 1
                return MagicMock(returncode=0, stderr=output)
            else:
                # Normalization call
                return MagicMock(returncode=0)

        mock_run.side_effect = side_effect

        segments = ["seg1.mp4", "seg2.mp4", "seg3.mp4"]
        outputs = ["out1.mp4", "out2.mp4", "out3.mp4"]

        result = normalizer.normalize_segments(segments, outputs)

        # Check spread calculation
        # Min: -25.0, Max: -20.0, Spread: 5.0 LU
        assert result["initial_spread"] == 5.0
        assert result["segments_normalized"] == 3

        # Verify normalization was applied to each segment
        assert mock_run.call_count >= 6  # 3 analysis + 3 normalization

    @patch("subprocess.run")
    def test_apply_ebur128_analysis(self, mock_run, normalizer):
        """Test EBU R128 analysis"""
        # Mock ebur128 output
        mock_output = """
        [Parsed_ebur128_0 @ 0x7f8b8c004080] Summary:
        
          Integrated loudness:
            I:         -16.1 LUFS
            Threshold: -26.1 LUFS
        
          Loudness range:
            LRA:         7.2 LU
            Threshold: -36.1 LUFS
        
          True peak:
            Peak:        -0.9 dBFS
        """

        mock_run.return_value = MagicMock(returncode=0, stderr=mock_output, stdout="")

        measurements = normalizer.apply_ebur128_analysis("input.mp4")

        # Verify measurements
        assert measurements["integrated_lufs"] == -16.1
        assert measurements["lra_lu"] == 7.2
        assert measurements["peak_dbfs"] == -0.9


class TestIntegration:
    """Integration tests"""

    @patch("subprocess.run")
    def test_normalize_video_audio(self, mock_run):
        """Test the convenience function"""
        # Mock successful normalization
        mock_run.side_effect = [
            # Analysis
            MagicMock(
                returncode=0,
                stderr='{"input_i": "-20.0", "input_tp": "-2.0", "input_lra": "8.0", "input_thresh": "-30.0", "target_offset": "4.0"}',
            ),
            # Normalization
            MagicMock(returncode=0),
            # Verification
            MagicMock(
                returncode=0,
                stderr='{"input_i": "-16.0", "input_tp": "-1.0", "input_lra": "7.0", "input_thresh": "-26.0", "target_offset": "0.0"}',
            ),
        ]

        result = normalize_video_audio("input.mp4", "output.mp4")

        assert result["input_loudness"] == -20.0
        assert result["output_loudness"] == -16.0
        assert result["adjustment_db"] == 4.0

    @patch("subprocess.run")
    def test_meets_spread_requirement(self, mock_run):
        """Test verification of spread requirement (≤ 1.5 LU)"""
        normalizer = AudioNormalizer()

        # Mock segments with good spread (1.0 LU)
        analysis_outputs = [
            '{"input_i": "-16.0", "input_tp": "-1.0", "input_lra": "7.0", "input_thresh": "-26.0", "target_offset": "0.0"}',
            '{"input_i": "-16.5", "input_tp": "-1.0", "input_lra": "7.0", "input_thresh": "-26.5", "target_offset": "0.5"}',
            '{"input_i": "-17.0", "input_tp": "-1.0", "input_lra": "7.0", "input_thresh": "-27.0", "target_offset": "1.0"}',
        ]

        call_count = 0

        def side_effect(*args, **kwargs):
            nonlocal call_count
            if "loudnorm=print_format=json" in args[0]:
                output = analysis_outputs[call_count % 3]
                call_count += 1
                return MagicMock(returncode=0, stderr=output)
            else:
                return MagicMock(returncode=0)

        mock_run.side_effect = side_effect

        segments = ["seg1.mp4", "seg2.mp4", "seg3.mp4"]
        outputs = ["out1.mp4", "out2.mp4", "out3.mp4"]

        result = normalizer.normalize_segments(segments, outputs)

        # Final spread: -16.0 to -17.0 = 1.0 LU
        assert result["final_spread"] == 1.0
        assert result["meets_target"] is True  # ≤ 1.5 LU


class TestErrorHandling:
    """Test error handling"""

    @patch("subprocess.run")
    def test_ffmpeg_error(self, mock_run):
        """Test handling of FFmpeg errors"""
        mock_run.side_effect = subprocess.CalledProcessError(
            1, "ffmpeg", stderr="Error: Invalid input"
        )

        normalizer = AudioNormalizer()

        with pytest.raises(AudioNormalizationError):
            normalizer._analyze_loudness("bad_input.mp4")

    @patch("subprocess.run")
    def test_invalid_json_response(self, mock_run):
        """Test handling of invalid JSON in response"""
        mock_run.return_value = MagicMock(returncode=0, stderr="{ invalid json }")

        normalizer = AudioNormalizer()

        with pytest.raises(AudioNormalizationError) as exc_info:
            normalizer._analyze_loudness("input.mp4")

        assert "parse" in str(exc_info.value).lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
