"""Test color space validation and conversion"""

import pytest
import json
from unittest.mock import patch, MagicMock
from color_converter import (
    ColorSpaceConverter,
    ColorSpaceInfo,
    ColorConversionError,
    ensure_bt709_output,
    get_safe_color_filter,
)


class TestColorSpaceInfo:
    """Test ColorSpaceInfo dataclass"""

    def test_from_ffprobe_sdr(self):
        """Test creating from SDR stream data"""
        stream_data = {
            "color_space": "bt709",
            "color_primaries": "bt709",
            "color_transfer": "bt709",
            "color_range": "tv",
        }

        info = ColorSpaceInfo.from_ffprobe(stream_data)

        assert info.color_space == "bt709"
        assert info.color_primaries == "bt709"
        assert info.color_transfer == "bt709"
        assert info.color_range == "tv"
        assert info.is_hdr is False

    def test_from_ffprobe_hdr(self):
        """Test HDR detection"""
        # HDR with BT.2020
        stream_data = {
            "color_space": "bt2020nc",
            "color_primaries": "bt2020",
            "color_transfer": "smpte2084",
            "color_range": "tv",
        }

        info = ColorSpaceInfo.from_ffprobe(stream_data)

        assert info.is_hdr is True

        # HDR with HLG
        stream_data["color_transfer"] = "arib-std-b67"
        info = ColorSpaceInfo.from_ffprobe(stream_data)
        assert info.is_hdr is True

    def test_from_ffprobe_missing_data(self):
        """Test with missing color data"""
        stream_data = {}

        info = ColorSpaceInfo.from_ffprobe(stream_data)

        assert info.color_space == "unknown"
        assert info.color_primaries == "unknown"
        assert info.color_transfer == "unknown"
        assert info.color_range == "tv"  # Default
        assert info.is_hdr is False


class TestColorSpaceConverter:
    """Test color space converter functionality"""

    @pytest.fixture
    def converter(self):
        """Create converter instance"""
        return ColorSpaceConverter()

    @patch("subprocess.run")
    def test_analyze_color_space(self, mock_run, converter):
        """Test color space analysis"""
        # Mock ffprobe output
        ffprobe_output = {
            "streams": [
                {
                    "codec_type": "video",
                    "color_space": "bt709",
                    "color_primaries": "bt709",
                    "color_transfer": "bt709",
                    "color_range": "tv",
                }
            ]
        }

        mock_run.return_value = MagicMock(
            returncode=0, stdout=json.dumps(ffprobe_output), stderr=""
        )

        info = converter.analyze_color_space("input.mp4")

        # Verify command
        cmd = mock_run.call_args[0][0]
        assert converter.ffprobe_path in cmd
        assert "-select_streams" in cmd
        assert "v:0" in cmd
        assert "-of" in cmd
        assert "json" in cmd

        # Verify result
        assert info.color_space == "bt709"
        assert info.is_hdr is False

    @patch("subprocess.run")
    def test_analyze_color_space_no_video(self, mock_run, converter):
        """Test handling of no video stream"""
        mock_run.return_value = MagicMock(
            returncode=0, stdout='{"streams": []}', stderr=""
        )

        with pytest.raises(ColorConversionError) as exc_info:
            converter.analyze_color_space("audio_only.mp3")

        assert "No video stream" in str(exc_info.value)

    @patch("color_converter.ColorSpaceConverter.analyze_color_space")
    def test_validate_sdr_input(self, mock_analyze, converter):
        """Test SDR validation"""
        # Test SDR input
        mock_analyze.return_value = ColorSpaceInfo(
            color_space="bt709",
            color_primaries="bt709",
            color_transfer="bt709",
            color_range="tv",
            is_hdr=False,
        )

        is_valid, error = converter.validate_sdr_input("sdr_video.mp4")

        assert is_valid is True
        assert error == ""

        # Test HDR input
        mock_analyze.return_value = ColorSpaceInfo(
            color_space="bt2020nc",
            color_primaries="bt2020",
            color_transfer="smpte2084",
            color_range="tv",
            is_hdr=True,
        )

        is_valid, error = converter.validate_sdr_input("hdr_video.mp4")

        assert is_valid is False
        assert "HDR input not supported" in error

    def test_build_color_conversion_filter(self, converter):
        """Test filter building"""
        filter_str = converter.build_color_conversion_filter()

        # Check filter components
        assert "zscale=t=linear:npl=100" in filter_str
        assert "format=gbrpf32le" in filter_str
        assert "zscale=p=bt709:t=bt709:m=bt709:r=tv" in filter_str
        assert "format=yuv420p" in filter_str

    @patch("subprocess.run")
    @patch("color_converter.ColorSpaceConverter.analyze_color_space")
    def test_convert_to_bt709(self, mock_analyze, mock_run, converter):
        """Test BT.709 conversion"""
        # Mock source as SDR
        mock_analyze.side_effect = [
            # First call - source analysis
            ColorSpaceInfo("bt601", "bt470bg", "bt470bg", "tv", False),
            # Second call - output verification
            ColorSpaceInfo("bt709", "bt709", "bt709", "tv", False),
        ]

        mock_run.return_value = MagicMock(returncode=0)

        result = converter.convert_to_bt709("input.mp4", "output.mp4")

        # Verify command
        cmd = mock_run.call_args[0][0]
        assert "-vf" in cmd
        assert "-colorspace" in cmd
        assert "bt709" in cmd[cmd.index("-colorspace") + 1]
        assert "-color_primaries" in cmd
        assert "-color_trc" in cmd

        # Verify result
        assert result["source_primaries"] == "bt470bg"
        assert result["output_primaries"] == "bt709"
        assert result["conversion_successful"] is True

    @patch("color_converter.ColorSpaceConverter.analyze_color_space")
    def test_convert_to_bt709_hdr_rejection(self, mock_analyze, converter):
        """Test HDR input rejection during conversion"""
        # Mock HDR input
        mock_analyze.return_value = ColorSpaceInfo(
            "bt2020nc", "bt2020", "smpte2084", "tv", True
        )

        with pytest.raises(ColorConversionError) as exc_info:
            converter.convert_to_bt709("hdr_input.mp4", "output.mp4")

        assert "HDR input not supported" in str(exc_info.value)

    def test_get_encoding_color_params(self, converter):
        """Test encoding parameter generation"""
        params = converter.get_encoding_color_params()

        assert params["-colorspace"] == "bt709"
        assert params["-color_primaries"] == "bt709"
        assert params["-color_trc"] == "bt709"
        assert params["-color_range"] == "tv"

    def test_build_safe_encoding_command(self, converter):
        """Test safe encoding command building"""
        cmd = converter.build_safe_encoding_command(
            "input.mp4", "output.mp4", video_filters="scale=1920:1080"
        )

        # Check command structure
        assert converter.ffmpeg_path == cmd[0]
        assert "-i" in cmd
        assert "input.mp4" in cmd

        # Check filter includes both user filter and color conversion
        filter_idx = cmd.index("-vf") + 1
        filter_str = cmd[filter_idx]
        assert "scale=1920:1080" in filter_str
        assert "zscale" in filter_str

        # Check color parameters
        assert "-colorspace" in cmd
        assert "bt709" in cmd[cmd.index("-colorspace") + 1]


class TestIntegrationFunctions:
    """Test integration helper functions"""

    @patch("color_converter.ColorSpaceConverter.validate_sdr_input")
    @patch("color_converter.ColorSpaceConverter.convert_to_bt709")
    def test_ensure_bt709_output(self, mock_convert, mock_validate):
        """Test ensure_bt709_output convenience function"""
        mock_validate.return_value = (True, "")
        mock_convert.return_value = {
            "conversion_successful": True,
            "output_primaries": "bt709",
        }

        result = ensure_bt709_output(
            "input.mp4", "output.mp4", preserve_filters="denoise"
        )

        # Verify validation was called
        mock_validate.assert_called_once_with("input.mp4")

        # Verify conversion was called with filters
        mock_convert.assert_called_once_with(
            "input.mp4", "output.mp4", additional_filters="denoise"
        )

    def test_get_safe_color_filter(self):
        """Test filter string getter"""
        filter_str = get_safe_color_filter()

        assert "zscale" in filter_str
        assert "bt709" in filter_str
        assert "format=yuv420p" in filter_str


class TestColorSafeVideoEditor:
    """Test example editor integration"""

    @patch("subprocess.run")
    @patch("color_converter.ColorSpaceConverter.validate_sdr_input")
    @patch("color_converter.ColorSpaceConverter.analyze_color_space")
    def test_edit_with_color_safety(self, mock_analyze, mock_validate, mock_run):
        """Test color-safe editing"""
        from color_converter import ColorSafeVideoEditor

        # Mock validation
        mock_validate.return_value = (True, "")

        # Mock color analysis
        mock_analyze.return_value = ColorSpaceInfo(
            "bt709", "bt709", "bt709", "tv", False
        )

        # Mock ffmpeg execution
        mock_run.return_value = MagicMock(returncode=0)

        editor = ColorSafeVideoEditor()
        result = editor.edit_with_color_safety(
            "input.mp4", "output.mp4", edit_filters="crop=1920:1080"
        )

        # Verify command was built with safety
        cmd = mock_run.call_args[0][0]
        assert any("bt709" in str(arg) for arg in cmd)

        # Verify result
        assert result["output_primaries"] == "bt709"
        assert result["is_bt709"] is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
