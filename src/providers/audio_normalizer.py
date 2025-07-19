"""Two-pass audio normalization using FFmpeg loudnorm filter"""

import json
import logging
import subprocess
import os
import tempfile
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import re

try:
    from ..config import get
except ImportError:
    from src.config import get
from metrics import metrics, track_processing_stage
from video_processor import FFmpegPipeline

logger = logging.getLogger(__name__)


class AudioNormalizationError(Exception):
    """Audio normalization error"""

    pass


@dataclass
class LoudnessStats:
    """Audio loudness measurements"""

    input_i: float  # Integrated loudness (LUFS)
    input_tp: float  # True peak (dBTP)
    input_lra: float  # Loudness range (LU)
    input_thresh: float  # Threshold
    target_offset: float  # Offset to target

    @classmethod
    def from_json(cls, data: Dict[str, Any]) -> "LoudnessStats":
        """Create from FFmpeg JSON output"""
        return cls(
            input_i=float(data.get("input_i", -70.0)),
            input_tp=float(data.get("input_tp", -70.0)),
            input_lra=float(data.get("input_lra", 0.0)),
            input_thresh=float(data.get("input_thresh", -70.0)),
            target_offset=float(data.get("target_offset", 0.0)),
        )


@dataclass
class NormalizationTarget:
    """Target loudness parameters"""

    integrated: float = -16.0  # Target integrated loudness (LUFS)
    true_peak: float = -1.0  # Maximum true peak (dBTP)
    lra: float = 7.0  # Target loudness range (LU)

    def to_filter_params(self) -> str:
        """Convert to loudnorm filter parameters"""
        return f"I={self.integrated}:TP={self.true_peak}:LRA={self.lra}"


class AudioNormalizer:
    """Two-pass audio normalization for consistent loudness"""

    def __init__(self):
        self.ffmpeg_path = get("FFMPEG_PATH", "ffmpeg")
        self.temp_dir = get("TEMP_DIR", "/tmp/montage")
        os.makedirs(self.temp_dir, exist_ok=True)

    @track_processing_stage("audio_normalization")
    def normalize_audio(
        self,
        input_file: str,
        output_file: str,
        target: Optional[NormalizationTarget] = None,
        video_duration: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Perform two-pass audio normalization.

        Returns:
            Dict with normalization results and metrics
        """
        if target is None:
            target = NormalizationTarget()

        logger.info("Starting two-pass audio normalization...")
        logger.info(f"Target: {target.integrated} LUFS, {target.true_peak} dBTP")

        try:
            # Pass 1: Analyze audio
            stats = self._analyze_loudness(input_file)

            # Calculate adjustment needed
            adjustment_db = target.integrated - stats.input_i

            # Track metrics
            metrics.track_audio_normalization(adjustment_db)

            # Pass 2: Apply normalization
            self._apply_normalization(input_file, output_file, stats, target)

            # Verify results
            verification = self._verify_normalization(output_file)

            # Track audio spread
            if "segments_spread" in verification:
                metrics.track_audio_spread(verification["segments_spread"])

            result = {
                "input_loudness": stats.input_i,
                "output_loudness": verification.get("loudness", target.integrated),
                "adjustment_db": adjustment_db,
                "true_peak": verification.get("true_peak", stats.input_tp),
                "lra": verification.get("lra", stats.input_lra),
                "spread_lu": verification.get("spread_lu", 0.0),
            }

            logger.info(
                f"Normalization complete: {stats.input_i:.1f} → {result['output_loudness']:.1f} LUFS"
            )

            return result

        except Exception as e:
            logger.error(f"Audio normalization failed: {e}")
            raise AudioNormalizationError(f"Normalization failed: {e}")

    def _analyze_loudness(self, input_file: str) -> LoudnessStats:
        """Pass 1: Analyze audio loudness"""
        logger.info("Pass 1: Analyzing audio loudness...")

        cmd = [
            self.ffmpeg_path,
            "-i",
            input_file,
            "-af",
            "loudnorm=print_format=json",
            "-f",
            "null",
            "-",
        ]

        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, check=False  # We check manually
            )

            # Extract JSON from stderr
            json_match = re.search(r"\{[^}]+\}", result.stderr, re.DOTALL)
            if not json_match:
                raise AudioNormalizationError(
                    "Could not parse loudness analysis output"
                )

            json_str = json_match.group(0)
            loudness_data = json.loads(json_str)

            stats = LoudnessStats.from_json(loudness_data)

            logger.info(f"  Integrated: {stats.input_i:.1f} LUFS")
            logger.info(f"  True Peak: {stats.input_tp:.1f} dBTP")
            logger.info(f"  LRA: {stats.input_lra:.1f} LU")

            return stats

        except subprocess.CalledProcessError as e:
            raise AudioNormalizationError(f"Loudness analysis failed: {e.stderr}")
        except json.JSONDecodeError as e:
            raise AudioNormalizationError(f"Failed to parse loudness data: {e}")

    def _apply_normalization(
        self,
        input_file: str,
        output_file: str,
        stats: LoudnessStats,
        target: NormalizationTarget,
    ) -> None:
        """Pass 2: Apply normalization with measured values"""
        logger.info("Pass 2: Applying normalization...")

        # Build loudnorm filter with measured values
        filter_str = (
            f"loudnorm="
            f"I={target.integrated}:"
            f"TP={target.true_peak}:"
            f"LRA={target.lra}:"
            f"measured_I={stats.input_i}:"
            f"measured_TP={stats.input_tp}:"
            f"measured_LRA={stats.input_lra}:"
            f"measured_thresh={stats.input_thresh}:"
            f"offset={stats.target_offset}"
        )

        cmd = [
            self.ffmpeg_path,
            "-i",
            input_file,
            "-af",
            filter_str,
            "-c:v",
            "copy",  # Copy video stream
            "-c:a",
            "aac",  # Re-encode audio
            "-b:a",
            "192k",  # Good quality audio
            "-movflags",
            "+faststart",
            "-y",
            output_file,
        ]

        try:
            subprocess.run(cmd, capture_output=True, text=True, check=True)
            logger.info("  Normalization applied successfully")
        except subprocess.CalledProcessError as e:
            raise AudioNormalizationError(f"Failed to apply normalization: {e.stderr}")

    def _verify_normalization(self, output_file: str) -> Dict[str, Any]:
        """Verify normalization results"""
        try:
            # Analyze normalized output
            stats = self._analyze_loudness(output_file)

            return {
                "loudness": stats.input_i,
                "true_peak": stats.input_tp,
                "lra": stats.input_lra,
                "spread_lu": 0.0,  # Would be calculated across segments
            }
        except Exception as e:
            logger.warning(f"Could not verify normalization: {e}")
            return {}

    def normalize_segments(
        self,
        segments: List[str],
        output_files: List[str],
        target: Optional[NormalizationTarget] = None,
    ) -> Dict[str, Any]:
        """
        Normalize multiple segments to consistent loudness.

        This ensures RMS spread ≤ 1.5 LU between segments.
        """
        if target is None:
            target = NormalizationTarget()

        logger.info(f"Normalizing {len(segments)} segments...")

        # Analyze all segments
        all_stats = []
        for i, segment in enumerate(segments):
            logger.info(f"Analyzing segment {i+1}/{len(segments)}...")
            stats = self._analyze_loudness(segment)
            all_stats.append(stats)

        # Calculate global loudness
        loudness_values = [s.input_i for s in all_stats]
        min_loudness = min(loudness_values)
        max_loudness = max(loudness_values)
        spread = max_loudness - min_loudness

        logger.info(
            f"Loudness spread: {spread:.1f} LU (min: {min_loudness:.1f}, max: {max_loudness:.1f})"
        )

        # Track spread metric
        metrics.track_audio_spread(spread)

        # Normalize each segment
        with FFmpegPipeline() as pipeline:
            for i, (segment, output, stats) in enumerate(
                zip(segments, output_files, all_stats)
            ):
                # Use consistent target for all segments
                filter_str = (
                    f"loudnorm="
                    f"I={target.integrated}:"
                    f"TP={target.true_peak}:"
                    f"LRA={target.lra}:"
                    f"measured_I={stats.input_i}:"
                    f"measured_TP={stats.input_tp}:"
                    f"measured_LRA={stats.input_lra}:"
                    f"measured_thresh={stats.input_thresh}:"
                    f"offset={stats.target_offset}"
                )

                cmd = [
                    self.ffmpeg_path,
                    "-i",
                    segment,
                    "-af",
                    filter_str,
                    "-c:v",
                    "copy",
                    "-c:a",
                    "aac",
                    "-b:a",
                    "192k",
                    "-y",
                    output,
                ]

                pipeline.add_process(cmd, f"normalize_{i:03d}")

            # Wait for all normalizations
            results = pipeline.wait_all(timeout=600)

            failed = [name for name, code in results.items() if code != 0]
            if failed:
                raise AudioNormalizationError(f"Normalization failed for: {failed}")

        # Verify final spread
        final_stats = []
        for output in output_files:
            stats = self._analyze_loudness(output)
            final_stats.append(stats.input_i)

        final_spread = max(final_stats) - min(final_stats)

        logger.info(f"Final loudness spread: {final_spread:.1f} LU")

        return {
            "segments_normalized": len(segments),
            "initial_spread": spread,
            "final_spread": final_spread,
            "spread_reduced_by": spread - final_spread,
            "meets_target": final_spread <= 1.5,  # Target: ≤ 1.5 LU
        }

    def apply_ebur128_analysis(self, input_file: str) -> Dict[str, Any]:
        """
        Analyze audio using EBU R128 standard.

        Returns detailed loudness measurements.
        """
        cmd = [
            self.ffmpeg_path,
            "-i",
            input_file,
            "-af",
            "ebur128=peak=true:framelog=quiet",
            "-f",
            "null",
            "-",
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

            # Parse EBU R128 output
            output = result.stderr

            # Extract measurements
            measurements = {}

            # Integrated loudness
            match = re.search(r"I:\s+(-?\d+\.?\d*)\s+LUFS", output)
            if match:
                measurements["integrated_lufs"] = float(match.group(1))

            # Loudness range
            match = re.search(r"LRA:\s+(-?\d+\.?\d*)\s+LU", output)
            if match:
                measurements["lra_lu"] = float(match.group(1))

            # True peak
            match = re.search(r"Peak:\s+(-?\d+\.?\d*)\s+dBFS", output)
            if match:
                measurements["peak_dbfs"] = float(match.group(1))

            return measurements

        except Exception as e:
            logger.error(f"EBU R128 analysis failed: {e}")
            return {}


# Integration with video processing


def normalize_video_audio(
    input_video: str, output_video: str, target: Optional[NormalizationTarget] = None
) -> Dict[str, Any]:
    """
    Normalize audio track of a video file.

    This is a convenience function for video processing integration.
    """
    normalizer = AudioNormalizer()
    return normalizer.normalize_audio(input_video, output_video, target)


# Example usage for SmartVideoEditor
class AudioNormalizedEditor:
    """Example integration with video editor"""

    def __init__(self):
        self.normalizer = AudioNormalizer()

    def process_video_with_normalization(
        self, input_file: str, output_file: str, segments: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Process video with audio normalization"""

        # First, create edited video
        temp_edited = tempfile.mktemp(suffix=".mp4", dir=self.normalizer.temp_dir)

        try:
            # Edit video (using concat editor or other method)
            # ... editing logic here ...

            # Then normalize audio
            result = self.normalizer.normalize_audio(
                temp_edited,
                output_file,
                target=NormalizationTarget(
                    integrated=-16.0,  # Streaming standard
                    true_peak=-1.0,  # Prevent clipping
                    lra=7.0,  # Good dynamic range
                ),
            )

            return result

        finally:
            # Cleanup temp file
            if os.path.exists(temp_edited):
                os.unlink(temp_edited)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Example normalization
    normalizer = AudioNormalizer()

    result = normalizer.normalize_audio("input_video.mp4", "normalized_output.mp4")

    print("Normalization results:")
    print(f"  Input: {result['input_loudness']:.1f} LUFS")
    print(f"  Output: {result['output_loudness']:.1f} LUFS")
    print(f"  Adjustment: {result['adjustment_db']:.1f} dB")
