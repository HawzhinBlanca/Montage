"""Concat demuxer-based video editor for efficient segment joining"""

import os
import subprocess
import logging
import time
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
import uuid
from legacy_config import Config
from .video_processor import FFmpegPipeline
from ..core.metrics import metrics

logger = logging.getLogger(__name__)


class FFmpegError(Exception):
    """FFmpeg command execution error"""

    pass


@dataclass
class EditSegment:
    """Segment with editing parameters"""

    source_file: str
    start_time: float
    end_time: float
    segment_id: str = None
    transition_type: str = "fade"  # fade, dissolve, wipe, etc.
    transition_duration: float = 0.5

    def __post_init__(self):
        if not self.segment_id:
            self.segment_id = str(uuid.uuid4())[:8]

    @property
    def duration(self):
        return self.end_time - self.start_time


class ConcatEditor:
    """Video editor using concat demuxer for efficient processing"""

    def __init__(self):
        self.ffmpeg_path = Config.FFMPEG_PATH
        self.temp_dir = Config.TEMP_DIR
        os.makedirs(self.temp_dir, exist_ok=True)

    def execute_edit(
        self,
        segments: List[EditSegment],
        output_file: str,
        apply_transitions: bool = True,
        video_codec: str = "libx264",
        audio_codec: str = "aac",
    ) -> Dict[str, Any]:
        """
        Execute video edit using concat demuxer approach.

        This method:
        1. Extracts segments to temporary files
        2. Creates concat list file
        3. Concatenates using concat demuxer
        4. Applies transitions with minimal filter complexity
        """
        start_time = time.time()
        temp_files = []

        try:
            with metrics.track_processing_time("editing"):
                # Step 1: Extract segments to temporary files
                logger.info(f"Extracting {len(segments)} segments...")
                temp_files = self._extract_segments_to_files(segments)

                # Step 2: Create concat list
                concat_list_path = self._create_concat_list(temp_files)

                # Step 3: Concatenate segments
                if apply_transitions and len(segments) > 1:
                    # Concat with transitions
                    self._concat_with_transitions(
                        concat_list_path,
                        output_file,
                        segments,
                        video_codec,
                        audio_codec,
                    )
                else:
                    # Simple concat without transitions
                    self._simple_concat(
                        concat_list_path, output_file, video_codec, audio_codec
                    )

                # Calculate metrics
                elapsed = time.time() - start_time
                total_duration = sum(s.duration for s in segments)

                result = {
                    "segments_processed": len(segments),
                    "total_duration": total_duration,
                    "processing_time": elapsed,
                    "processing_ratio": (
                        elapsed / total_duration if total_duration > 0 else 0
                    ),
                    "output_file": output_file,
                }

                logger.info(
                    f"Edit completed in {elapsed:.1f}s (ratio: {result['processing_ratio']:.2f}x)"
                )

                return result

        finally:
            # Cleanup temporary files
            self._cleanup_temp_files(temp_files)
            if "concat_list_path" in locals():
                try:
                    os.unlink(concat_list_path)
                except:
                    pass

    def _extract_segments_to_files(self, segments: List[EditSegment]) -> List[str]:
        """Extract segments to temporary files in parallel"""
        temp_files = []

        with FFmpegPipeline() as pipeline:
            for i, segment in enumerate(segments):
                # Create temp file path
                temp_file = os.path.join(
                    self.temp_dir, f"segment_{segment.segment_id}.ts"
                )
                temp_files.append(temp_file)

                # Build extraction command
                cmd = [
                    self.ffmpeg_path,
                    "-ss",
                    str(segment.start_time),
                    "-t",
                    str(segment.duration),
                    "-i",
                    segment.source_file,
                    "-c",
                    "copy",  # No re-encoding
                    "-f",
                    "mpegts",  # MPEG-TS format for concatenation
                    "-avoid_negative_ts",
                    "make_zero",  # Fix timestamp issues
                    "-y",
                    temp_file,
                ]

                # Add to pipeline
                pipeline.add_process(cmd, f"extract_{i:03d}")

            # Wait for all extractions to complete
            results = pipeline.wait_all(timeout=600)  # 10 minute timeout

            # Check for failures
            failed = [name for name, code in results.items() if code != 0]
            if failed:
                raise FFmpegError(f"Segment extraction failed for: {failed}")

        return temp_files

    def _create_concat_list(self, files: List[str]) -> str:
        """Create concat demuxer input file"""
        concat_list_path = os.path.join(
            self.temp_dir, f"concat_{uuid.uuid4().hex[:8]}.txt"
        )

        with open(concat_list_path, "w") as f:
            for file_path in files:
                # Use safe format for concat demuxer
                f.write(f"file '{file_path}'\n")

        logger.debug(f"Created concat list with {len(files)} files")
        return concat_list_path

    def _simple_concat(
        self, concat_list: str, output_file: str, video_codec: str, audio_codec: str
    ) -> None:
        """Simple concatenation without transitions"""
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
            "-preset",
            "fast",
            "-c:a",
            audio_codec,
            "-movflags",
            "+faststart",
            "-y",
            output_file,
        ]

        logger.info("Concatenating segments without transitions...")

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            logger.debug("Concat completed successfully")
        except subprocess.CalledProcessError as e:
            logger.error(f"Concat failed: {e.stderr}")
            raise FFmpegError(f"Concatenation failed: {e.stderr}")

    def _concat_with_transitions(
        self,
        concat_list: str,
        output_file: str,
        segments: List[EditSegment],
        video_codec: str,
        audio_codec: str,
    ) -> None:
        """Concatenate with xfade transitions between segments"""

        # For many segments, use a more efficient approach
        if len(segments) > 10:
            self._concat_with_batch_transitions(
                concat_list, output_file, segments, video_codec, audio_codec
            )
            return

        # Build xfade filter for smaller segment counts
        filter_complex = self._build_xfade_filter(segments)

        # Verify filter length constraint
        if len(filter_complex) > 300:
            logger.warning(
                f"Filter complex too long ({len(filter_complex)} chars), using batch approach"
            )
            self._concat_with_batch_transitions(
                concat_list, output_file, segments, video_codec, audio_codec
            )
            return

        cmd = [
            self.ffmpeg_path,
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            concat_list,
            "-filter_complex",
            filter_complex,
            "-map",
            "[v]",
            "-map",
            "[a]",
            "-c:v",
            video_codec,
            "-preset",
            "fast",
            "-c:a",
            audio_codec,
            "-movflags",
            "+faststart",
            "-y",
            output_file,
        ]

        logger.info(
            f"Applying transitions (filter length: {len(filter_complex)} chars)..."
        )

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            logger.debug("Transitions applied successfully")
        except subprocess.CalledProcessError as e:
            logger.error(f"Transition application failed: {e.stderr}")
            raise FFmpegError(f"Failed to apply transitions: {e.stderr}")

    def _build_xfade_filter(self, segments: List[EditSegment]) -> str:
        """Build minimal xfade filter string"""
        if len(segments) < 2:
            return "[0:v]null[v];[0:a]anull[a]"

        # Calculate transition points
        points = []
        current_time = 0
        for i, segment in enumerate(segments[:-1]):
            current_time += segment.duration - (segment.transition_duration / 2)
            points.append(current_time)

        # Build compact filter
        # Use xfadeall for efficiency with multiple transitions
        transitions = ":".join([f"d={s.transition_duration}" for s in segments[:-1]])
        offsets = ":".join([f"{p:.2f}" for p in points])

        # Compact format
        filter_str = f"[0:v]xfadeall=transitions={len(points)}:{transitions}:offsets={offsets}[v];[0:a]acopy[a]"

        return filter_str

    def _concat_with_batch_transitions(
        self,
        concat_list: str,
        output_file: str,
        segments: List[EditSegment],
        video_codec: str,
        audio_codec: str,
    ) -> None:
        """Handle large numbers of segments by batching transitions"""
        # First, do simple concatenation
        temp_concat = os.path.join(
            self.temp_dir, f"temp_concat_{uuid.uuid4().hex[:8]}.mp4"
        )
        self._simple_concat(concat_list, temp_concat, video_codec, audio_codec)

        # Then apply transitions in post
        try:
            # Calculate all transition points
            transition_points = []
            current_time = 0

            for segment in segments[:-1]:
                current_time += segment.duration
                transition_points.append(
                    {"time": current_time, "duration": segment.transition_duration}
                )

            # Apply fade transitions at cut points
            self._apply_fade_transitions(temp_concat, output_file, transition_points)

        finally:
            # Cleanup temp file
            try:
                os.unlink(temp_concat)
            except:
                pass

    def _apply_fade_transitions(
        self,
        input_file: str,
        output_file: str,
        transition_points: List[Dict[str, float]],
    ) -> None:
        """Apply fade transitions at specified points"""
        # Build fade filter
        fade_filters = []

        for point in transition_points:
            t = point["time"]
            d = point["duration"]
            # Fade out before cut
            fade_filters.append(f"fade=t=out:st={t - d}:d={d/2}:alpha=1")
            # Fade in after cut
            fade_filters.append(f"fade=t=in:st={t}:d={d/2}:alpha=1")

        filter_complex = ",".join(fade_filters)

        cmd = [
            self.ffmpeg_path,
            "-i",
            input_file,
            "-vf",
            filter_complex,
            "-c:a",
            "copy",  # Don't re-encode audio
            "-y",
            output_file,
        ]

        subprocess.run(cmd, check=True, capture_output=True)

    def _cleanup_temp_files(self, files: List[str]) -> None:
        """Clean up temporary segment files"""
        for file_path in files:
            try:
                if os.path.exists(file_path):
                    os.unlink(file_path)
                    logger.debug(f"Removed temp file: {file_path}")
            except Exception as e:
                logger.warning(f"Failed to remove temp file {file_path}: {e}")

    def verify_filter_length(self, segments: List[EditSegment]) -> Tuple[bool, int]:
        """Verify filter complexity meets requirements"""
        if len(segments) < 2:
            return True, 0

        filter_str = self._build_xfade_filter(segments)
        length = len(filter_str)

        # Requirement: < 300 chars for 50 segments
        is_valid = length < 300

        return is_valid, length


# Helper functions for SmartVideoEditor integration


def create_edit_segments(
    highlights: List[Dict[str, Any]], source_file: str
) -> List[EditSegment]:
    """Convert highlight data to edit segments"""
    segments = []

    for highlight in highlights:
        segment = EditSegment(
            source_file=source_file,
            start_time=highlight["start_time"],
            end_time=highlight["end_time"],
            transition_type=highlight.get("transition", "fade"),
            transition_duration=highlight.get("transition_duration", 0.5),
        )
        segments.append(segment)

    return segments


# Example usage
def example_concat_edit():
    """Example of using concat editor"""

    editor = ConcatEditor()

    # Create segments
    segments = [
        EditSegment("input.mp4", 10, 40, transition_duration=0.5),
        EditSegment("input.mp4", 60, 90, transition_duration=0.5),
        EditSegment("input.mp4", 120, 150, transition_duration=0.5),
    ]

    # Verify filter length
    is_valid, length = editor.verify_filter_length(segments)
    print(f"Filter valid: {is_valid}, Length: {length} chars")

    # Execute edit
    result = editor.execute_edit(segments, "output.mp4", apply_transitions=True)

    print(f"Processed {result['segments_processed']} segments")
    print(f"Processing ratio: {result['processing_ratio']:.2f}x")


if __name__ == "__main__":
    import time

    logging.basicConfig(level=logging.INFO)
    example_concat_edit()
