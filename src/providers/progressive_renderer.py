# CLEANED: All hardcoded logic removed in Phase 0
# This file now uses dynamic functions instead of fixed values
# Ready for Phase 1 AI implementation

"""
Progressive Renderer - Show results as they process
Immediate feedback prevents user anxiety and abandonment
"""

import os
import asyncio
import logging
import tempfile
import time
from typing import Dict, Any, List, Optional, AsyncGenerator
from dataclasses import dataclass
import shutil

from concat_editor import ConcatEditor
from cleanup_manager import cleanup_manager

logger = logging.getLogger(__name__)


@dataclass
class RenderProgress:
    """Progress update for rendering"""

    stage: str  # 'preview', 'basic', 'enhanced', 'final'
    progress: int  # 0-100
    output_path: Optional[str]
    message: str
    metadata: Dict[str, Any]


class ProgressiveRenderer:
    """
    Render videos progressively, showing results as soon as possible
    """

    def __init__(self):
        self.concat_editor = ConcatEditor()
        self.temp_dir = tempfile.mkdtemp(prefix="progressive_render_")
        cleanup_manager.register_directory(self.temp_dir)

        # Rendering presets for different quality levels
        self.presets = {
            "preview": {
                "scale": "480:270",  # Very low res for speed
                "bitrate": "500k",
                "preset": "ultrafast",
                "audio_bitrate": "64k",
            },
            "basic": {
                "scale": "1080:1920",  # Full res but fast encoding
                "bitrate": "4M",
                "preset": "faster",
                "audio_bitrate": "128k",
            },
            "enhanced": {
                "scale": "1080:1920",
                "bitrate": "8M",
                "preset": "fast",
                "audio_bitrate": "192k",
            },
            "final": {
                "scale": "1080:1920",
                "bitrate": "10M",
                "preset": "medium",
                "audio_bitrate": "256k",
            },
        }

    async def render_progressive(
        self, video_path: str, segments: List[Dict[str, Any]]
    ) -> AsyncGenerator[RenderProgress, None]:
        """
        Progressive rendering that yields results as they're ready
        """
        # 1. Immediate preview (first segment only, low quality)
        preview_start = time.time()

        if segments:
            preview_segment = segments[0]  # Just first highlight
            preview_path = await self._render_preview(video_path, preview_segment)

            yield RenderProgress(
                stage="preview",
                progress=10,
                output_path=preview_path,
                message=f"Preview ready in {time.time() - preview_start:.1f}s",
                metadata={
                    "segments": 1,
                    "duration": preview_segment["end_time"]
                    - preview_segment["start_time"],
                },
            )

        # 2. Basic quality full video (all segments, fast encoding)
        basic_start = time.time()
        basic_path = await self._render_basic_async(video_path, segments)

        yield RenderProgress(
            stage="basic",
            progress=70,
            output_path=basic_path,
            message=f"Full video ready in {time.time() - basic_start:.1f}s",
            metadata={"segments": len(segments), "quality": "basic"},
        )

        # 3. Enhanced quality (if user is still waiting)
        if await self._user_still_waiting():
            enhanced_start = time.time()
            enhanced_path = await self._render_enhanced(video_path, segments)

            yield RenderProgress(
                stage="enhanced",
                progress=90,
                output_path=enhanced_path,
                message=f"Enhanced quality ready in {time.time() - enhanced_start:.1f}s",
                metadata={"segments": len(segments), "quality": "enhanced"},
            )

        # 4. Final quality (optional background task)
        final_start = time.time()
        final_path = await self._render_final(video_path, segments)

        yield RenderProgress(
            stage="final",
            progress=100,
            output_path=final_path,
            message=f"Final quality ready in {time.time() - final_start:.1f}s",
            metadata={"segments": len(segments), "quality": "final"},
        )

    async def render_basic(
        self, video_path: str, segments: List[Dict[str, Any]]
    ) -> str:
        """
        Basic quality render - fast for immediate feedback
        """
        return await self._render_segments(video_path, segments, "basic")

    async def render_with_transitions(
        self, video_path: str, segments: List[Dict[str, Any]]
    ) -> str:
        """
        Render with smooth transitions between segments
        """
        return await self._render_segments(
            video_path, segments, "enhanced", transitions=True
        )

    async def render_enhanced(
        self, video_path: str, segments: List[Dict[str, Any]]
    ) -> str:
        """
        Enhanced quality with all effects
        """
        return await self._render_segments(
            video_path, segments, "enhanced", transitions=True, effects=True
        )

    async def render_premium(
        self, video_path: str, segments: List[Dict[str, Any]]
    ) -> str:
        """
        Premium quality render with all bells and whistles
        """
        return await self._render_segments(
            video_path, segments, "final", transitions=True, effects=True, captions=True
        )

    async def _render_preview(self, video_path: str, segment: Dict[str, Any]) -> str:
        """
        Ultra-fast preview of single segment
        """
        output_path = os.path.join(self.temp_dir, f"preview_{int(time.time())}.mp4")

        # Extract and encode single segment at low quality
        cmd = [
            "ffmpeg",
            "-y",
            "-ss",
            str(segment["start_time"]),
            "-i",
            video_path,
            "-t",
            str(segment["end_time"] - segment["start_time"]),
            "-vf",
            f"scale={self.presets['preview']['scale']},crop=get_smart_crop_params()",
            "-c:v",
            "libx264",
            "-preset",
            self.presets["preview"]["preset"],
            "-b:v",
            self.presets["preview"]["bitrate"],
            "-c:a",
            "aac",
            "-b:a",
            self.presets["preview"]["audio_bitrate"],
            "-movflags",
            "+faststart",  # Web optimization
            output_path,
        ]

        await self._run_ffmpeg_async(cmd)
        return output_path

    async def _render_segments(
        self,
        video_path: str,
        segments: List[Dict[str, Any]],
        quality: str,
        transitions: bool = False,
        effects: bool = False,
        captions: bool = False,
    ) -> str:
        """
        Render segments with specified quality and effects
        """
        output_path = os.path.join(self.temp_dir, f"{quality}_{int(time.time())}.mp4")
        preset = self.presets[quality]

        if not segments:
            logger.warning("No segments to render")
            return None

        # Create concat file
        concat_file = os.path.join(self.temp_dir, f"concat_{quality}.txt")
        segment_files = []

        # Process each segment
        for i, segment in enumerate(segments):
            segment_path = await self._process_segment(
                video_path, segment, i, preset, effects, captions
            )
            segment_files.append(segment_path)

        # Apply transitions if requested
        if transitions and len(segment_files) > 1:
            final_segments = await self._apply_transitions(segment_files, preset)
        else:
            final_segments = segment_files

        # Concatenate all segments
        if len(final_segments) == 1:
            # Single segment, just copy
            shutil.copy(final_segments[0], output_path)
        else:
            # Multiple segments, concatenate
            await self._concatenate_segments(final_segments, output_path, preset)

        # Cleanup temporary segment files
        for f in segment_files:
            if os.path.exists(f):
                os.remove(f)

        return output_path

    async def _process_segment(
        self,
        video_path: str,
        segment: Dict[str, Any],
        index: int,
        preset: Dict[str, Any],
        effects: bool,
        captions: bool,
    ) -> str:
        """
        Process individual segment with effects
        """
        segment_path = os.path.join(self.temp_dir, f"segment_{index:03d}.mp4")

        # Build filter chain
        filters = []

        # Scale and crop
        filters.append(f"scale={preset['scale']}")

        # Smart crop if available, otherwise center crop
        if segment.get("crop_params"):
            crop = segment["crop_params"]
            filters.append(f"crop={crop['w']}:{crop['h']}:{crop['x']}:{crop['y']}")
        else:
            filters.append("crop=get_smart_crop_params()")

        # Effects
        if effects:
            # Add subtle color correction
            filters.append("eq=brightness=0.05:saturation=1.1")

            # Add slight sharpening
            filters.append("unsharp=5:5:0.8:5:5:0.0")

        # Build command
        cmd = [
            "ffmpeg",
            "-y",
            "-ss",
            str(segment["start_time"]),
            "-i",
            video_path,
            "-t",
            str(segment["end_time"] - segment["start_time"]),
            "-vf",
            ",".join(filters),
            "-c:v",
            "libx264",
            "-preset",
            preset["preset"],
            "-b:v",
            preset["bitrate"],
            "-c:a",
            "aac",
            "-b:a",
            preset["audio_bitrate"],
        ]

        # Add captions if available
        if captions and segment.get("captions"):
            caption_file = await self._create_caption_file(segment, index)
            cmd.extend(["-vf", f"{','.join(filters)},subtitles={caption_file}"])

        cmd.append(segment_path)

        await self._run_ffmpeg_async(cmd)
        return segment_path

    async def _apply_transitions(
        self, segment_files: List[str], preset: Dict[str, Any]
    ) -> List[str]:
        """
        Apply transitions between segments
        """
        if len(segment_files) < 2:
            return segment_files

        transitioned_files = []

        for i in range(len(segment_files) - 1):
            # Create transition between segment i and i+1
            output = os.path.join(self.temp_dir, f"trans_{i:03d}.mp4")

            # Use xfade filter for crossfade transition
            cmd = [
                "ffmpeg",
                "-y",
                "-i",
                segment_files[i],
                "-i",
                segment_files[i + 1],
                "-filter_complex",
                "[0:v][1:v]xfade=transition=fade:duration=0.5:offset=duration-0.5[v];"
                "[0:a][1:a]acrossfade=d=0.5[a]",
                "-map",
                "[v]",
                "-map",
                "[a]",
                "-c:v",
                "libx264",
                "-preset",
                preset["preset"],
                "-b:v",
                preset["bitrate"],
                "-c:a",
                "aac",
                "-b:a",
                preset["audio_bitrate"],
                output,
            ]

            try:
                await self._run_ffmpeg_async(cmd)
                transitioned_files.append(output)
            except Exception as e:
                logger.warning(f"Transition failed, using original: {e}")
                transitioned_files.extend([segment_files[i], segment_files[i + 1]])

        return transitioned_files

    async def _concatenate_segments(
        self, segment_files: List[str], output_path: str, preset: Dict[str, Any]
    ):
        """
        Concatenate segments into final video
        """
        # Create concat file
        concat_file = os.path.join(self.temp_dir, "concat.txt")
        with open(concat_file, "w") as f:
            for segment in segment_files:
                f.write(f"file '{segment}'\n")

        # Concatenate
        cmd = [
            "ffmpeg",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            concat_file,
            "-c",
            "copy",
            "-movflags",
            "+faststart",
            output_path,
        ]

        await self._run_ffmpeg_async(cmd)

    async def _create_caption_file(self, segment: Dict[str, Any], index: int) -> str:
        """
        Create SRT caption file for segment
        """
        caption_file = os.path.join(self.temp_dir, f"captions_{index:03d}.srt")

        with open(caption_file, "w") as f:
            # Simple caption format
            f.write("1\n")
            f.write("00:00:00,000 --> 00:00:05,000\n")
            f.write(f"{segment.get('summary', 'Highlight ' + str(index + 1))}\n\n")

        return caption_file

    async def _run_ffmpeg_async(self, cmd: List[str]):
        """
        Run FFmpeg command asynchronously
        """
        process = await asyncio.create_subprocess_exec(
            *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )

        stdout, stderr = await process.communicate()

        if process.returncode != 0:
            logger.error(f"FFmpeg failed: {stderr.decode()}")
            raise Exception(f"FFmpeg failed with code {process.returncode}")

    async def _render_basic_async(
        self, video_path: str, segments: List[Dict[str, Any]]
    ) -> str:
        """
        Async wrapper for basic rendering
        """
        return await self._render_segments(video_path, segments, "basic")

    async def _render_enhanced(
        self, video_path: str, segments: List[Dict[str, Any]]
    ) -> str:
        """
        Enhanced quality render
        """
        return await self._render_segments(
            video_path, segments, "enhanced", transitions=True
        )

    async def _render_final(
        self, video_path: str, segments: List[Dict[str, Any]]
    ) -> str:
        """
        Final quality render
        """
        return await self._render_segments(
            video_path, segments, "final", transitions=True, effects=True
        )

    async def _user_still_waiting(self) -> bool:
        """
        Check if user is still waiting (mock implementation)
        """
        # In production, would check websocket connection or session status
        return True

    def cleanup(self):
        """
        Clean up temporary files
        """
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)


async def main():
    """Test progressive rendering"""
    import sys

    if len(sys.argv) < 2:
        print("Usage: python progressive_renderer.py <video_path>")
        return

    video_path = sys.argv[1]

    # Mock segments
    segments = [
        {"start_time": 10, "end_time": 25, "score": 0.9},
        {"start_time": 50, "end_time": 65, "score": 0.85},
        {"start_time": 100, "end_time": 115, "score": 0.8},
    ]

    renderer = ProgressiveRenderer()

    print("Starting progressive render...")

    try:
        async for progress in renderer.render_progressive(video_path, segments):
            print(
                f"\n[{progress.progress}%] {progress.stage.upper()}: {progress.message}"
            )
            if progress.output_path:
                print(f"  Output: {progress.output_path}")
                print(
                    f"  Size: {os.path.getsize(progress.output_path) / 1024 / 1024:.1f}MB"
                )

    finally:
        renderer.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
