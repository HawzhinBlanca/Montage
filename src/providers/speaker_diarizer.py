"""
Speaker Diarizer - Selective diarization for multi-speaker segments only
Cost-effective approach: only diarize when multiple speakers detected
"""

import os
import logging
import numpy as np
import subprocess
import tempfile
import json
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass
import asyncio
import time

from audio_normalizer import AudioNormalizer
from cleanup_manager import cleanup_manager
from budget_guard import budget_guard

logger = logging.getLogger(__name__)


@dataclass
class SpeakerSegment:
    """A segment with speaker label"""

    speaker_id: str
    start_time: float
    end_time: float
    confidence: float


@dataclass
class DiarizationResult:
    """Result of speaker diarization"""

    segments: List[SpeakerSegment]
    speaker_count: int
    cost: float
    processing_time: float
    method: str  # 'api', 'local', 'energy_based'


class SpeakerDiarizer:
    """
    Intelligent speaker diarization that only processes multi-speaker content
    """

    def __init__(self):
        self.audio_normalizer = AudioNormalizer()
        self.temp_dir = tempfile.mkdtemp(prefix="diarizer_")
        cleanup_manager.register_directory(self.temp_dir)

        # Cost per minute for different methods
        self.costs = {
            "pyannote_api": 0.05,  # Cloud API
            "pyannote_local": 0.0,  # Local model
            "energy_based": 0.0,  # Simple energy analysis
        }

        # Thresholds for speaker detection
        self.energy_change_threshold = 0.3
        self.silence_threshold = 0.1

    async def diarize_if_needed(
        self,
        video_path: str,
        probe_result: Any = None,
        segments: Optional[List[Tuple[float, float]]] = None,
    ) -> DiarizationResult:
        """
        Main entry point - only diarizes if multiple speakers detected
        """
        logger.info("Checking if diarization needed...")

        # Quick check from probe
        if probe_result and probe_result.speaker_count <= 1:
            logger.info("Single speaker detected, skipping diarization")
            return self._single_speaker_result()

        # If no probe, do quick energy-based check
        if not probe_result:
            speaker_estimate = await self._estimate_speakers_quick(video_path)
            if speaker_estimate <= 1:
                logger.info(
                    "Energy analysis suggests single speaker, skipping diarization"
                )
                return self._single_speaker_result()

        # Multiple speakers detected - proceed with diarization
        if segments:
            # Only diarize specified segments
            return await self._diarize_segments(video_path, segments)
        else:
            # Diarize full video
            return await self._diarize_full_video(video_path)

    async def diarize_section(
        self, video_path: str, start: float, end: float
    ) -> Dict[str, Any]:
        """
        Diarize a specific section of video
        """
        logger.info(f"Diarizing section {start:.1f}-{end:.1f}s")

        # Extract audio for section
        audio_path = await self._extract_audio_section(video_path, start, end)

        try:
            # Check budget before using paid API
            if budget_guard.check_budget("current_job")[0]:
                # Try API diarization first (better quality)
                result = await self._diarize_with_api(audio_path, start, end)
            else:
                # Fallback to local methods
                result = await self._diarize_local(audio_path, start, end)

            return {
                "speakers": self._extract_speaker_list(result),
                "segments": [
                    {
                        "speaker": seg.speaker_id,
                        "start": seg.start_time,
                        "end": seg.end_time,
                        "confidence": seg.confidence,
                    }
                    for seg in result.segments
                ],
                "cost": result.cost,
            }

        finally:
            if os.path.exists(audio_path):
                os.remove(audio_path)

    async def _estimate_speakers_quick(
        self, video_path: str, sample_duration: int = 30
    ) -> int:
        """
        Quick speaker count estimation using energy patterns
        """
        # Extract short audio sample
        audio_path = os.path.join(self.temp_dir, "sample.wav")

        cmd = [
            "ffmpeg",
            "-i",
            video_path,
            "-t",
            str(sample_duration),
            "-vn",
            "-acodec",
            "pcm_s16le",
            "-ar",
            "16000",
            "-ac",
            "1",
            "-y",
            audio_path,
        ]

        try:
            subprocess.run(cmd, check=True, capture_output=True)

            # Analyze energy patterns
            import wave

            with wave.open(audio_path, "rb") as wav:
                frames = wav.readframes(wav.getnframes())
                audio_data = np.frombuffer(frames, dtype=np.int16).astype(float)

            # Detect speaker changes based on energy patterns
            speaker_changes = self._detect_speaker_changes(audio_data, 16000)

            # Estimate speaker count
            if speaker_changes < 2:
                return 1
            elif speaker_changes < 10:
                return 2
            else:
                return 3  # Multiple speakers

        except Exception as e:
            logger.error(f"Quick speaker estimation failed: {e}")
            return 1  # Assume single speaker on error
        finally:
            if os.path.exists(audio_path):
                os.remove(audio_path)

    def _detect_speaker_changes(self, audio_data: np.ndarray, sample_rate: int) -> int:
        """
        Detect potential speaker changes from audio energy
        """
        # Calculate energy in windows
        window_size = int(sample_rate * 0.5)  # 0.5 second windows
        hop_size = window_size // 2

        energies = []
        for i in range(0, len(audio_data) - window_size, hop_size):
            window = audio_data[i : i + window_size]
            energy = np.sqrt(np.mean(window**2))
            energies.append(energy)

        if not energies:
            return 0

        # Normalize energies
        energies = np.array(energies)
        if np.max(energies) > 0:
            energies = energies / np.max(energies)

        # Detect significant changes
        changes = 0
        for i in range(1, len(energies)):
            if abs(energies[i] - energies[i - 1]) > self.energy_change_threshold:
                # Check if it's not just silence
                if (
                    energies[i] > self.silence_threshold
                    or energies[i - 1] > self.silence_threshold
                ):
                    changes += 1

        return changes

    async def _diarize_segments(
        self, video_path: str, segments: List[Tuple[float, float]]
    ) -> DiarizationResult:
        """
        Diarize only specified segments
        """
        all_speaker_segments = []
        total_cost = 0.0
        start_time = time.time()

        for start, end in segments:
            # Extract and process segment
            result = await self._diarize_section_internal(video_path, start, end)
            all_speaker_segments.extend(result.segments)
            total_cost += result.cost

        # Merge and renumber speakers across segments
        merged_segments = self._merge_speaker_labels(all_speaker_segments)

        # Count unique speakers
        unique_speakers = len(set(seg.speaker_id for seg in merged_segments))

        return DiarizationResult(
            segments=merged_segments,
            speaker_count=unique_speakers,
            cost=total_cost,
            processing_time=time.time() - start_time,
            method="selective",
        )

    async def _diarize_full_video(self, video_path: str) -> DiarizationResult:
        """
        Diarize entire video
        """
        # For full video, use energy-based segmentation first
        segments = await self._find_multi_speaker_sections(video_path)

        if not segments:
            # No multi-speaker sections found
            return self._single_speaker_result()

        # Diarize only the multi-speaker sections
        return await self._diarize_segments(video_path, segments)

    async def _diarize_section_internal(
        self, video_path: str, start: float, end: float
    ) -> DiarizationResult:
        """
        Internal method to diarize a section
        """
        duration = end - start

        # Choose method based on duration and budget
        if duration > 300:  # > 5 minutes
            # Use local method for long segments to save cost
            return await self._diarize_local_section(video_path, start, end)
        else:
            # Use API for short segments if budget allows
            if budget_guard.check_budget("current_job")[0]:
                return await self._diarize_api_section(video_path, start, end)
            else:
                return await self._diarize_local_section(video_path, start, end)

    async def _diarize_api_section(
        self, video_path: str, start: float, end: float
    ) -> DiarizationResult:
        """
        Use cloud API for diarization (mock implementation)
        """
        # In production, would use pyannote.audio API or similar
        await asyncio.sleep(0.5)  # Simulate API call

        duration = end - start
        cost = duration / 60 * self.costs["pyannote_api"]

        # Mock result with realistic speaker segments
        segments = []
        current_time = start
        speaker_idx = 0

        while current_time < end:
            segment_duration = np.random.uniform(5, 20)  # 5-20 second segments
            segment_end = min(current_time + segment_duration, end)

            segments.append(
                SpeakerSegment(
                    speaker_id=f"SPEAKER_{speaker_idx % 2}",  # Alternate between 2 speakers
                    start_time=current_time,
                    end_time=segment_end,
                    confidence=0.85 + np.random.uniform(-0.1, 0.1),
                )
            )

            current_time = segment_end
            speaker_idx += 1

        return DiarizationResult(
            segments=segments,
            speaker_count=2,
            cost=cost,
            processing_time=0.5,
            method="pyannote_api",
        )

    async def _diarize_local_section(
        self, video_path: str, start: float, end: float
    ) -> DiarizationResult:
        """
        Use local diarization (energy-based for demo)
        """
        # Extract audio
        audio_path = await self._extract_audio_section(video_path, start, end)

        try:
            # Simple energy-based diarization
            import wave

            with wave.open(audio_path, "rb") as wav:
                frames = wav.readframes(wav.getnframes())
                audio_data = np.frombuffer(frames, dtype=np.int16).astype(float)
                sample_rate = wav.getframerate()

            # Detect speaker segments based on energy patterns
            segments = self._energy_based_diarization(audio_data, sample_rate, start)

            return DiarizationResult(
                segments=segments,
                speaker_count=len(set(s.speaker_id for s in segments)),
                cost=0.0,
                processing_time=0.1,
                method="energy_based",
            )

        finally:
            if os.path.exists(audio_path):
                os.remove(audio_path)

    def _energy_based_diarization(
        self, audio_data: np.ndarray, sample_rate: int, offset: float
    ) -> List[SpeakerSegment]:
        """
        Simple energy-based speaker segmentation
        """
        segments = []

        # Calculate energy in 1-second windows
        window_size = sample_rate
        energies = []

        for i in range(0, len(audio_data) - window_size, window_size):
            window = audio_data[i : i + window_size]
            energy = np.sqrt(np.mean(window**2))
            energies.append((i / sample_rate, energy))

        if not energies:
            return segments

        # Normalize
        max_energy = max(e[1] for e in energies)
        if max_energy > 0:
            energies = [(t, e / max_energy) for t, e in energies]

        # Group into speaker segments based on energy patterns
        current_speaker = "SPEAKER_0"
        segment_start = offset
        last_energy = energies[0][1]

        for i, (time_pos, energy) in enumerate(energies[1:], 1):
            # Detect speaker change
            if abs(energy - last_energy) > self.energy_change_threshold:
                # End current segment
                segments.append(
                    SpeakerSegment(
                        speaker_id=current_speaker,
                        start_time=segment_start,
                        end_time=offset + time_pos,
                        confidence=0.7,
                    )
                )

                # Start new segment
                current_speaker = (
                    "SPEAKER_1" if current_speaker == "SPEAKER_0" else "SPEAKER_0"
                )
                segment_start = offset + time_pos

            last_energy = energy

        # Add final segment
        if segment_start < offset + len(audio_data) / sample_rate:
            segments.append(
                SpeakerSegment(
                    speaker_id=current_speaker,
                    start_time=segment_start,
                    end_time=offset + len(audio_data) / sample_rate,
                    confidence=0.7,
                )
            )

        return segments

    async def _find_multi_speaker_sections(
        self, video_path: str
    ) -> List[Tuple[float, float]]:
        """
        Find sections likely to have multiple speakers
        """
        # Quick analysis of entire video
        sections = []

        # For demo, return some mock sections
        # In production, would analyze audio energy variance
        duration = await self._get_video_duration(video_path)

        # Check every 60 seconds
        for start in range(0, int(duration), 60):
            end = min(start + 60, duration)

            # Mock: assume middle third has multiple speakers
            if duration / 3 < start < 2 * duration / 3:
                sections.append((start, end))

        return sections

    async def _extract_audio_section(
        self, video_path: str, start: float, end: float
    ) -> str:
        """
        Extract audio section for processing
        """
        output_path = os.path.join(self.temp_dir, f"audio_{start}_{end}.wav")

        cmd = [
            "ffmpeg",
            "-i",
            video_path,
            "-ss",
            str(start),
            "-t",
            str(end - start),
            "-vn",
            "-acodec",
            "pcm_s16le",
            "-ar",
            "16000",
            "-ac",
            "1",
            "-y",
            output_path,
        ]

        process = await asyncio.create_subprocess_exec(
            *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )

        await process.communicate()

        if process.returncode != 0:
            raise Exception("Audio extraction failed")

        return output_path

    async def _get_video_duration(self, video_path: str) -> float:
        """Get video duration"""
        cmd = [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "json",
            video_path,
        ]

        process = await asyncio.create_subprocess_exec(
            *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )

        stdout, _ = await process.communicate()

        if process.returncode == 0:
            data = json.loads(stdout.decode())
            return float(data.get("format", {}).get("duration", 0))

        return 0

    def _merge_speaker_labels(
        self, segments: List[SpeakerSegment]
    ) -> List[SpeakerSegment]:
        """
        Merge speaker labels across segments for consistency
        """
        if not segments:
            return segments

        # Sort by time
        segments.sort(key=lambda x: x.start_time)

        # Simple merging - in production would use speaker embeddings
        return segments

    def _extract_speaker_list(self, result: DiarizationResult) -> List[str]:
        """Extract unique speaker IDs"""
        return sorted(list(set(seg.speaker_id for seg in result.segments)))

    def _single_speaker_result(self) -> DiarizationResult:
        """Return result for single speaker video"""
        return DiarizationResult(
            segments=[],
            speaker_count=1,
            cost=0.0,
            processing_time=0.0,
            method="skipped",
        )

    async def _diarize_with_api(
        self, audio_path: str, start: float, end: float
    ) -> DiarizationResult:
        """Wrapper for API diarization"""
        return await self._diarize_api_section(audio_path, start, end)

    async def _diarize_local(
        self, audio_path: str, start: float, end: float
    ) -> DiarizationResult:
        """Wrapper for local diarization"""
        return await self._diarize_local_section(audio_path, start, end)


async def main():
    """Test speaker diarization"""
    import sys

    if len(sys.argv) < 2:
        print("Usage: python speaker_diarizer.py <video_path>")
        return

    video_path = sys.argv[1]
    diarizer = SpeakerDiarizer()

    print("Testing selective speaker diarization...")

    # Test 1: Full video check
    print("\n1. Checking if diarization needed...")
    result = await diarizer.diarize_if_needed(video_path)

    print(f"Speaker count: {result.speaker_count}")
    print(f"Method used: {result.method}")
    print(f"Cost: ${result.cost:.2f}")
    print(f"Processing time: {result.processing_time:.2f}s")

    if result.segments:
        print(f"\nFound {len(result.segments)} speaker segments:")
        for i, seg in enumerate(result.segments[:5]):  # Show first 5
            print(
                f"  {seg.speaker_id}: {seg.start_time:.1f}-{seg.end_time:.1f}s "
                f"(confidence: {seg.confidence:.2f})"
            )

    # Test 2: Specific section
    print("\n2. Testing specific section (30-60s)...")
    section_result = await diarizer.diarize_section(video_path, 30, 60)

    print(f"Speakers in section: {section_result['speakers']}")
    print(f"Cost: ${section_result['cost']:.2f}")


if __name__ == "__main__":
    asyncio.run(main())
