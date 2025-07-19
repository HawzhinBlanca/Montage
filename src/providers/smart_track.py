"""
SmartTrack - Local ML processing without cloud APIs
Fast, free, and privacy-preserving video analysis
"""

import os
import cv2
import numpy as np
import logging
import subprocess
import tempfile
from typing import Dict, Any, List, Tuple
from dataclasses import dataclass
import asyncio
from concurrent.futures import ThreadPoolExecutor
import time

from audio_normalizer import AudioNormalizer
from cleanup_manager import cleanup_manager

logger = logging.getLogger(__name__)


@dataclass
class SmartSegment:
    """Segment identified by smart analysis"""

    start_time: float
    end_time: float
    score: float
    features: Dict[str, Any]
    segment_type: str  # 'motion', 'face', 'audio_peak', 'scene_change'


class SmartTrack:
    """
    Local-only video analysis using OpenCV and signal processing
    No cloud APIs, no costs, instant results
    """

    def __init__(self):
        self.audio_normalizer = AudioNormalizer()
        self.temp_dir = tempfile.mkdtemp(prefix="smart_track_")
        cleanup_manager.register_directory(self.temp_dir)

        # Initialize detectors
        self.face_cascade = None
        self.motion_detector = None
        self._init_detectors()

        # Analysis parameters
        self.min_segment_duration = 3.0  # seconds
        self.max_segment_duration = 30.0
        self.target_segments = 5  # aim for 5 highlights

    def _init_detectors(self):
        """Initialize ML detectors"""
        try:
            # Face detection
            self.face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            )

            # Motion detection using background subtraction
            self.motion_detector = cv2.createBackgroundSubtractorMOG2(
                detectShadows=False
            )

        except Exception as e:
            logger.error(f"Detector initialization failed: {e}")

    async def process(
        self, video_path: str, probe_result: Any = None
    ) -> Dict[str, Any]:
        """
        Main smart track processing
        """
        logger.info("Processing video with SmartTrack (local ML)")

        start_time = time.time()

        # Run multiple analyses in parallel
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {
                executor.submit(self._analyze_motion, video_path): "motion",
                executor.submit(self._analyze_faces, video_path): "faces",
                executor.submit(self._analyze_audio_energy, video_path): "audio",
                executor.submit(self._detect_scene_changes, video_path): "scenes",
            }

            analysis_results = {}
            for future in futures:
                try:
                    analysis_results[futures[future]] = future.result()
                except Exception as e:
                    logger.error(f"Analysis {futures[future]} failed: {e}")
                    analysis_results[futures[future]] = []

        # Combine all detected segments
        all_segments = self._combine_analyses(analysis_results)

        # Score and rank segments
        scored_segments = self._score_segments(all_segments, probe_result)

        # Select best segments
        final_segments = self._select_best_segments(scored_segments)

        # Add smart crop parameters for each segment
        final_segments = await self._add_crop_params(video_path, final_segments)

        processing_time = time.time() - start_time

        return {
            "segments": final_segments,
            "metadata": {
                "processing_time": processing_time,
                "analyses_performed": list(analysis_results.keys()),
                "total_candidates": len(all_segments),
                "selected": len(final_segments),
            },
        }

    def extract_quick_highlights(
        self, video_path: str, probe_result: Any
    ) -> List[Dict[str, Any]]:
        """
        Ultra-fast highlight extraction for FAST mode
        """
        # Simple uniform sampling based on video duration
        duration = probe_result.duration

        # Extract 3-5 segments evenly distributed
        num_segments = min(5, max(3, int(duration / 60)))  # 1 segment per minute
        segment_duration = min(20, duration / (num_segments * 2))  # Leave gaps

        segments = []
        step = duration / (num_segments + 1)

        for i in range(num_segments):
            start = step * (i + 1) - segment_duration / 2
            start = max(0, start)
            end = min(start + segment_duration, duration)

            segments.append(
                {
                    "start_time": start,
                    "end_time": end,
                    "score": 0.7,  # Default score
                    "type": "uniform_sample",
                }
            )

        return segments

    async def process_sections(
        self,
        video_path: str,
        probe_result: Any,
        exclude_ranges: List[Tuple[float, float]] = None,
    ) -> Dict[str, Any]:
        """
        Process only non-excluded sections
        """
        # Get full analysis first
        full_result = await self.process(video_path, probe_result)

        if not exclude_ranges:
            return full_result

        # Filter out segments in excluded ranges
        filtered_segments = []
        for segment in full_result["segments"]:
            excluded = False

            for start, end in exclude_ranges:
                if (segment["start_time"] >= start and segment["start_time"] < end) or (
                    segment["end_time"] > start and segment["end_time"] <= end
                ):
                    excluded = True
                    break

            if not excluded:
                filtered_segments.append(segment)

        full_result["segments"] = filtered_segments
        return full_result

    def calc_optical_flow(
        self, video_path: str, sample_interval: int = 10
    ) -> List[Tuple[float, float]]:
        """
        Calculate optical flow/motion scores for video frames
        Returns list of (timestamp, motion_score) tuples
        """
        motion_scores = []
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)

        if not self.motion_detector:
            cap.release()
            return motion_scores

        try:
            frame_count = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_count % sample_interval == 0:
                    # Resize for faster processing
                    small_frame = cv2.resize(frame, (320, 180))

                    # Apply motion detection
                    fg_mask = self.motion_detector.apply(small_frame)

                    # Calculate motion score
                    motion_score = np.sum(fg_mask > 0) / fg_mask.size
                    motion_scores.append((frame_count / fps, motion_score))

                frame_count += 1

        except Exception as e:
            logger.error(f"Optical flow calculation failed: {e}")
        finally:
            cap.release()

        return motion_scores

    def classify_motion(
        self, motion_scores: List[Tuple[float, float]], percentile: int = 75
    ) -> List[SmartSegment]:
        """
        Classify motion scores into high-motion segments
        """
        segments = []

        if not motion_scores:
            return segments

        motion_array = np.array([s[1] for s in motion_scores])
        threshold = np.percentile(motion_array, percentile)  # Top motion percentile

        in_segment = False
        segment_start = 0

        for time_pos, score in motion_scores:
            if score > threshold and not in_segment:
                segment_start = time_pos
                in_segment = True
            elif score <= threshold and in_segment:
                if time_pos - segment_start >= self.min_segment_duration:
                    segments.append(
                        SmartSegment(
                            start_time=segment_start,
                            end_time=time_pos,
                            score=0.8,
                            features={"motion_intensity": float(np.mean(motion_array))},
                            segment_type="motion",
                        )
                    )
                in_segment = False

        return segments

    def _analyze_motion(self, video_path: str) -> List[SmartSegment]:
        """
        Detect high-motion segments
        """
        logger.info("Analyzing motion...")

        # Step 1: Calculate optical flow
        motion_scores = self.calc_optical_flow(video_path)

        # Step 2: Classify into segments
        segments = self.classify_motion(motion_scores)

        return segments

    def _analyze_faces(self, video_path: str) -> List[SmartSegment]:
        """
        Detect segments with faces
        """
        logger.info("Analyzing faces...")
        segments = []

        if not self.face_cascade:
            return segments

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)

        try:
            face_detections = []
            frame_count = 0

            # Sample every 30th frame (1 per second at 30fps)
            sample_interval = int(fps)

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_count % sample_interval == 0:
                    # Convert to grayscale and resize
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    small_gray = cv2.resize(gray, (640, 360))

                    # Detect faces
                    faces = self.face_cascade.detectMultiScale(
                        small_gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
                    )

                    if len(faces) > 0:
                        # Calculate face positions and sizes
                        face_data = []
                        for x, y, w, h in faces:
                            # Scale back to original size
                            scale_x = frame.shape[1] / 640
                            scale_y = frame.shape[0] / 360
                            face_data.append(
                                {
                                    "x": x * scale_x,
                                    "y": y * scale_y,
                                    "w": w * scale_x,
                                    "h": h * scale_y,
                                    "size": w * h * scale_x * scale_y,
                                }
                            )

                        face_detections.append((frame_count / fps, face_data))

                frame_count += 1

            # Group consecutive face detections into segments
            if face_detections:
                segments = self._group_face_detections(face_detections)

        except Exception as e:
            logger.error(f"Face analysis failed: {e}")
        finally:
            cap.release()

        return segments

    def _analyze_audio_energy(self, video_path: str) -> List[SmartSegment]:
        """
        Detect audio peaks and interesting moments
        """
        logger.info("Analyzing audio energy...")
        segments = []

        # Extract audio
        audio_path = os.path.join(self.temp_dir, "audio_analysis.wav")

        try:
            # Extract audio
            cmd = [
                "ffmpeg",
                "-i",
                video_path,
                "-vn",
                "-acodec",
                "pcm_s16le",
                "-ar",
                "44100",
                "-ac",
                "1",
                "-y",
                audio_path,
            ]
            subprocess.run(cmd, check=True, capture_output=True)

            # Analyze audio energy
            import wave

            with wave.open(audio_path, "rb") as wav:
                frames = wav.readframes(wav.getnframes())
                audio_data = np.frombuffer(frames, dtype=np.int16)
                sample_rate = wav.getframerate()

            # Calculate energy in 1-second windows
            window_size = sample_rate
            energy_curve = []

            for i in range(0, len(audio_data) - window_size, window_size // 2):
                window = audio_data[i : i + window_size]
                energy = np.sqrt(np.mean(window.astype(float) ** 2))
                energy_curve.append((i / sample_rate, energy))

            # Find peaks
            if energy_curve:
                energies = np.array([e[1] for e in energy_curve])
                threshold = np.percentile(energies, 80)  # Top 20%

                # Find continuous high-energy segments
                in_segment = False
                segment_start = 0

                for time_pos, energy in energy_curve:
                    if energy > threshold and not in_segment:
                        segment_start = time_pos
                        in_segment = True
                    elif energy <= threshold and in_segment:
                        if time_pos - segment_start >= self.min_segment_duration:
                            segments.append(
                                SmartSegment(
                                    start_time=segment_start,
                                    end_time=time_pos,
                                    score=0.7,
                                    features={"audio_energy": float(energy)},
                                    segment_type="audio_peak",
                                )
                            )
                        in_segment = False

        except Exception as e:
            logger.error(f"Audio analysis failed: {e}")
        finally:
            if os.path.exists(audio_path):
                os.remove(audio_path)

        return segments

    def _detect_scene_changes(self, video_path: str) -> List[SmartSegment]:
        """
        Detect scene changes using histogram differences
        """
        logger.info("Detecting scene changes...")
        segments = []

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)

        try:
            prev_hist = None
            scene_changes = []
            frame_count = 0

            # Sample every 5 frames
            sample_interval = 5

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_count % sample_interval == 0:
                    # Calculate histogram
                    hist = cv2.calcHist(
                        [frame], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256]
                    )
                    hist = cv2.normalize(hist, hist).flatten()

                    if prev_hist is not None:
                        # Compare histograms
                        diff = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CORREL)

                        # Scene change if correlation is low
                        if diff < 0.7:
                            scene_changes.append(frame_count / fps)

                    prev_hist = hist

                frame_count += 1

            # Create segments around scene changes
            for i, change_time in enumerate(scene_changes):
                # Create segment starting slightly before scene change
                start = max(0, change_time - 2)
                end = min(cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps, change_time + 10)

                if end - start >= self.min_segment_duration:
                    segments.append(
                        SmartSegment(
                            start_time=start,
                            end_time=end,
                            score=0.75,
                            features={"scene_change_at": change_time},
                            segment_type="scene_change",
                        )
                    )

        except Exception as e:
            logger.error(f"Scene detection failed: {e}")
        finally:
            cap.release()

        return segments

    def _group_face_detections(
        self, detections: List[Tuple[float, List[Dict]]]
    ) -> List[SmartSegment]:
        """
        Group face detections into continuous segments
        """
        segments = []

        if not detections:
            return segments

        # Group consecutive detections
        current_segment_start = detections[0][0]
        last_time = detections[0][0]
        face_positions = []

        for time_pos, faces in detections:
            # If gap is more than 2 seconds, end segment
            if time_pos - last_time > 2.0:
                if last_time - current_segment_start >= self.min_segment_duration:
                    segments.append(
                        SmartSegment(
                            start_time=current_segment_start,
                            end_time=last_time,
                            score=0.85,
                            features={
                                "face_count": len(faces),
                                "face_positions": face_positions,
                            },
                            segment_type="face",
                        )
                    )
                current_segment_start = time_pos
                face_positions = []

            face_positions.extend(faces)
            last_time = time_pos

        # Add final segment
        if last_time - current_segment_start >= self.min_segment_duration:
            segments.append(
                SmartSegment(
                    start_time=current_segment_start,
                    end_time=last_time,
                    score=0.85,
                    features={"face_count": len(face_positions)},
                    segment_type="face",
                )
            )

        return segments

    def _combine_analyses(
        self, results: Dict[str, List[SmartSegment]]
    ) -> List[SmartSegment]:
        """
        Combine segments from different analyses
        """
        all_segments = []

        for analysis_type, segments in results.items():
            all_segments.extend(segments)

        # Sort by start time
        all_segments.sort(key=lambda x: x.start_time)

        # Merge overlapping segments
        merged = []
        for segment in all_segments:
            if not merged:
                merged.append(segment)
            else:
                last = merged[-1]
                # If segments overlap, merge them
                if segment.start_time < last.end_time:
                    # Extend the segment
                    last.end_time = max(last.end_time, segment.end_time)
                    # Combine scores (weighted by type)
                    last.score = max(last.score, segment.score)
                    # Merge features
                    last.features.update(segment.features)
                    # Update type
                    if last.segment_type != segment.segment_type:
                        last.segment_type = "mixed"
                else:
                    merged.append(segment)

        return merged

    def _score_segments(
        self, segments: List[SmartSegment], probe_result: Any = None
    ) -> List[Dict[str, Any]]:
        """
        Score segments based on multiple factors
        """
        scored = []

        for segment in segments:
            # Base score from segment type
            score = segment.score

            # Boost for faces
            if segment.segment_type == "face":
                score *= 1.2

            # Boost for optimal duration
            duration = segment.end_time - segment.start_time
            if 10 <= duration <= 20:
                score *= 1.1

            # Penalty for very short or very long
            if duration < 5:
                score *= 0.8
            elif duration > 30:
                score *= 0.9

            # Convert to dict format
            scored.append(
                {
                    "start_time": segment.start_time,
                    "end_time": segment.end_time,
                    "score": min(score, 1.0),
                    "type": segment.segment_type,
                    "features": segment.features,
                }
            )

        return scored

    def _select_best_segments(
        self, segments: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Select best non-overlapping segments
        """
        # Sort by score
        segments.sort(key=lambda x: x["score"], reverse=True)

        selected = []
        total_duration = 0
        target_duration = 60  # Default target

        for segment in segments:
            # Check if overlaps with already selected
            overlaps = False
            for selected_seg in selected:
                if (
                    segment["start_time"] < selected_seg["end_time"]
                    and segment["end_time"] > selected_seg["start_time"]
                ):
                    overlaps = True
                    break

            if not overlaps:
                duration = segment["end_time"] - segment["start_time"]
                if (
                    total_duration + duration <= target_duration * 1.5
                ):  # Allow some overrun
                    selected.append(segment)
                    total_duration += duration

                    if len(selected) >= self.target_segments:
                        break

        # Sort by time for final output
        selected.sort(key=lambda x: x["start_time"])

        return selected

    async def _add_crop_params(
        self, video_path: str, segments: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Add smart crop parameters for segments with faces
        """
        for segment in segments:
            if segment.get("type") == "face" and segment.get("features", {}).get(
                "face_positions"
            ):
                # Use face positions to determine crop
                faces = segment["features"]["face_positions"]
                if faces:
                    # Calculate average face position
                    avg_x = np.mean([f["x"] for f in faces])

                    # Calculate crop to center on faces
                    # Assuming 1920x1080 -> 607x1080 crop
                    crop_x = max(0, min(1920 - 607, avg_x - 303))

                    segment["crop_params"] = {
                        "w": 607,
                        "h": 1080,
                        "x": int(crop_x),
                        "y": 0,
                    }

        return segments


async def main():
    """Test smart track processing"""
    import sys

    if len(sys.argv) < 2:
        print("Usage: python smart_track.py <video_path>")
        return

    video_path = sys.argv[1]

    smart_track = SmartTrack()

    print("Processing with SmartTrack...")
    result = await smart_track.process(video_path)

    print(f"\nFound {len(result['segments'])} segments:")
    for i, segment in enumerate(result["segments"]):
        print(f"\nSegment {i+1}:")
        print(f"  Time: {segment['start_time']:.1f}-{segment['end_time']:.1f}s")
        print(f"  Duration: {segment['end_time'] - segment['start_time']:.1f}s")
        print(f"  Score: {segment['score']:.2f}")
        print(f"  Type: {segment['type']}")
        if segment.get("crop_params"):
            print(f"  Smart crop: x={segment['crop_params']['x']}")


if __name__ == "__main__":
    asyncio.run(main())
