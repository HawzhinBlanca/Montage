"""
Video Probe - Automatic quality detection and complexity analysis
Decides optimal processing path BEFORE wasting time/money
"""

import os
import subprocess
import json
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
import logging
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import tempfile
import cv2
import wave

logger = logging.getLogger(__name__)


@dataclass
class ProbeResult:
    """Video probe analysis results"""

    duration: float
    resolution: Tuple[int, int]
    fps: float
    codec: str
    speaker_count: int
    has_music: bool
    scene_changes: int
    technical_density: float
    complexity: str  # low, medium, high
    motion_intensity: float
    face_count: int
    audio_energy_variance: float
    recommended_track: str
    estimated_cost: float
    estimated_time: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "duration": self.duration,
            "resolution": f"{self.resolution[0]}x{self.resolution[1]}",
            "fps": self.fps,
            "codec": self.codec,
            "speaker_count": self.speaker_count,
            "has_music": self.has_music,
            "scene_changes": self.scene_changes,
            "technical_density": self.technical_density,
            "complexity": self.complexity,
            "motion_intensity": self.motion_intensity,
            "face_count": self.face_count,
            "audio_energy_variance": self.audio_energy_variance,
            "recommended_track": self.recommended_track,
            "estimated_cost": self.estimated_cost,
            "estimated_time": self.estimated_time,
        }


class VideoProbe:
    """Intelligent video analysis for automatic quality detection"""

    def __init__(self):
        self.sample_duration = 10  # seconds to sample
        self.face_cascade = None
        self._init_face_detection()

    def _init_face_detection(self):
        """Initialize face detection cascade"""
        try:
            self.face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            )
        except Exception as e:
            logger.warning(f"Face detection init failed: {e}")

    def quick_probe(self, video_path: str, sample_seconds: int = 10) -> ProbeResult:
        """
        Quickly analyze video to determine optimal processing path
        Samples only first N seconds for efficiency
        """
        logger.info(f"Probing video: {video_path}")

        # Run all analyses in parallel for speed
        with ThreadPoolExecutor(max_workers=6) as executor:
            # Submit all analysis tasks
            futures = {
                executor.submit(self._get_basic_metadata, video_path): "metadata",
                executor.submit(
                    self._estimate_speakers, video_path, sample_seconds
                ): "speakers",
                executor.submit(
                    self._detect_music, video_path, sample_seconds
                ): "music",
                executor.submit(
                    self._count_scene_changes, video_path, sample_seconds
                ): "scenes",
                executor.submit(
                    self._analyze_motion, video_path, sample_seconds
                ): "motion",
                executor.submit(
                    self._detect_faces, video_path, sample_seconds
                ): "faces",
            }

            results = {}
            for future in as_completed(futures):
                task_name = futures[future]
                try:
                    results[task_name] = future.result()
                except Exception as e:
                    logger.error(f"Task {task_name} failed: {e}")
                    results[task_name] = None

        # Combine results
        metadata = results.get("metadata", {})

        probe_result = ProbeResult(
            duration=metadata.get("duration", 0),
            resolution=metadata.get("resolution", (0, 0)),
            fps=metadata.get("fps", 30),
            codec=metadata.get("codec", "unknown"),
            speaker_count=results.get("speakers", 1),
            has_music=results.get("music", False),
            scene_changes=results.get("scenes", 0),
            technical_density=self._calculate_technical_density(results),
            complexity="unknown",
            motion_intensity=results.get("motion", {}).get("intensity", 0),
            face_count=results.get("faces", 0),
            audio_energy_variance=results.get("motion", {}).get("audio_variance", 0),
            recommended_track="",
            estimated_cost=0,
            estimated_time=0,
        )

        # Classify complexity and recommend track
        self._classify_complexity(probe_result)
        self._recommend_track(probe_result)

        logger.info(
            f"Probe complete: {probe_result.complexity} complexity, {probe_result.recommended_track} track"
        )

        return probe_result

    def _get_basic_metadata(self, video_path: str) -> Dict[str, Any]:
        """Extract basic video metadata using ffprobe"""
        cmd = [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,r_frame_rate,codec_name:format=duration",
            "-of",
            "json",
            video_path,
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            data = json.loads(result.stdout)

            stream = data["streams"][0] if data.get("streams") else {}

            # Parse frame rate
            fps = 30
            if stream.get("r_frame_rate"):
                num, den = map(int, stream["r_frame_rate"].split("/"))
                fps = num / den if den > 0 else 30

            return {
                "duration": float(data.get("format", {}).get("duration", 0)),
                "resolution": (stream.get("width", 0), stream.get("height", 0)),
                "fps": fps,
                "codec": stream.get("codec_name", "unknown"),
            }
        except Exception as e:
            logger.error(f"Metadata extraction failed: {e}")
            return {}

    def _estimate_speakers(self, video_path: str, sample_seconds: int) -> int:
        """Estimate number of speakers using audio energy patterns"""
        # Extract audio sample
        audio_sample = self._extract_audio_sample(video_path, sample_seconds)
        if not audio_sample:
            return 1

        try:
            # Analyze audio energy patterns
            with wave.open(audio_sample, "rb") as wav:
                frames = wav.readframes(wav.getnframes())
                # Convert bytes to numpy array
                samples = np.frombuffer(frames, dtype=np.int16)

            # Simple speaker estimation based on energy variance
            # Split into 1-second chunks
            chunk_size = 44100  # 1 second at 44.1kHz
            chunks = [
                samples[i : i + chunk_size] for i in range(0, len(samples), chunk_size)
            ]

            # Calculate energy for each chunk
            energies = [
                np.sqrt(np.mean(chunk**2)) for chunk in chunks if len(chunk) > 1000
            ]

            if not energies:
                return 1

            # Analyze energy patterns
            energy_variance = np.var(energies)
            energy_changes = np.diff(energies)
            significant_changes = np.sum(np.abs(energy_changes) > np.std(energies))

            # Estimate speakers based on energy patterns
            if energy_variance < 100:
                return 1  # Very consistent, likely one speaker
            elif significant_changes < 3:
                return 1  # Few changes, likely one speaker
            elif significant_changes < 10:
                return 2  # Moderate changes, likely dialogue
            else:
                return 3  # Many changes, multiple speakers

        except Exception as e:
            logger.error(f"Speaker estimation failed: {e}")
            return 1
        finally:
            if os.path.exists(audio_sample):
                os.remove(audio_sample)

    def _detect_music(self, video_path: str, sample_seconds: int) -> bool:
        """Detect if video contains music using frequency analysis"""
        audio_sample = self._extract_audio_sample(video_path, sample_seconds)
        if not audio_sample:
            return False

        try:
            # Analyze frequency spectrum
            with wave.open(audio_sample, "rb") as wav:
                frames = wav.readframes(wav.getnframes())
                samples = np.frombuffer(frames, dtype=np.int16)

            # FFT to get frequency spectrum
            fft = np.fft.fft(samples[:44100])  # First second
            freqs = np.fft.fftfreq(len(fft), 1 / 44100)
            magnitude = np.abs(fft)

            # Music detection heuristics
            # 1. Check for harmonic patterns (musical notes)
            fundamental_freqs = [
                82.4,
                110,
                146.8,
                196,
                246.9,
                329.6,
                440,
                587.3,
            ]  # Musical notes
            harmonic_score = 0

            for freq in fundamental_freqs:
                idx = np.argmin(np.abs(freqs - freq))
                if magnitude[idx] > np.mean(magnitude) * 2:
                    harmonic_score += 1

            # 2. Check for consistent rhythm (beat detection)
            # Simple beat detection using energy peaks
            window_size = 2048
            hop_size = 512
            energy_curve = []

            for i in range(0, len(samples) - window_size, hop_size):
                window = samples[i : i + window_size]
                energy_curve.append(np.sum(window**2))

            # Find peaks in energy curve
            energy_curve = np.array(energy_curve)
            peaks = self._find_peaks(energy_curve)

            # Check if peaks are regularly spaced (indicates rhythm)
            if len(peaks) > 4:
                peak_intervals = np.diff(peaks)
                rhythm_score = 1 - (np.std(peak_intervals) / np.mean(peak_intervals))
            else:
                rhythm_score = 0

            # Music detected if we have both harmonic content and rhythm
            has_music = (harmonic_score >= 3) or (rhythm_score > 0.7)

            return has_music

        except Exception as e:
            logger.error(f"Music detection failed: {e}")
            return False
        finally:
            if os.path.exists(audio_sample):
                os.remove(audio_sample)

    def _count_scene_changes(self, video_path: str, sample_seconds: int) -> int:
        """Count scene changes using frame difference analysis"""
        # Extract frames at 1fps for analysis
        frames_dir = tempfile.mkdtemp()

        try:
            cmd = [
                "ffmpeg",
                "-i",
                video_path,
                "-t",
                str(sample_seconds),
                "-vf",
                "fps=1,scale=160:90",  # Low res for speed
                "-pix_fmt",
                "gray",  # Grayscale for simplicity
                f"{frames_dir}/frame_%03d.png",
            ]

            subprocess.run(cmd, capture_output=True, check=True)

            # Analyze frame differences
            frame_files = sorted(
                [f for f in os.listdir(frames_dir) if f.endswith(".png")]
            )

            if len(frame_files) < 2:
                return 0

            scene_changes = 0
            prev_frame = cv2.imread(
                os.path.join(frames_dir, frame_files[0]), cv2.IMREAD_GRAYSCALE
            )

            for frame_file in frame_files[1:]:
                curr_frame = cv2.imread(
                    os.path.join(frames_dir, frame_file), cv2.IMREAD_GRAYSCALE
                )

                # Calculate frame difference
                diff = cv2.absdiff(prev_frame, curr_frame)
                mean_diff = np.mean(diff)

                # Scene change if difference is significant
                if mean_diff > 30:  # Threshold for scene change
                    scene_changes += 1

                prev_frame = curr_frame

            return scene_changes

        except Exception as e:
            logger.error(f"Scene detection failed: {e}")
            return 0
        finally:
            # Cleanup
            for f in os.listdir(frames_dir):
                os.remove(os.path.join(frames_dir, f))
            os.rmdir(frames_dir)

    def _analyze_motion(self, video_path: str, sample_seconds: int) -> Dict[str, float]:
        """Analyze motion intensity and audio energy variance"""
        results = {"intensity": 0, "audio_variance": 0}

        # Motion analysis using optical flow
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)

        try:
            # Sample frames
            prev_gray = None
            motion_scores = []
            frame_count = 0
            max_frames = int(fps * sample_seconds)

            while frame_count < max_frames:
                ret, frame = cap.read()
                if not ret:
                    break

                # Convert to grayscale and resize for speed
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray = cv2.resize(gray, (320, 180))

                if prev_gray is not None:
                    # Calculate optical flow
                    flow = cv2.calcOpticalFlowFarneback(
                        prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
                    )

                    # Calculate motion magnitude
                    magnitude = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
                    motion_scores.append(np.mean(magnitude))

                prev_gray = gray
                frame_count += 1

            if motion_scores:
                results["intensity"] = np.mean(motion_scores)

        except Exception as e:
            logger.error(f"Motion analysis failed: {e}")
        finally:
            cap.release()

        # Audio energy variance
        audio_sample = self._extract_audio_sample(video_path, sample_seconds)
        if audio_sample:
            try:
                with wave.open(audio_sample, "rb") as wav:
                    frames = wav.readframes(wav.getnframes())
                    samples = np.frombuffer(frames, dtype=np.int16)

                # Calculate energy variance
                chunk_size = 4410  # 0.1 second chunks
                chunks = [
                    samples[i : i + chunk_size]
                    for i in range(0, len(samples), chunk_size)
                ]
                energies = [
                    np.sqrt(np.mean(chunk**2)) for chunk in chunks if len(chunk) > 100
                ]

                if energies:
                    results["audio_variance"] = np.var(energies)

            except Exception as e:
                logger.error(f"Audio variance analysis failed: {e}")
            finally:
                if os.path.exists(audio_sample):
                    os.remove(audio_sample)

        return results

    def _detect_faces(self, video_path: str, sample_seconds: int) -> int:
        """Count unique faces in video sample"""
        if self.face_cascade is None:
            return 0

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)

        try:
            # Sample every second
            face_counts = []
            frame_count = 0
            sample_interval = int(fps)  # Sample once per second
            max_frames = int(fps * sample_seconds)

            while frame_count < max_frames:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_count % sample_interval == 0:
                    # Convert to grayscale
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                    # Detect faces
                    faces = self.face_cascade.detectMultiScale(
                        gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
                    )

                    face_counts.append(len(faces))

                frame_count += 1

            # Return maximum face count seen
            return max(face_counts) if face_counts else 0

        except Exception as e:
            logger.error(f"Face detection failed: {e}")
            return 0
        finally:
            cap.release()

    def _extract_audio_sample(self, video_path: str, duration: int) -> Optional[str]:
        """Extract audio sample for analysis"""
        output = tempfile.mktemp(suffix=".wav")

        cmd = [
            "ffmpeg",
            "-i",
            video_path,
            "-t",
            str(duration),
            "-vn",
            "-acodec",
            "pcm_s16le",
            "-ar",
            "44100",
            "-ac",
            "1",
            output,
        ]

        try:
            subprocess.run(cmd, capture_output=True, check=True)
            return output
        except Exception as e:
            logger.error(f"Audio extraction failed: {e}")
            if os.path.exists(output):
                os.remove(output)
            return None

    def _find_peaks(
        self, signal: np.ndarray, threshold_factor: float = 1.5
    ) -> List[int]:
        """Find peaks in signal for rhythm detection"""
        threshold = np.mean(signal) * threshold_factor
        peaks = []

        for i in range(1, len(signal) - 1):
            if (
                signal[i] > threshold
                and signal[i] > signal[i - 1]
                and signal[i] > signal[i + 1]
            ):
                peaks.append(i)

        return peaks

    def _calculate_technical_density(self, results: Dict[str, Any]) -> float:
        """Calculate technical content density score"""
        # Simple heuristic based on various factors
        score = 0.0

        # Many scene changes suggest technical content
        scenes = results.get("scenes", 0)
        if scenes > 5:
            score += 0.3

        # Low motion often means presentation/lecture
        motion = results.get("motion", {}).get("intensity", 0)
        if motion < 2.0:
            score += 0.2

        # Multiple speakers suggest discussion/interview
        speakers = results.get("speakers", 1)
        if speakers > 2:
            score += 0.3

        # No music usually means technical content
        if not results.get("music", False):
            score += 0.2

        return min(score, 1.0)

    def _classify_complexity(self, probe: ProbeResult):
        """Classify video complexity based on probe results"""
        complexity_score = 0

        # Duration factor
        if probe.duration > 1800:  # >30 min
            complexity_score += 2
        elif probe.duration > 600:  # >10 min
            complexity_score += 1

        # Speaker complexity
        if probe.speaker_count > 2:
            complexity_score += 2
        elif probe.speaker_count == 2:
            complexity_score += 1

        # Content complexity
        if probe.has_music:
            complexity_score += 1

        if probe.scene_changes > 10:
            complexity_score += 2
        elif probe.scene_changes > 5:
            complexity_score += 1

        if probe.technical_density > 0.7:
            complexity_score += 2
        elif probe.technical_density > 0.4:
            complexity_score += 1

        # Motion complexity
        if probe.motion_intensity > 5:
            complexity_score += 1

        # Classify
        if complexity_score >= 6:
            probe.complexity = "high"
        elif complexity_score >= 3:
            probe.complexity = "medium"
        else:
            probe.complexity = "low"

    def _recommend_track(self, probe: ProbeResult):
        """Recommend processing track based on analysis"""
        # Decision logic based on complexity and characteristics

        if probe.complexity == "low" and probe.duration < 300:
            # Simple, short video - use fast track
            probe.recommended_track = "fast"
            probe.estimated_cost = 0.0
            probe.estimated_time = probe.duration * 0.1  # 10x faster

        elif probe.complexity == "high" or probe.speaker_count > 2:
            # Complex video - use selective premium
            probe.recommended_track = "selective_premium"
            # Cost based on what features we need
            base_cost = 0.0
            if probe.speaker_count > 2:
                base_cost += 0.50  # Diarization
            if probe.technical_density > 0.7:
                base_cost += 0.30  # GPT analysis
            probe.estimated_cost = base_cost
            probe.estimated_time = probe.duration * 0.3

        else:
            # Medium complexity - smart track with enhancements
            probe.recommended_track = "smart_enhanced"
            probe.estimated_cost = 0.10  # Minimal API usage
            probe.estimated_time = probe.duration * 0.2

        logger.info(
            f"Recommended: {probe.recommended_track} "
            f"(cost: ${probe.estimated_cost:.2f}, time: {probe.estimated_time:.0f}s)"
        )


def main():
    """Test video probe functionality"""
    import sys

    if len(sys.argv) < 2:
        print("Usage: python video_probe.py <video_path>")
        return

    video_path = sys.argv[1]
    probe = VideoProbe()
    result = probe.quick_probe(video_path)

    print("\nVideo Probe Results:")
    print("-" * 50)
    for key, value in result.to_dict().items():
        print(f"{key:20}: {value}")


if __name__ == "__main__":
    main()
