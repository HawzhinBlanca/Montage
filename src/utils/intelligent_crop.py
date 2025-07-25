#!/usr/bin/env python3
"""
Intelligent video cropping for vertical format using face detection and content analysis
"""
import cv2
import numpy as np
import tempfile
import subprocess
import os
from typing import Tuple, List, Optional, Dict
import logging

logger = logging.getLogger(__name__)


class IntelligentCropper:
    """Intelligent video cropping for vertical format conversion"""

    def __init__(self):
        # Load OpenCV face detection models
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        self.profile_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_profileface.xml"
        )

        # Fallback to DNN face detection if available
        self.dnn_face_net = None
        try:
            # Try to load more accurate DNN model
            model_path = "/tmp/opencv_face_detector_uint8.pb"
            config_path = "/tmp/opencv_face_detector.pbtxt"

            if os.path.exists(model_path) and os.path.exists(config_path):
                self.dnn_face_net = cv2.dnn.readNetFromTensorflow(
                    model_path, config_path
                )
        except (cv2.error, FileNotFoundError, IOError) as e:
            # DNN model loading is optional, so we silently continue
            logger.debug(f"DNN face model not loaded: {e}")

    def analyze_video_content(
        self, video_path: str, start_ms: int, end_ms: int
    ) -> Dict:
        """Analyze video segment for optimal cropping"""

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Cannot open video: {video_path}")
            return {
                "crop_center": (0.5, 0.5),
                "confidence": 0.0,
                "face_count": 0,
                "motion_detected": False,
                "optimal_crop_center": (0.5, 0.5),
                "crop_dimensions": "unknown",
                "motion_level": 0.0,
                "energy_score": 0.0,
                "error": "Cannot open video",
            }

        try:
            # Set video position
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps <= 0:
                fps = 30  # Default fallback

            start_frame = int((start_ms / 1000) * fps)
            end_frame = int((end_ms / 1000) * fps)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            # Validate frame ranges
            start_frame = max(0, min(start_frame, total_frames - 1))
            end_frame = max(start_frame + 1, min(end_frame, total_frames))

            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

            face_centers = []
            motion_centers = []
            frame_count = 0
            prev_frame = None
            total_motion_magnitude = 0
            energy_scores = []

            # Sample frames throughout the segment (more samples for better analysis)
            sample_interval = max(
                1, (end_frame - start_frame) // 15
            )  # Sample up to 15 frames

            for frame_idx in range(
                start_frame,
                min(end_frame, start_frame + sample_interval * 15),
                sample_interval,
            ):
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()

                if not ret:
                    continue

                frame_count += 1
                h, w = frame.shape[:2]

                # 1. Enhanced face detection
                faces = self._detect_faces_enhanced(frame)
                if faces:
                    # Weight face positions by face size (larger faces get more weight)
                    total_weight = 0
                    weighted_x = 0
                    weighted_y = 0

                    for x, y, fw, fh in faces:
                        weight = fw * fh  # Face area as weight
                        center_x = x + fw // 2
                        center_y = y + fh // 2

                        weighted_x += (center_x / w) * weight
                        weighted_y += (center_y / h) * weight
                        total_weight += weight

                    if total_weight > 0:
                        face_centers.append(
                            (weighted_x / total_weight, weighted_y / total_weight)
                        )

                # 2. Motion detection (if we have previous frame)
                if prev_frame is not None:
                    motion_result = self._detect_motion_center_enhanced(
                        prev_frame, frame
                    )
                    if motion_result:
                        motion_center, motion_magnitude = motion_result
                        motion_centers.append(motion_center)
                        total_motion_magnitude += motion_magnitude

                # 3. Calculate frame energy score
                energy_score = self._calculate_frame_energy(frame)
                energy_scores.append(energy_score)

                prev_frame = frame.copy()

            cap.release()

            # Calculate additional metrics
            avg_motion_level = total_motion_magnitude / max(
                1, frame_count - 1
            )  # -1 because first frame has no motion
            avg_energy_score = sum(energy_scores) / max(1, len(energy_scores))

            # Determine optimal crop center
            crop_center, confidence = self._calculate_optimal_crop_center(
                face_centers, motion_centers, frame_count
            )

            # Calculate crop dimensions for display
            crop_dimensions = self._get_crop_dimensions_info(
                w if frame_count > 0 else 640, h if frame_count > 0 else 360
            )

            return {
                "crop_center": crop_center,
                "confidence": confidence,
                "face_count": len(face_centers),
                "motion_detected": len(motion_centers) > 0,
                "optimal_crop_center": crop_center,
                "crop_dimensions": crop_dimensions,
                "motion_level": min(1.0, avg_motion_level),
                "energy_score": min(1.0, avg_energy_score),
                "frames_analyzed": frame_count,
                "motion_points": len(motion_centers),
            }

        except (cv2.error, ValueError, AttributeError, IndexError) as e:
            logger.error(f"Error analyzing video content: {e}")
            cap.release()
            return {
                "crop_center": (0.5, 0.5),
                "confidence": 0.1,
                "face_count": 0,
                "motion_detected": False,
                "optimal_crop_center": (0.5, 0.5),
                "crop_dimensions": "unknown",
                "motion_level": 0.0,
                "energy_score": 0.0,
                "error": str(e),
            }

    def _detect_faces(self, frame) -> List[Tuple[int, int, int, int]]:
        """Detect faces in frame using multiple methods"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Method 1: Haar cascades
        faces = []

        # Frontal faces
        frontal_faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )
        faces.extend(frontal_faces)

        # Profile faces
        profile_faces = self.profile_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )
        faces.extend(profile_faces)

        # Method 2: DNN if available
        if self.dnn_face_net is not None:
            dnn_faces = self._detect_faces_dnn(frame)
            faces.extend(dnn_faces)

        # Remove duplicates and overlapping detections
        faces = self._remove_overlapping_faces(faces)

        return faces

    def _detect_faces_enhanced(self, frame) -> List[Tuple[int, int, int, int]]:
        """Enhanced face detection with multiple scale and preprocessing"""

        faces = []

        # Method 1: Standard detection on original frame
        faces.extend(self._detect_faces(frame))

        # Method 2: Enhanced preprocessing for difficult lighting
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Histogram equalization for better contrast
            enhanced_gray = cv2.equalizeHist(gray)

            # Try detection on enhanced frame
            enhanced_faces = self.face_cascade.detectMultiScale(
                enhanced_gray,
                scaleFactor=1.05,
                minNeighbors=3,
                minSize=(20, 20),
                maxSize=(300, 300),
            )
            faces.extend(enhanced_faces)

            # Try smaller scale factors for distant faces
            small_faces = self.face_cascade.detectMultiScale(
                enhanced_gray,
                scaleFactor=1.02,
                minNeighbors=4,
                minSize=(15, 15),
                maxSize=(100, 100),
            )
            faces.extend(small_faces)

        except (cv2.error, ValueError) as e:
            logger.debug(f"Enhanced preprocessing failed: {e}")

        # Method 3: Profile faces with relaxed parameters
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            profile_faces = self.profile_cascade.detectMultiScale(
                gray, scaleFactor=1.05, minNeighbors=3, minSize=(20, 20)
            )
            faces.extend(profile_faces)
        except (cv2.error, ValueError) as e:
            logger.debug(f"Profile detection failed: {e}")

        # Remove duplicates and filter by quality
        faces = self._remove_overlapping_faces(faces)

        return faces

    def _detect_faces_dnn(self, frame) -> List[Tuple[int, int, int, int]]:
        """DNN-based face detection (more accurate)"""
        h, w = frame.shape[:2]

        # Create blob from image
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123])
        self.dnn_face_net.setInput(blob)
        detections = self.dnn_face_net.forward()

        faces = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            if confidence > 0.5:  # Confidence threshold
                x1 = int(detections[0, 0, i, 3] * w)
                y1 = int(detections[0, 0, i, 4] * h)
                x2 = int(detections[0, 0, i, 5] * w)
                y2 = int(detections[0, 0, i, 6] * h)

                faces.append((x1, y1, x2 - x1, y2 - y1))

        return faces

    def _remove_overlapping_faces(
        self, faces: List[Tuple[int, int, int, int]]
    ) -> List[Tuple[int, int, int, int]]:
        """Remove overlapping face detections"""
        if len(faces) <= 1:
            return faces

        # Calculate areas and remove smaller overlapping faces
        faces_with_area = [(face, face[2] * face[3]) for face in faces]
        faces_with_area.sort(key=lambda x: x[1], reverse=True)  # Sort by area

        filtered_faces = []
        for face, area in faces_with_area:
            x1, y1, w1, h1 = face
            overlaps = False

            for existing_face in filtered_faces:
                x2, y2, w2, h2 = existing_face

                # Check for overlap
                if x1 < x2 + w2 and x1 + w1 > x2 and y1 < y2 + h2 and y1 + h1 > y2:
                    overlaps = True
                    break

            if not overlaps:
                filtered_faces.append(face)

        return filtered_faces

    def _detect_motion_center(
        self, prev_frame, curr_frame
    ) -> Optional[Tuple[float, float]]:
        """Detect center of motion between frames using multiple methods"""

        try:
            gray1 = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
            h, w = gray1.shape

            # Method 1: Dense optical flow (most robust)
            try:
                flow = cv2.calcOpticalFlowFarneback(
                    gray1, gray2, None, 0.5, 3, 15, 3, 5, 1.2, 0
                )

                # Calculate magnitude and angle of flow vectors
                magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])

                # Find areas with significant motion (above threshold)
                motion_threshold = np.mean(magnitude) + np.std(magnitude)
                motion_mask = magnitude > motion_threshold

                if np.any(motion_mask):
                    # Find center of mass of motion
                    motion_points = np.where(motion_mask)
                    if len(motion_points[0]) > 20:  # Minimum motion pixels
                        motion_y = np.mean(motion_points[0]) / h
                        motion_x = np.mean(motion_points[1]) / w
                        return (motion_x, motion_y)

            except (cv2.error, ValueError, AttributeError) as e:
                logger.debug(f"Dense optical flow failed: {e}")

            # Method 2: Corner tracking (Lucas-Kanade with proper initialization)
            try:
                # Detect corners in previous frame
                corners = cv2.goodFeaturesToTrack(
                    gray1,
                    maxCorners=200,
                    qualityLevel=0.01,
                    minDistance=10,
                    blockSize=3,
                )

                if corners is not None and len(corners) > 10:
                    # Track corners to current frame
                    new_corners, status, error = cv2.calcOpticalFlowPyrLK(
                        gray1,
                        gray2,
                        corners,
                        None,
                        winSize=(15, 15),
                        maxLevel=2,
                        criteria=(
                            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                            10,
                            0.03,
                        ),
                    )

                    # Filter good tracking results
                    good_old = corners[status == 1]
                    good_new = new_corners[status == 1]

                    if len(good_new) > 10:
                        # Calculate motion vectors
                        motion_vectors = good_new - good_old
                        motion_magnitudes = np.linalg.norm(motion_vectors, axis=1)

                        # Filter significant motion (above threshold)
                        motion_threshold = np.mean(motion_magnitudes) + 0.5 * np.std(
                            motion_magnitudes
                        )
                        significant_motion = motion_magnitudes > motion_threshold

                        if np.any(significant_motion):
                            moving_points = good_new[significant_motion]
                            motion_x = np.mean(moving_points[:, 0]) / w
                            motion_y = np.mean(moving_points[:, 1]) / h
                            return (motion_x, motion_y)

            except (cv2.error, ValueError, np.linalg.LinAlgError) as e:
                logger.debug(f"Corner tracking failed: {e}")

            # Method 3: Background subtraction (fallback)
            try:
                # Simple frame difference
                diff = cv2.absdiff(gray1, gray2)

                # Apply threshold to get motion mask
                _, motion_mask = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)

                # Morphological operations to reduce noise
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_OPEN, kernel)
                motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_CLOSE, kernel)

                # Find contours of motion areas
                contours, _ = cv2.findContours(
                    motion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )

                if contours:
                    # Filter significant contours
                    significant_contours = [
                        c for c in contours if cv2.contourArea(c) > 500
                    ]

                    if significant_contours:
                        # Calculate center of mass of all motion areas
                        total_area = 0
                        weighted_x = 0
                        weighted_y = 0

                        for contour in significant_contours:
                            area = cv2.contourArea(contour)
                            M = cv2.moments(contour)

                            if M["m00"] != 0:
                                cx = M["m10"] / M["m00"]
                                cy = M["m01"] / M["m00"]

                                weighted_x += cx * area
                                weighted_y += cy * area
                                total_area += area

                        if total_area > 0:
                            motion_x = (weighted_x / total_area) / w
                            motion_y = (weighted_y / total_area) / h
                            return (motion_x, motion_y)

            except (cv2.error, ValueError, AttributeError) as e:
                logger.debug(f"Background subtraction failed: {e}")

        except (cv2.error, ValueError, AttributeError, TypeError) as e:
            logger.error(f"Motion detection failed completely: {e}")

        return None

    def _detect_motion_center_enhanced(
        self, prev_frame, curr_frame
    ) -> Optional[Tuple[Tuple[float, float], float]]:
        """Enhanced motion detection that returns both center and magnitude"""

        motion_center = self._detect_motion_center(prev_frame, curr_frame)
        if motion_center is None:
            return None

        # Calculate motion magnitude for energy scoring
        try:
            gray1 = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

            # Simple frame difference for magnitude
            diff = cv2.absdiff(gray1, gray2)
            motion_magnitude = np.mean(diff) / 255.0  # Normalize to 0-1

            return motion_center, motion_magnitude

        except (cv2.error, ValueError, AttributeError, TypeError) as e:
            logger.debug(f"Motion magnitude calculation failed: {e}")
            return motion_center, 0.5  # Default magnitude

    def _calculate_frame_energy(self, frame) -> float:
        """Calculate visual energy/complexity of frame"""

        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Method 1: Edge density (higher edges = more visual complexity)
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])

            # Method 2: Contrast (standard deviation of pixel intensities)
            contrast = np.std(gray) / 255.0

            # Method 3: Texture (Laplacian variance)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var() / 10000.0

            # Combine metrics
            energy_score = (
                edge_density * 0.4 + contrast * 0.4 + min(1.0, laplacian_var) * 0.2
            )

            return min(1.0, energy_score)

        except (cv2.error, ValueError, AttributeError, TypeError) as e:
            logger.debug(f"Frame energy calculation failed: {e}")
            return 0.5  # Default energy

    def _get_crop_dimensions_info(self, width: int, height: int) -> str:
        """Get human-readable crop dimensions info"""

        try:
            target_aspect = 9 / 16  # Vertical video aspect ratio
            current_aspect = width / height

            if current_aspect > target_aspect:
                # Need to crop width
                new_width = int(height * target_aspect)
                crop_percent = ((width - new_width) / width) * 100
                return f"{new_width}x{height} (crop {crop_percent:.1f}% width)"
            else:
                # Need to crop height
                new_height = int(width / target_aspect)
                crop_percent = ((height - new_height) / height) * 100
                return f"{width}x{new_height} (crop {crop_percent:.1f}% height)"

        except (ValueError, ZeroDivisionError, TypeError) as e:
            return f"Error calculating: {e}"

    def _calculate_optimal_crop_center(
        self,
        face_centers: List[Tuple[float, float]],
        motion_centers: List[Tuple[float, float]],
        frame_count: int,
    ) -> Tuple[Tuple[float, float], float]:
        """Calculate optimal crop center based on face and motion data"""

        if not face_centers and not motion_centers:
            # No faces or motion detected - use safe center crop
            return (0.5, 0.4), 0.3  # Slightly above center for talking heads

        # Priority: Faces > Motion > Center
        if face_centers:
            # Average face positions with temporal smoothing
            avg_face_x = sum(x for x, y in face_centers) / len(face_centers)
            avg_face_y = sum(y for x, y in face_centers) / len(face_centers)

            # Bias slightly upward for better framing (rule of thirds)
            optimal_y = max(0.3, min(0.6, avg_face_y - 0.1))
            optimal_x = max(0.2, min(0.8, avg_face_x))

            confidence = min(0.9, 0.5 + (len(face_centers) / frame_count))
            return (optimal_x, optimal_y), confidence

        elif motion_centers:
            # Use motion center as fallback
            avg_motion_x = sum(x for x, y in motion_centers) / len(motion_centers)
            avg_motion_y = sum(y for x, y in motion_centers) / len(motion_centers)

            optimal_y = max(0.3, min(0.7, avg_motion_y))
            optimal_x = max(0.2, min(0.8, avg_motion_x))

            confidence = 0.4
            return (optimal_x, optimal_y), confidence

        # Fallback
        return (0.5, 0.4), 0.2

    def generate_crop_filter(
        self,
        input_width: int,
        input_height: int,
        crop_center: Tuple[float, float],
        target_width: int = 1080,
        target_height: int = 1920,
    ) -> str:
        """Generate FFmpeg filter for intelligent cropping"""

        crop_x, crop_y = crop_center

        # Calculate crop dimensions maintaining aspect ratio
        input_aspect = input_width / input_height
        target_aspect = target_width / target_height

        if input_aspect > target_aspect:
            # Input is wider - crop width
            crop_h = input_height
            crop_w = int(input_height * target_aspect)

            # Center the crop horizontally based on detected content
            crop_x_pos = int((input_width - crop_w) * crop_x)
            crop_y_pos = 0

        else:
            # Input is taller - crop height
            crop_w = input_width
            crop_h = int(input_width / target_aspect)

            # Center the crop vertically based on detected content
            crop_x_pos = 0
            crop_y_pos = int((input_height - crop_h) * crop_y)

        # Ensure crop stays within bounds
        crop_x_pos = max(0, min(crop_x_pos, input_width - crop_w))
        crop_y_pos = max(0, min(crop_y_pos, input_height - crop_h))

        # Generate filter
        crop_filter = f"crop={crop_w}:{crop_h}:{crop_x_pos}:{crop_y_pos}"
        scale_filter = f"scale={target_width}:{target_height}"

        return f"{crop_filter},{scale_filter}"


def _sanitize_path(path: str) -> str:
    """SECURITY: Sanitize file path to prevent command injection"""
    import os
    import re

    # Basic validation
    if not path or not isinstance(path, str):
        raise ValueError("Invalid path: must be non-empty string")

    # Resolve to absolute path to prevent path traversal
    abs_path = os.path.abspath(path)

    # Check for suspicious characters that could be used for injection
    dangerous_chars = [";", "&", "|", "`", "$", "(", ")", "<", ">", "\\n", "\\r"]
    for char in dangerous_chars:
        if char in abs_path:
            raise ValueError(
                f"SECURITY: Dangerous character '{char}' found in path: {abs_path}"
            )

    # Ensure path contains only allowed characters
    if not re.match(r"^[a-zA-Z0-9/_.-]+$", abs_path):
        raise ValueError(f"SECURITY: Path contains invalid characters: {abs_path}")

    # Must be an existing file
    if not os.path.exists(abs_path) or not os.path.isfile(abs_path):
        raise ValueError(f"SECURITY: Path is not an existing file: {abs_path}")

    # SECURITY: Only allow files in safe directories
    allowed_bases = [
        os.path.realpath(os.getcwd()),
        os.path.realpath(os.path.expanduser("~/Videos")),
        os.path.realpath(os.path.expanduser("~/Downloads")),
        os.path.realpath("/tmp"),
        os.path.realpath("/var/tmp"),
    ]

    path_is_safe = False
    for base in allowed_bases:
        try:
            os.path.relpath(abs_path, base)
            if abs_path.startswith(base + os.sep) or abs_path == base:
                path_is_safe = True
                break
        except ValueError:
            continue

    if not path_is_safe:
        raise ValueError(f"SECURITY: File outside allowed directories: {abs_path}")

    return abs_path


def _sanitize_numeric_param(
    param: float, min_val: float = 0.0, max_val: float = 1000000.0
) -> str:
    """SECURITY: Sanitize numeric parameters"""
    try:
        num_val = float(param)
        if not (min_val <= num_val <= max_val):
            raise ValueError(
                f"SECURITY: Numeric parameter {num_val} outside allowed range [{min_val}, {max_val}]"
            )
        return str(num_val)
    except (ValueError, TypeError):
        raise ValueError(f"SECURITY: Invalid numeric parameter: {param}")


def create_intelligent_vertical_video(
    video_clips: List[Dict], source_video: str, output_path: str
) -> bool:
    """Create vertical video using intelligent cropping with security validation"""

    cropper = IntelligentCropper()

    try:
        # SECURITY: Sanitize all input paths
        safe_source_video = _sanitize_path(source_video)
        safe_output_path = os.path.abspath(output_path)

        # SECURITY: Validate output directory exists and is writable
        output_dir = os.path.dirname(safe_output_path)
        if not os.path.exists(output_dir) or not os.access(output_dir, os.W_OK):
            raise ValueError(f"SECURITY: Output directory not writable: {output_dir}")

        # Get source video dimensions with sanitized path
        probe_cmd = [
            "ffprobe",
            "-v",
            "quiet",
            "-print_format",
            "json",
            "-show_streams",
            safe_source_video,
        ]

        result = subprocess.run(probe_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            logger.error("Failed to probe source video")
            return False

        import json

        info = json.loads(result.stdout)
        video_stream = next(s for s in info["streams"] if s["codec_type"] == "video")

        input_width = int(video_stream["width"])
        input_height = int(video_stream["height"])

        # Process each clip with intelligent cropping
        processed_clips = []

        for i, clip in enumerate(video_clips):
            print(f"🎯 Analyzing clip {i+1}/{len(video_clips)} for optimal framing...")

            # Analyze this clip for optimal cropping
            analysis = cropper.analyze_video_content(
                source_video, clip["start_ms"], clip["end_ms"]
            )

            crop_center = analysis["crop_center"]
            confidence = analysis["confidence"]

            print(
                f"   Face detection: {analysis['face_count']} faces, "
                f"confidence: {confidence:.2f}"
            )

            # Generate intelligent crop filter
            crop_filter = cropper.generate_crop_filter(
                input_width, input_height, crop_center
            )

            # SECURITY: Extract clip with intelligent cropping using sanitized inputs
            # Generate secure temporary filename
            import tempfile

            temp_dir = tempfile.gettempdir()
            clip_output = os.path.join(temp_dir, f"secure_clip_{i:04d}.mp4")

            # SECURITY: Sanitize and validate all numeric parameters
            try:
                start_seconds = _sanitize_numeric_param(
                    clip["start_ms"] / 1000, 0.0, 86400.0
                )  # Max 24 hours
                duration_seconds = _sanitize_numeric_param(
                    (clip["end_ms"] - clip["start_ms"]) / 1000, 0.1, 3600.0
                )  # Max 1 hour clips
            except (KeyError, ValueError) as e:
                logger.error(f"SECURITY: Invalid clip timing parameters: {e}")
                return False

            # SECURITY: Validate crop_filter contains only safe characters
            if not isinstance(crop_filter, str) or not crop_filter:
                logger.error("SECURITY: Invalid crop_filter")
                return False

            # Check crop_filter for injection attempts
            import re

            if not re.match(r"^[a-zA-Z0-9=:,.\-]+$", crop_filter):
                logger.error(
                    f"SECURITY: crop_filter contains invalid characters: {crop_filter}"
                )
                return False

            extract_cmd = [
                "ffmpeg",
                "-y",
                "-ss",
                start_seconds,
                "-t",
                duration_seconds,
                "-i",
                safe_source_video,
                "-vf",
                f"{crop_filter},fps=30",
                "-c:v",
                "libx264",
                "-preset",
                "fast",
                "-crf",
                "18",
                "-c:a",
                "aac",
                "-b:a",
                "192k",
                clip_output,
            ]

            result = subprocess.run(extract_cmd, capture_output=True)
            if result.returncode == 0:
                processed_clips.append(clip_output)
                print(f"   ✅ Clip {i+1} processed with intelligent crop")
            else:
                logger.error(f"Failed to process clip {i+1}: {result.stderr}")
                return False

        # Concatenate all processed clips
        if processed_clips:
            concat_file = "/tmp/smart_concat.txt"
            with open(concat_file, "w") as f:
                for clip_file in processed_clips:
                    f.write(f"file '{clip_file}'\n")

            concat_cmd = [
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
                output_path,
            ]

            result = subprocess.run(concat_cmd, capture_output=True)

            # Cleanup temp files
            for clip_file in processed_clips:
                try:
                    os.unlink(clip_file)
                except (OSError, PermissionError):
                    pass  # Cleanup errors are non-critical
            try:
                os.unlink(concat_file)
            except (OSError, PermissionError):
                pass  # Cleanup errors are non-critical

            if result.returncode == 0:
                print("✅ Intelligent vertical video created successfully")
                return True
            else:
                logger.error(f"Failed to concatenate clips: {result.stderr}")
                return False

        return False

    except (
        subprocess.CalledProcessError,
        FileNotFoundError,
        OSError,
        ValueError,
        cv2.error,
    ) as e:
        logger.error(f"Intelligent cropping failed: {e}")
        return False
    except Exception as e:
        # Catch any truly unexpected errors at top level
        logger.critical(
            f"Unexpected error in intelligent cropping: {type(e).__name__}: {e}"
        )
        return False
