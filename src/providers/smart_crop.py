"""Spring-damped smart cropping with face tracking for vertical video exports"""

import cv2
import numpy as np
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from collections import deque

from ..core.metrics import track_processing_stage

logger = logging.getLogger(__name__)


@dataclass
class Face:
    """Represents a detected face"""

    x: int
    y: int
    width: int
    height: int
    confidence: float = 1.0

    @property
    def center_x(self):
        return self.x + self.width // 2

    @property
    def center_y(self):
        return self.y + self.height // 2

    @property
    def area(self):
        return self.width * self.height


@dataclass
class CropBox:
    """Represents a crop region with smooth animation support"""

    x: int
    y: int
    width: int
    height: int
    velocity_x: float = 0.0
    velocity_y: float = 0.0

    @property
    def center_x(self):
        return self.x + self.width // 2

    @property
    def center_y(self):
        return self.y + self.height // 2

    def contains_point(self, x: int, y: int) -> bool:
        """Check if point is within crop box"""
        return (
            self.x <= x <= self.x + self.width and self.y <= y <= self.y + self.height
        )


class SpringDamper:
    """Spring-damper system for smooth animation"""

    def __init__(self, stiffness: float = 20.0, damping: float = 10.0):
        self.stiffness = stiffness  # Spring constant (k)
        self.damping = damping  # Damping coefficient (c)

    def update(
        self, current: float, target: float, velocity: float, dt: float
    ) -> Tuple[float, float]:
        """
        Update position using spring-damper physics.

        Returns:
            (new_position, new_velocity)
        """
        # Spring force: F = -k * (x - target)
        spring_force = -self.stiffness * (current - target)

        # Damping force: F = -c * v
        damping_force = -self.damping * velocity

        # Total force
        force = spring_force + damping_force

        # Update velocity: v = v + F * dt
        new_velocity = velocity + force * dt

        # Update position: x = x + v * dt
        new_position = current + new_velocity * dt

        return new_position, new_velocity


class SmartCropper:
    """Intelligent video cropping with face tracking"""

    def __init__(self, output_aspect_ratio: float = 9.0 / 16.0):
        """
        Initialize smart cropper.

        Args:
            output_aspect_ratio: Target aspect ratio (9:16 for vertical)
        """
        self.output_aspect_ratio = output_aspect_ratio
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        self.spring_damper = SpringDamper(stiffness=15.0, damping=8.0)
        self.face_history = deque(maxlen=30)  # 1 second at 30fps

    @track_processing_stage("smart_crop")
    def process_video(
        self, input_path: str, output_path: str, video_duration: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Process video with smart cropping.

        Returns:
            Processing statistics and crop metadata
        """
        logger.info(f"Starting smart crop for {input_path}")

        # Open video
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise ValueError(f"Failed to open video: {input_path}")

        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Calculate output dimensions
        output_width, output_height = self._calculate_output_dimensions(
            frame_width, frame_height
        )

        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps, (output_width, output_height))

        # Initialize crop box
        crop_box = self._initialize_crop_box(
            frame_width, frame_height, output_width, output_height
        )

        # Process statistics
        stats = {
            "frames_processed": 0,
            "faces_detected": 0,
            "crop_adjustments": 0,
            "multiple_faces_frames": 0,
            "crop_metadata": [],
        }

        # Process each frame
        frame_count = 0
        dt = 1.0 / fps  # Time step

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Detect faces
            faces = self._detect_faces(frame)

            # Update face history
            self.face_history.append(faces)

            # Determine target position
            target_x, target_y, confidence = self._calculate_target_position(
                faces, frame_width, frame_height, crop_box
            )

            # Update crop box with spring damping
            crop_box = self._update_crop_box(
                crop_box, target_x, target_y, dt, frame_width, frame_height
            )

            # Apply crop
            cropped = self._apply_crop(frame, crop_box)

            # Resize to output dimensions if needed
            if cropped.shape[:2] != (output_height, output_width):
                cropped = cv2.resize(cropped, (output_width, output_height))

            # Write frame
            out.write(cropped)

            # Update statistics
            stats["frames_processed"] += 1
            if faces:
                stats["faces_detected"] += 1
            if len(faces) > 1:
                stats["multiple_faces_frames"] += 1

            # Save crop metadata periodically
            if frame_count % int(fps) == 0:  # Every second
                stats["crop_metadata"].append(
                    {
                        "time": frame_count / fps,
                        "crop_x": crop_box.x,
                        "crop_y": crop_box.y,
                        "faces": len(faces),
                        "confidence": confidence,
                    }
                )

            frame_count += 1

            # Progress logging
            if frame_count % int(fps * 10) == 0:  # Every 10 seconds
                progress = frame_count / total_frames * 100
                logger.info(f"Smart crop progress: {progress:.1f}%")

        # Cleanup
        cap.release()
        out.release()

        # Calculate quality metrics
        face_coverage = (
            stats["faces_detected"] / stats["frames_processed"]
            if stats["frames_processed"] > 0
            else 0
        )
        stats["face_coverage_ratio"] = face_coverage
        stats["output_dimensions"] = f"{output_width}x{output_height}"

        logger.info(f"Smart crop complete. Face coverage: {face_coverage:.1%}")

        return stats

    def _detect_faces(self, frame: np.ndarray) -> List[Face]:
        """Detect faces in frame using OpenCV"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces_raw = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )

        # Convert to Face objects
        faces = []
        for x, y, w, h in faces_raw:
            faces.append(Face(x, y, w, h))

        return faces

    def _calculate_target_position(
        self, faces: List[Face], frame_width: int, frame_height: int, crop_box: CropBox
    ) -> Tuple[int, int, float]:
        """
        Calculate target position for crop box based on faces.

        Returns:
            (target_x, target_y, confidence)
        """
        if not faces:
            # No faces - use recent history
            recent_faces = self._get_recent_faces()
            if recent_faces:
                faces = recent_faces
            else:
                # No faces at all - return current position
                return crop_box.x, crop_box.y, 0.0

        if len(faces) == 1:
            # Single face - center on it
            face = faces[0]
            target_x = face.center_x - crop_box.width // 2
            target_y = face.center_y - crop_box.height // 2
            confidence = 1.0

        else:
            # Multiple faces - find bounding box
            min_x = min(f.x for f in faces)
            min_y = min(f.y for f in faces)
            max_x = max(f.x + f.width for f in faces)
            max_y = max(f.y + f.height for f in faces)

            # Center of bounding box
            center_x = (min_x + max_x) // 2
            center_y = (min_y + max_y) // 2

            target_x = center_x - crop_box.width // 2
            target_y = center_y - crop_box.height // 2
            confidence = 0.8

        # Clamp to frame boundaries
        target_x = max(0, min(target_x, frame_width - crop_box.width))
        target_y = max(0, min(target_y, frame_height - crop_box.height))

        return target_x, target_y, confidence

    def _update_crop_box(
        self,
        crop_box: CropBox,
        target_x: int,
        target_y: int,
        dt: float,
        frame_width: int,
        frame_height: int,
    ) -> CropBox:
        """Update crop box position with spring damping"""
        # Update X position
        new_x, new_vx = self.spring_damper.update(
            crop_box.x, target_x, crop_box.velocity_x, dt
        )

        # Update Y position
        new_y, new_vy = self.spring_damper.update(
            crop_box.y, target_y, crop_box.velocity_y, dt
        )

        # Ensure within bounds
        new_x = max(0, min(new_x, frame_width - crop_box.width))
        new_y = max(0, min(new_y, frame_height - crop_box.height))

        return CropBox(
            x=int(new_x),
            y=int(new_y),
            width=crop_box.width,
            height=crop_box.height,
            velocity_x=new_vx,
            velocity_y=new_vy,
        )

    def _apply_crop(self, frame: np.ndarray, crop_box: CropBox) -> np.ndarray:
        """Apply crop to frame"""
        return frame[
            crop_box.y : crop_box.y + crop_box.height,
            crop_box.x : crop_box.x + crop_box.width,
        ]

    def _calculate_output_dimensions(
        self, frame_width: int, frame_height: int
    ) -> Tuple[int, int]:
        """Calculate output dimensions based on aspect ratio"""
        # For vertical video (9:16), prioritize height
        if self.output_aspect_ratio < 1:
            # Portrait orientation
            output_height = frame_height
            output_width = int(output_height * self.output_aspect_ratio)

            # Ensure width doesn't exceed frame width
            if output_width > frame_width:
                output_width = frame_width
                output_height = int(output_width / self.output_aspect_ratio)
        else:
            # Landscape orientation
            output_width = frame_width
            output_height = int(output_width / self.output_aspect_ratio)

            # Ensure height doesn't exceed frame height
            if output_height > frame_height:
                output_height = frame_height
                output_width = int(output_height * self.output_aspect_ratio)

        # Ensure even dimensions for video encoding
        output_width = output_width if output_width % 2 == 0 else output_width - 1
        output_height = output_height if output_height % 2 == 0 else output_height - 1

        return output_width, output_height

    def _initialize_crop_box(
        self, frame_width: int, frame_height: int, output_width: int, output_height: int
    ) -> CropBox:
        """Initialize crop box at center of frame"""
        x = (frame_width - output_width) // 2
        y = (frame_height - output_height) // 2

        return CropBox(x=x, y=y, width=output_width, height=output_height)

    def _get_recent_faces(self, lookback_frames: int = 10) -> List[Face]:
        """Get faces from recent history"""
        for faces in reversed(list(self.face_history)[-lookback_frames:]):
            if faces:
                return faces
        return []


class LetterboxCropper:
    """Fallback cropper with blurred letterbox for multiple faces"""

    def __init__(self, output_aspect_ratio: float = 9.0 / 16.0):
        self.output_aspect_ratio = output_aspect_ratio

    def create_letterbox_frame(
        self, frame: np.ndarray, output_width: int, output_height: int
    ) -> np.ndarray:
        """Create letterboxed frame with blurred background"""
        frame_height, frame_width = frame.shape[:2]

        # Create blurred background
        background = cv2.resize(frame, (output_width, output_height))
        background = cv2.GaussianBlur(background, (21, 21), 0)

        # Calculate scaling to fit frame within output
        scale = min(output_width / frame_width, output_height / frame_height)
        scaled_width = int(frame_width * scale)
        scaled_height = int(frame_height * scale)

        # Resize frame
        scaled_frame = cv2.resize(frame, (scaled_width, scaled_height))

        # Calculate position to center scaled frame
        x_offset = (output_width - scaled_width) // 2
        y_offset = (output_height - scaled_height) // 2

        # Overlay scaled frame on background
        background[
            y_offset : y_offset + scaled_height, x_offset : x_offset + scaled_width
        ] = scaled_frame

        return background


# Integration functions


def apply_smart_crop(
    input_video: str, output_video: str, aspect_ratio: str = "9:16"
) -> Dict[str, Any]:
    """
    Apply smart cropping to video.

    Args:
        input_video: Path to input video
        output_video: Path to output video
        aspect_ratio: Target aspect ratio (e.g., "9:16", "1:1", "4:5")

    Returns:
        Crop statistics and metadata
    """
    # Parse aspect ratio
    if ":" in aspect_ratio:
        w, h = map(int, aspect_ratio.split(":"))
        ratio = float(w) / float(h)
    else:
        ratio = float(aspect_ratio)

    cropper = SmartCropper(output_aspect_ratio=ratio)
    return cropper.process_video(input_video, output_video)


def generate_crop_filter(crop_metadata: List[Dict[str, Any]], fps: float = 30) -> str:
    """
    Generate FFmpeg crop filter from metadata.

    This can be used to apply the same crop using FFmpeg instead of OpenCV.
    """
    if not crop_metadata:
        return ""

    # Build animated crop filter
    filter_parts = []

    for i, meta in enumerate(crop_metadata):
        time = meta["time"]
        x = meta["crop_x"]
        y = meta["crop_y"]

        if i == 0:
            # Initial crop
            filter_parts.append(f"crop=w=iw/2:h=ih:x={x}:y={y}")
        else:
            # Animated crop
            prev_meta = crop_metadata[i - 1]
            prev_time = prev_meta["time"]

            # Linear interpolation between keyframes
            filter_parts.append(
                f"crop=w=iw/2:h=ih:"
                f"x='lerp({prev_meta['crop_x']},{x},(t-{prev_time})/{time-prev_time})':"
                f"y='lerp({prev_meta['crop_y']},{y},(t-{prev_time})/{time-prev_time})'"
            )

    return ",".join(filter_parts)


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Test smart cropping
    cropper = SmartCropper(output_aspect_ratio=9.0 / 16.0)

    # Process video (would need actual video file)
    # stats = cropper.process_video("input.mp4", "output_vertical.mp4")
    # print(f"Processed {stats['frames_processed']} frames")
    # print(f"Face coverage: {stats['face_coverage_ratio']:.1%}")
