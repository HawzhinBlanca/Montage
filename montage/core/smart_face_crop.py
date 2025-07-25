#!/usr/bin/env python3
"""
Smart Face Detection and Cropping for Vertical Video
Advanced face tracking with smooth transitions
"""
import cv2
import numpy as np
import logging
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import mediapipe as mp

logger = logging.getLogger(__name__)


@dataclass
class FaceRegion:
    """Represents a detected face region"""
    x: int
    y: int
    width: int
    height: int
    confidence: float
    tracking_id: Optional[int] = None
    landmarks: Optional[Dict] = None


@dataclass
class CropRegion:
    """Defines the crop region for a frame"""
    x: int
    y: int
    width: int
    height: int
    scale: float = 1.0


class SmartFaceCropper:
    """Advanced face detection and smart cropping for 9:16 video"""
    
    def __init__(self):
        """Initialize face detection models"""
        # OpenCV Haar Cascade (fast, basic)
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        # MediaPipe Face Detection (more accurate)
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_face_mesh = mp.solutions.face_mesh
        
        # Initialize detectors
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=1,  # Long range model
            min_detection_confidence=0.5
        )
        
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=5,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Tracking state
        self.face_tracks = {}
        self.next_track_id = 0
        self.crop_history = []
        
    def process_video(self, input_path: str, output_path: str, 
                     target_aspect: float = 9/16) -> bool:
        """
        Process video with smart face cropping
        
        Args:
            input_path: Input video path
            output_path: Output video path
            target_aspect: Target aspect ratio (9:16 for vertical)
        """
        cap = cv2.VideoCapture(input_path)
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Calculate output dimensions
        if width / height > target_aspect:
            # Video is wider than target
            out_height = height
            out_width = int(height * target_aspect)
        else:
            # Video is taller than target
            out_width = width
            out_height = int(width / target_aspect)
            
        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (out_width, out_height))
        
        frame_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # Detect faces
            faces = self._detect_faces_multi(frame)
            
            # Update tracking
            self._update_tracking(faces, frame_count)
            
            # Calculate optimal crop
            crop_region = self._calculate_smart_crop(
                faces, width, height, out_width, out_height
            )
            
            # Apply smooth transition
            crop_region = self._smooth_crop_transition(crop_region)
            
            # Crop and resize frame
            cropped = self._apply_crop(frame, crop_region, out_width, out_height)
            
            # Write frame
            out.write(cropped)
            frame_count += 1
            
            if frame_count % 30 == 0:
                logger.info(f"Processed {frame_count} frames")
                
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        
        return True
        
    def _detect_faces_multi(self, frame: np.ndarray) -> List[FaceRegion]:
        """Detect faces using multiple methods for robustness"""
        faces = []
        
        # Convert to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # MediaPipe detection
        mp_results = self.face_detection.process(rgb_frame)
        if mp_results.detections:
            for detection in mp_results.detections:
                bbox = detection.location_data.relative_bounding_box
                h, w = frame.shape[:2]
                
                faces.append(FaceRegion(
                    x=int(bbox.xmin * w),
                    y=int(bbox.ymin * h),
                    width=int(bbox.width * w),
                    height=int(bbox.height * h),
                    confidence=detection.score[0]
                ))
                
        # Fallback to Haar Cascade if no faces found
        if not faces:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            haar_faces = self.face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5
            )
            
            for (x, y, w, h) in haar_faces:
                faces.append(FaceRegion(
                    x=x, y=y, width=w, height=h, confidence=0.7
                ))
                
        # Get facial landmarks for main face
        if faces:
            mesh_results = self.face_mesh.process(rgb_frame)
            if mesh_results.multi_face_landmarks:
                # Add landmarks to first face
                faces[0].landmarks = self._extract_key_landmarks(
                    mesh_results.multi_face_landmarks[0],
                    frame.shape
                )
                
        return faces
        
    def _extract_key_landmarks(self, landmarks, shape) -> Dict:
        """Extract key facial landmarks"""
        h, w = shape[:2]
        
        # Key landmark indices
        key_points = {
            'left_eye': 33,
            'right_eye': 263,
            'nose_tip': 1,
            'mouth_center': 13,
            'chin': 152
        }
        
        extracted = {}
        for name, idx in key_points.items():
            landmark = landmarks.landmark[idx]
            extracted[name] = (int(landmark.x * w), int(landmark.y * h))
            
        return extracted
        
    def _update_tracking(self, faces: List[FaceRegion], frame_num: int):
        """Update face tracking across frames"""
        # Simple IoU-based tracking
        unmatched_faces = faces.copy()
        
        for track_id, track in self.face_tracks.items():
            best_match = None
            best_iou = 0.3  # Minimum IoU threshold
            
            for face in unmatched_faces:
                iou = self._calculate_iou(track['last_bbox'], face)
                if iou > best_iou:
                    best_iou = iou
                    best_match = face
                    
            if best_match:
                # Update track
                face.tracking_id = track_id
                track['last_bbox'] = best_match
                track['last_seen'] = frame_num
                unmatched_faces.remove(best_match)
                
        # Create new tracks for unmatched faces
        for face in unmatched_faces:
            face.tracking_id = self.next_track_id
            self.face_tracks[self.next_track_id] = {
                'last_bbox': face,
                'last_seen': frame_num
            }
            self.next_track_id += 1
            
        # Remove old tracks
        to_remove = []
        for track_id, track in self.face_tracks.items():
            if frame_num - track['last_seen'] > 30:  # 1 second at 30fps
                to_remove.append(track_id)
                
        for track_id in to_remove:
            del self.face_tracks[track_id]
            
    def _calculate_iou(self, box1: FaceRegion, box2: FaceRegion) -> float:
        """Calculate Intersection over Union"""
        x1 = max(box1.x, box2.x)
        y1 = max(box1.y, box2.y)
        x2 = min(box1.x + box1.width, box2.x + box2.width)
        y2 = min(box1.y + box1.height, box2.y + box2.height)
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = box1.width * box1.height
        area2 = box2.width * box2.height
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0
        
    def _calculate_smart_crop(self, faces: List[FaceRegion], 
                            frame_w: int, frame_h: int,
                            out_w: int, out_h: int) -> CropRegion:
        """Calculate optimal crop region"""
        if not faces:
            # No faces - use center crop
            x = (frame_w - out_w) // 2
            y = (frame_h - out_h) // 2
            return CropRegion(x, y, out_w, out_h)
            
        # Calculate bounding box of all faces
        min_x = min(face.x for face in faces)
        min_y = min(face.y for face in faces)
        max_x = max(face.x + face.width for face in faces)
        max_y = max(face.y + face.height for face in faces)
        
        # Add padding around faces
        padding = 0.2  # 20% padding
        face_center_x = (min_x + max_x) // 2
        face_center_y = (min_y + max_y) // 2
        face_width = max_x - min_x
        face_height = max_y - min_y
        
        # Calculate crop to include all faces with padding
        crop_width = min(int(face_width * (1 + padding)), frame_w)
        crop_height = min(int(face_height * (1 + padding)), frame_h)
        
        # Adjust to match aspect ratio
        if crop_width / crop_height > out_w / out_h:
            # Too wide - increase height
            crop_height = int(crop_width * out_h / out_w)
        else:
            # Too tall - increase width
            crop_width = int(crop_height * out_w / out_h)
            
        # Center crop on faces
        crop_x = face_center_x - crop_width // 2
        crop_y = face_center_y - crop_height // 2
        
        # Keep within frame bounds
        crop_x = max(0, min(crop_x, frame_w - crop_width))
        crop_y = max(0, min(crop_y, frame_h - crop_height))
        
        # Apply rule of thirds for single face
        if len(faces) == 1 and faces[0].landmarks:
            # Position face at 1/3 point vertically
            face_y = faces[0].landmarks['nose_tip'][1]
            ideal_y = crop_height // 3
            offset = ideal_y - (face_y - crop_y)
            crop_y = max(0, min(crop_y + offset, frame_h - crop_height))
            
        return CropRegion(crop_x, crop_y, crop_width, crop_height)
        
    def _smooth_crop_transition(self, crop: CropRegion, 
                               smoothing: float = 0.1) -> CropRegion:
        """Smooth crop transitions between frames"""
        if not self.crop_history:
            self.crop_history.append(crop)
            return crop
            
        # Exponential moving average
        prev_crop = self.crop_history[-1]
        
        smooth_x = int(prev_crop.x * (1 - smoothing) + crop.x * smoothing)
        smooth_y = int(prev_crop.y * (1 - smoothing) + crop.y * smoothing)
        smooth_w = int(prev_crop.width * (1 - smoothing) + crop.width * smoothing)
        smooth_h = int(prev_crop.height * (1 - smoothing) + crop.height * smoothing)
        
        smoothed = CropRegion(smooth_x, smooth_y, smooth_w, smooth_h)
        
        # Keep history limited
        self.crop_history.append(smoothed)
        if len(self.crop_history) > 30:
            self.crop_history.pop(0)
            
        return smoothed
        
    def _apply_crop(self, frame: np.ndarray, crop: CropRegion,
                   out_w: int, out_h: int) -> np.ndarray:
        """Apply crop and resize to target dimensions"""
        # Extract crop region
        cropped = frame[
            crop.y:crop.y + crop.height,
            crop.x:crop.x + crop.width
        ]
        
        # Resize to output dimensions
        resized = cv2.resize(cropped, (out_w, out_h), interpolation=cv2.INTER_CUBIC)
        
        return resized


class AdaptiveCropper:
    """Adaptive cropping based on content analysis"""
    
    def __init__(self):
        self.saliency_detector = cv2.saliency.StaticSaliencyFineGrained_create()
        
    def detect_salient_regions(self, frame: np.ndarray) -> np.ndarray:
        """Detect visually important regions"""
        success, saliency_map = self.saliency_detector.computeSaliency(frame)
        
        if success:
            # Normalize and threshold
            saliency_map = (saliency_map * 255).astype(np.uint8)
            _, binary = cv2.threshold(saliency_map, 128, 255, cv2.THRESH_BINARY)
            
            # Find contours of salient regions
            contours, _ = cv2.findContours(
                binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            
            return contours
        
        return []