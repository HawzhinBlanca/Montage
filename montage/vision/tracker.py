"""Lightweight Visual Tracker - AI Creative Director"""
import cv2
import numpy as np
from typing import List, Dict, Tuple
import logging

logger = logging.getLogger(__name__)

class VisualTracker:
    """Lightweight visual intelligence without heavy dependencies"""
    
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        self.body_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_fullbody.xml'
        )
    
    def analyze_video(self, video_path: str) -> Dict:
        """Fast video analysis for AI Creative Director"""
        cap = cv2.VideoCapture(video_path)
        
        scenes = []
        frame_count = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Sample every 30 frames for speed
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_count % 30 == 0:  # Sample every second
                timestamp = frame_count / fps
                scene_data = self.analyze_frame(frame, timestamp)
                scenes.append(scene_data)
            
            frame_count += 1
            
            # Progress tracking
            if frame_count % 300 == 0:
                progress = (frame_count / total_frames) * 100
                logger.info(f"Visual analysis: {progress:.1f}% complete")
        
        cap.release()
        return {"scenes": scenes, "total_duration": total_frames / fps}
    
    def analyze_frame(self, frame: np.ndarray, timestamp: float) -> Dict:
        """Analyze single frame for subjects and composition"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Face detection
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
        
        # Body detection  
        bodies = self.body_cascade.detectMultiScale(gray, 1.1, 4)
        
        # Composition analysis
        height, width = frame.shape[:2]
        brightness = np.mean(gray)
        contrast = np.std(gray)
        
        return {
            "timestamp": timestamp,
            "faces": len(faces),
            "bodies": len(bodies),
            "primary_subjects": self.get_primary_subjects(faces, bodies),
            "composition": {
                "brightness": float(brightness),
                "contrast": float(contrast),
                "aspect_ratio": width / height
            },
            "visual_interest_score": self.calculate_interest_score(faces, bodies, brightness, contrast)
        }
    
    def get_primary_subjects(self, faces, bodies) -> List[Dict]:
        """Extract primary subject information"""
        subjects = []
        
        for i, (x, y, w, h) in enumerate(faces):
            subjects.append({
                "type": "face",
                "id": f"face_{i}",
                "bbox": [int(x), int(y), int(w), int(h)],
                "confidence": 0.8,
                "size_ratio": (w * h) / (640 * 360)  # Assume standard resolution
            })
        
        return subjects
    
    def calculate_interest_score(self, faces, bodies, brightness, contrast) -> float:
        """Calculate visual interest score (0-1)"""
        # Face prominence (more faces = more interesting)
        face_score = min(len(faces) / 3, 1.0) * 0.4
        
        # Contrast score (higher contrast = more visually interesting)
        contrast_score = min(contrast / 100, 1.0) * 0.3
        
        # Brightness score (avoid too dark/bright)
        brightness_score = 1.0 - abs(brightness - 128) / 128 * 0.2
        
        # Subject presence
        subject_score = 0.1 if (len(faces) > 0 or len(bodies) > 0) else 0
        
        return face_score + contrast_score + brightness_score + subject_score 