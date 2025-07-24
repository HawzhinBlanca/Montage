#!/usr/bin/env python3
"""
Visual Tracker using MMTracking for object tracking in videos
Implements multi-object tracking with ByteTrack algorithm
"""
import json
import logging
from pathlib import Path
from typing import List, Dict, Optional, Any
import numpy as np

logger = logging.getLogger(__name__)

# Try to import MMTracking
try:
    from mmtrack.apis import init_model, inference_mot
    from mmcv import Config
    MMTRACK_AVAILABLE = True
except ImportError:
    MMTRACK_AVAILABLE = False
    logger.warning("MMTracking not available - visual tracking will be disabled")


class VisualTracker:
    """Visual tracking using MMTracking"""
    
    def __init__(self, 
                 config_path: str = 'configs/mot/bytetrack/bytetrack_yolox_x_8xb4-80e_crowdhuman-mot20.py',
                 checkpoint_path: Optional[str] = None,
                 device: str = 'cuda:0'):
        """
        Initialize visual tracker with MMTracking
        
        Args:
            config_path: Path to MMTracking config file
            checkpoint_path: Path to model checkpoint (will auto-download if None)
            device: Device to run inference on ('cuda:0' or 'cpu')
        """
        if not MMTRACK_AVAILABLE:
            raise ImportError("MMTracking is not installed. Install with: pip install mmtrack")
        
        self.device = device if device.startswith('cuda') and self._check_cuda() else 'cpu'
        
        # Initialize ByteTrack model
        try:
            # Use default ByteTrack config if custom not provided
            if not Path(config_path).exists():
                logger.info("Using default ByteTrack configuration")
                config_path = self._get_default_config()
            
            # Initialize model
            self.model = init_model(config_path, checkpoint_path, device=self.device)
            logger.info(f"Visual tracker initialized on {self.device}")
            
        except Exception as e:
            logger.error(f"Failed to initialize MMTracking: {e}")
            raise
    
    def _check_cuda(self) -> bool:
        """Check if CUDA is available"""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False
    
    def _get_default_config(self) -> str:
        """Get default ByteTrack config"""
        # Create a minimal ByteTrack config
        config = {
            'model': {
                'type': 'ByteTrack',
                'detector': {
                    'type': 'YOLOX',
                    'backbone': {'type': 'CSPDarknet', 'deepen_factor': 1.33, 'widen_factor': 1.25},
                    'neck': {'type': 'YOLOXPAFPN', 'in_channels': [256, 512, 1024], 'out_channels': 256},
                    'bbox_head': {
                        'type': 'YOLOXHead',
                        'num_classes': 1,
                        'in_channels': 256,
                        'feat_channels': 256
                    }
                },
                'motion': {'type': 'KalmanFilter'},
                'tracker': {
                    'type': 'ByteTracker',
                    'track_high_thresh': 0.6,
                    'track_low_thresh': 0.1,
                    'new_track_thresh': 0.7,
                    'track_buffer': 30,
                    'match_thresh': 0.8,
                    'frame_rate': 30
                }
            }
        }
        
        # Write to temp file
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(f"model = {json.dumps(config, indent=2)}")
            return f.name
    
    def track(self, video_path: str, output_file: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Perform multi-object tracking on video
        
        Args:
            video_path: Path to input video
            output_file: Optional path to save tracking results
            
        Returns:
            List of tracking results per frame
        """
        if not MMTRACK_AVAILABLE:
            logger.warning("MMTracking not available, returning empty tracks")
            return []
        
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")
        
        logger.info(f"Starting visual tracking on: {video_path}")
        
        try:
            # Run inference
            results = inference_mot(self.model, str(video_path), output_file)
            
            # Process results
            processed_tracks = self._process_tracking_results(results)
            
            logger.info(f"Tracking complete: {len(processed_tracks)} frames processed")
            return processed_tracks
            
        except Exception as e:
            logger.error(f"Tracking failed: {e}")
            return []
    
    def _process_tracking_results(self, results: Any) -> List[Dict[str, Any]]:
        """Process raw MMTracking results into our format"""
        processed = []
        
        for frame_idx, frame_result in enumerate(results):
            frame_data = {
                'frame_idx': frame_idx,
                'timestamp': frame_idx / 30.0,  # Assume 30fps, adjust as needed
                'tracks': []
            }
            
            # Extract tracks from frame
            if hasattr(frame_result, 'pred_track_instances'):
                tracks = frame_result.pred_track_instances
                
                for i in range(len(tracks)):
                    track = {
                        'track_id': int(tracks.ids[i]),
                        'bbox': tracks.bboxes[i].tolist(),  # [x1, y1, x2, y2]
                        'score': float(tracks.scores[i]) if hasattr(tracks, 'scores') else 1.0,
                        'category': 'person'  # ByteTrack typically tracks people
                    }
                    
                    # Calculate additional metrics
                    bbox = track['bbox']
                    track['center'] = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
                    track['size'] = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                    
                    frame_data['tracks'].append(track)
            
            processed.append(frame_data)
        
        return processed
    
    def get_track_statistics(self, tracks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate statistics from tracking results"""
        if not tracks:
            return {}
        
        # Collect all unique track IDs
        all_track_ids = set()
        total_detections = 0
        
        for frame in tracks:
            for track in frame.get('tracks', []):
                all_track_ids.add(track['track_id'])
                total_detections += 1
        
        # Calculate track lengths
        track_lengths = {}
        for track_id in all_track_ids:
            track_lengths[track_id] = sum(
                1 for frame in tracks 
                for track in frame.get('tracks', []) 
                if track['track_id'] == track_id
            )
        
        return {
            'total_frames': len(tracks),
            'unique_tracks': len(all_track_ids),
            'total_detections': total_detections,
            'avg_tracks_per_frame': total_detections / len(tracks) if tracks else 0,
            'track_lengths': track_lengths,
            'longest_track': max(track_lengths.values()) if track_lengths else 0,
            'average_track_length': sum(track_lengths.values()) / len(track_lengths) if track_lengths else 0
        }
    
    def filter_stable_tracks(self, tracks: List[Dict[str, Any]], min_length: int = 10) -> List[Dict[str, Any]]:
        """Filter out short/unstable tracks"""
        stats = self.get_track_statistics(tracks)
        stable_ids = {
            track_id for track_id, length in stats['track_lengths'].items() 
            if length >= min_length
        }
        
        # Filter tracks
        filtered = []
        for frame in tracks:
            filtered_frame = {
                'frame_idx': frame['frame_idx'],
                'timestamp': frame['timestamp'],
                'tracks': [
                    track for track in frame.get('tracks', [])
                    if track['track_id'] in stable_ids
                ]
            }
            filtered.append(filtered_frame)
        
        return filtered
    
    def export_for_cropping(self, tracks: List[Dict[str, Any]], video_width: int, video_height: int) -> List[Dict[str, Any]]:
        """Export tracking data for intelligent cropping"""
        crop_data = []
        
        for frame in tracks:
            if not frame.get('tracks'):
                continue
            
            # Find primary subject (largest bbox)
            primary_track = max(frame['tracks'], key=lambda t: t['size'])
            
            # Calculate crop center based on primary subject
            center_x, center_y = primary_track['center']
            
            # Ensure all tracks fit in crop (if multiple)
            if len(frame['tracks']) > 1:
                all_centers = [t['center'] for t in frame['tracks']]
                center_x = sum(c[0] for c in all_centers) / len(all_centers)
                center_y = sum(c[1] for c in all_centers) / len(all_centers)
            
            crop_data.append({
                'frame_idx': frame['frame_idx'],
                'timestamp': frame['timestamp'],
                'crop_center': (int(center_x), int(center_y)),
                'primary_subject': primary_track['track_id'],
                'num_subjects': len(frame['tracks']),
                'subjects': frame['tracks']
            })
        
        return crop_data


def create_visual_tracker(device: str = 'cuda:0') -> Optional[VisualTracker]:
    """
    Factory function to create visual tracker
    
    Returns:
        VisualTracker instance or None if not available
    """
    if not MMTRACK_AVAILABLE:
        logger.warning("MMTracking not available")
        return None
    
    try:
        return VisualTracker(device=device)
    except Exception as e:
        logger.error(f"Failed to create visual tracker: {e}")
        return None