#!/usr/bin/env python3
"""
Core Pipeline - Main video processing pipeline with visual tracking integration
Contains the pipeline hook for VisualTracker.track() after smart_crop as specified in Tasks.md
"""
import logging
from typing import Dict, Any, List, Optional
from pathlib import Path
import json

from .visual_tracker import VisualTracker, create_visual_tracker

logger = logging.getLogger(__name__)


class Pipeline:
    """
    Core video processing pipeline with visual tracking integration
    Implements the pipeline hook as specified in Tasks.md Phase 2.1
    """
    
    def __init__(self, enable_visual_tracking: bool = True):
        """Initialize pipeline"""
        self.enable_visual_tracking = enable_visual_tracking
        self.visual_tracker = None
        
        if self.enable_visual_tracking:
            self.visual_tracker = create_visual_tracker()
            if self.visual_tracker:
                logger.info("Pipeline initialized with visual tracking enabled")
            else:
                logger.warning("Visual tracking not available, running without it")
        
    def process_video(self, video_path: str, output_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Process video through the complete pipeline
        
        Args:
            video_path: Path to input video
            output_dir: Output directory for results
            
        Returns:
            Pipeline results
        """
        video_path = Path(video_path)
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
        else:
            output_dir = video_path.parent
            
        results = {
            "success": True,
            "video_path": str(video_path),
            "output_dir": str(output_dir),
            "pipeline_steps": []
        }
        
        try:
            # Step 1: Smart crop processing (existing functionality)
            logger.info("Running smart crop analysis...")
            smart_crop_result = self._run_smart_crop(video_path)
            results["pipeline_steps"].append({
                "step": "smart_crop",
                "success": smart_crop_result.get("success", False),
                "result": smart_crop_result
            })
            
            # Step 2: Visual tracking hook (as specified in Tasks.md)
            # "Pipeline Hook: In montage/core/pipeline.py, call VisualTracker.track() after smart_crop"
            if self.visual_tracker and self.enable_visual_tracking:
                logger.info("Running visual tracking after smart_crop...")
                tracks = self.visual_tracker.track(str(video_path))
                
                # Filter stable tracks
                stable_tracks = self.visual_tracker.filter_stable_tracks(tracks, min_length=30)
                
                # Export for cropping
                if hasattr(self.visual_tracker, 'export_for_cropping'):
                    crop_data = self.visual_tracker.export_for_cropping(
                        stable_tracks,
                        1920,  # Default width
                        1080   # Default height
                    )
                else:
                    crop_data = []
                
                # Save tracking data as specified in Tasks.md
                track_file = output_dir / "tracks.json"
                track_output = {
                    'tracks': stable_tracks,
                    'crop_data': crop_data,
                    'statistics': self.visual_tracker.get_track_statistics(tracks) if hasattr(self.visual_tracker, 'get_track_statistics') else {}
                }
                
                with open(track_file, 'w') as f:
                    json.dump(track_output, f, indent=2)
                
                results["pipeline_steps"].append({
                    "step": "visual_tracking",
                    "success": True,
                    "tracks_count": len(stable_tracks),
                    "output_file": str(track_file)
                })
                
                logger.info(f"Visual tracking completed: {len(stable_tracks)} stable tracks")
            else:
                logger.info("Visual tracking disabled or not available")
                results["pipeline_steps"].append({
                    "step": "visual_tracking",
                    "success": False,
                    "reason": "disabled_or_unavailable"
                })
            
            # Step 3: Additional pipeline steps can be added here
            logger.info("Pipeline processing completed successfully")
            
        except Exception as e:
            logger.error(f"Pipeline processing failed: {e}")
            results["success"] = False
            results["error"] = str(e)
            
        return results
    
    def _run_smart_crop(self, video_path: Path) -> Dict[str, Any]:
        """
        Run smart crop analysis
        
        Args:
            video_path: Path to video file
            
        Returns:
            Smart crop results
        """
        try:
            from ..providers.smart_track import SmartTrack
            
            smart_track = SmartTrack()
            result = smart_track.analyze_video(str(video_path))
            
            return {
                "success": True,
                "result": result
            }
            
        except Exception as e:
            logger.error(f"Smart crop failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }


def run_pipeline(video_path: str, output_dir: Optional[str] = None, enable_visual_tracking: bool = True) -> Dict[str, Any]:
    """
    Convenience function to run the complete pipeline
    
    Args:
        video_path: Path to input video
        output_dir: Output directory for results
        enable_visual_tracking: Whether to enable visual tracking
        
    Returns:
        Pipeline results
    """
    pipeline = Pipeline(enable_visual_tracking=enable_visual_tracking)
    return pipeline.process_video(video_path, output_dir)