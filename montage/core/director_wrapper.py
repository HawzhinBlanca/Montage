#!/usr/bin/env python3
"""
Director Wrapper - AI Orchestration using VideoDB Director
Wraps Montage functions as Director agents for intelligent pipeline orchestration
"""
import logging
from typing import Dict, Any, List, Optional, Callable
from pathlib import Path
import json

logger = logging.getLogger(__name__)

# Try to import VideoDB Director
try:
    from videodb import Director
    DIRECTOR_AVAILABLE = True
except ImportError:
    DIRECTOR_AVAILABLE = False
    logger.warning("VideoDB Director not available - orchestration will use fallback mode")


# Import actual implementations - EXACT pattern from Tasks.md
from montage.core.whisper_transcriber import WhisperTranscriber
from montage.core.visual_tracker import VisualTracker  
from montage.core.ffmpeg_editor import FFMPEGEditor

# Create global director instance as per Tasks.md
if DIRECTOR_AVAILABLE:
    director = Director()
    # Create wrapper instances - EXACT pattern from Tasks.md  
    whisper_transcriber = WhisperTranscriber()
    visual_tracker = VisualTracker()
    ffmpeg_editor = FFMPEGEditor()
    
    director.add_agent("transcribe", whisper_transcriber.transcribe)
    director.add_agent("track",     visual_tracker.track)
    director.add_agent("edit",      ffmpeg_editor.process)
else:
    director = None


class DirectorOrchestrator:
    """Orchestrates video processing pipeline using VideoDB Director"""
    
    def __init__(self):
        """Initialize Director orchestrator"""
        if not DIRECTOR_AVAILABLE:
            logger.warning("Director not available, using fallback orchestration")
            self.director = None
        else:
            self.director = director  # Use global instance
            self._register_additional_agents()
    
    def _register_additional_agents(self):
        """Register additional Montage functions as Director agents"""
        if not self.director:
            return
        
        # Additional agents beyond the core 3 from Tasks.md
        # Smart analysis agent
        self.director.add_agent(
            "analyze_smart",
            self._create_smart_wrapper(),
            description="Analyze video for highlights using local ML"
        )
        
        # Speaker diarization agent
        self.director.add_agent(
            "diarize_speakers",
            self._create_diarizer_wrapper(),
            description="Identify different speakers in audio"
        )
        
        # Audio normalization agent
        self.director.add_agent(
            "normalize_audio",
            self._create_normalizer_wrapper(),
            description="Normalize audio levels to EBU R128 standard"
        )
        
        # Highlight selection agent
        self.director.add_agent(
            "select_highlights",
            self._create_highlight_wrapper(),
            description="Select best highlight clips from analysis"
        )
        
        logger.info("Registered additional Montage agents with Director")
    
    
    def _create_smart_wrapper(self) -> Callable:
        """Create wrapper for smart analysis with lazy import"""
        def smart_wrapper(video_path: str) -> Dict[str, Any]:
            from montage.providers.smart_track import SmartTrack
            smart_track = SmartTrack()
            return smart_track.analyze_video(video_path)
        return smart_wrapper
    
    def _create_diarizer_wrapper(self) -> Callable:
        """Create wrapper for speaker diarization with lazy import"""
        def diarizer_wrapper(audio_path: str) -> Optional[Dict]:
            from montage.core.diarization import get_diarizer
            diarizer = get_diarizer()
            if diarizer:
                return diarizer.diarize(audio_path)
            return None
        return diarizer_wrapper
    
    def _create_normalizer_wrapper(self) -> Callable:
        """Create wrapper for audio normalization with lazy import"""
        def normalizer_wrapper(input_path: str, output_path: str) -> Dict[str, Any]:
            from montage.providers.audio_normalizer import AudioNormalizer
            normalizer = AudioNormalizer()
            return normalizer.normalize_audio(input_path, output_path)
        return normalizer_wrapper
    
    
    def _create_highlight_wrapper(self) -> Callable:
        """Create wrapper for highlight selection"""
        def highlight_wrapper(analysis_results: Dict, target_duration: int = 60) -> List[Dict]:
            from montage.core.highlight_selector import analyze_highlights
            return analyze_highlights(
                analysis_results.get("segments", []),
                target_duration=target_duration
            )
        return highlight_wrapper
    
    def run_pipeline(self, video_path: str, instruction: str = None) -> Dict[str, Any]:
        """
        Run AI-orchestrated pipeline
        
        Args:
            video_path: Path to input video
            instruction: Natural language instruction for Director
            
        Returns:
            Pipeline results
        """
        if not self.director:
            return self._run_fallback_pipeline(video_path)
        
        # Default instruction if none provided
        if not instruction:
            instruction = (
                "Analyze this video comprehensively: "
                "1. Transcribe the audio with speaker identification "
                "2. Track people and objects throughout the video "
                "3. Find the most interesting 60-second highlights "
                "4. Create a highlight reel with smooth transitions"
            )
        
        try:
            # Run Director with instruction
            logger.info(f"Running Director pipeline: {instruction}")
            result = self.director.run(instruction, context={"video_path": video_path})
            
            # Process Director output
            return self._process_director_output(result)
            
        except Exception as e:
            logger.error(f"Director pipeline failed: {e}")
            return self._run_fallback_pipeline(video_path)
    
    def _process_director_output(self, director_result: Any) -> Dict[str, Any]:
        """Process and structure Director output"""
        # Director returns varied formats, normalize it
        output = {
            "success": True,
            "pipeline": "director",
            "results": {}
        }
        
        # Extract results from Director response
        if isinstance(director_result, dict):
            output["results"] = director_result
        elif isinstance(director_result, str):
            try:
                output["results"] = json.loads(director_result)
            except:
                output["results"] = {"message": director_result}
        else:
            output["results"] = {"raw_output": str(director_result)}
        
        return output
    
    def _run_fallback_pipeline(self, video_path: str) -> Dict[str, Any]:
        """Run traditional pipeline without Director"""
        logger.info("Running fallback pipeline without Director")
        
        from montage.core.whisper_transcriber import WhisperTranscriber
        from montage.providers.smart_track import SmartTrack
        from montage.core.highlight_selector import analyze_highlights
        from montage.core.ffmpeg_editor import FFMPEGEditor
        
        results = {
            "success": True,
            "pipeline": "fallback",
            "results": {}
        }
        
        try:
            # 1. Transcribe
            whisper_transcriber = WhisperTranscriber()
            transcript = whisper_transcriber.transcribe(video_path)
            results["results"]["transcript"] = transcript
            
            # 2. Analyze
            smart_track = SmartTrack()
            analysis = smart_track.analyze_video(video_path)
            results["results"]["analysis"] = analysis
            
            # 3. Select highlights
            highlights = analyze_highlights(
                analysis.get("segments", []),
                target_duration=60
            )
            results["results"]["highlights"] = highlights
            
            # 4. Create output
            if highlights:
                output_path = str(Path(video_path).parent / "highlights_output.mp4")
                editor = FFMPEGEditor(source_path=video_path)
                edit_result = editor.process(highlights, output_path)
                results["results"]["output_path"] = output_path
                results["results"]["edit_result"] = edit_result
            
        except Exception as e:
            logger.error(f"Fallback pipeline failed: {e}")
            results["success"] = False
            results["error"] = str(e)
        
        return results
    
    def create_custom_pipeline(self, agents: List[str], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create custom pipeline with specific agents
        
        Args:
            agents: List of agent names to use
            context: Context data for agents
            
        Returns:
            Pipeline results
        """
        if not self.director:
            logger.warning("Director not available for custom pipeline")
            return {"success": False, "error": "Director not available"}
        
        # Build agent chain
        instruction = f"Run these agents in sequence: {', '.join(agents)}"
        
        try:
            result = self.director.run(instruction, context=context)
            return self._process_director_output(result)
        except Exception as e:
            logger.error(f"Custom pipeline failed: {e}")
            return {"success": False, "error": str(e)}
    
    def list_available_agents(self) -> List[Dict[str, str]]:
        """List all available agents"""
        if not self.director:
            return []
        
        # This would depend on Director's API
        # For now, return our known agents
        return [
            {"name": "transcribe", "description": "Transcribe video audio"},
            {"name": "track_objects", "description": "Track objects in video"},
            {"name": "analyze_smart", "description": "Smart video analysis"},
            {"name": "diarize_speakers", "description": "Identify speakers"},
            {"name": "normalize_audio", "description": "Normalize audio levels"},
            {"name": "edit_video", "description": "Edit video clips"},
            {"name": "select_highlights", "description": "Select highlight clips"}
        ]


# Global instance
director_orchestrator = DirectorOrchestrator()


def run_director_pipeline(video_path: str, instruction: str = None) -> Dict[str, Any]:
    """
    Convenience function to run Director pipeline
    
    Args:
        video_path: Path to video file
        instruction: Natural language instruction
        
    Returns:
        Pipeline results
    """
    return director_orchestrator.run_pipeline(video_path, instruction)


def run_simple_director_example():
    """
    Example matching Tasks.md pattern exactly:
    
    result = director.run("Extract clips where people speak and track them")
    """
    if not DIRECTOR_AVAILABLE or not director:
        logger.error("Director not available")
        return None
    
    # Run exactly as shown in Tasks.md
    result = director.run("Extract clips where people speak and track them")
    return result