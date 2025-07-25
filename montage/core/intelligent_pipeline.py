#!/usr/bin/env python3
"""
Intelligent Video Pipeline Integration
End-to-end pipeline with all advanced features
"""
import os
import asyncio
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
import tempfile
import shutil

# Import all components
from .real_whisper_transcriber import RealWhisperTranscriber, UltraAccurateWhisperTranscriber
from .real_speaker_diarization import RealSpeakerDiarization
from .ai_highlight_selector import AIHighlightSelector, StoryBeatDetector
from .narrative_flow import NarrativeFlowOptimizer, EmotionalArcAnalyzer
from .smart_face_crop import SmartFaceCropper
from .animated_captions import AnimatedCaptionGenerator
from .audio_normalizer_fixed import EBUAudioNormalizer
from .creative_titles import AICreativeTitleGenerator, MultiPlatformOptimizer
from .emoji_overlay import IntelligentEmojiSystem, EmojiRenderer
from .process_metrics import get_resource_monitor

logger = logging.getLogger(__name__)


class IntelligentVideoPipeline:
    """Complete intelligent video processing pipeline"""
    
    def __init__(self, enable_metrics: bool = True):
        """Initialize all pipeline components"""
        # Core components
        self.transcriber = UltraAccurateWhisperTranscriber()
        self.diarizer = RealSpeakerDiarization()
        self.highlight_selector = AIHighlightSelector()
        self.story_detector = StoryBeatDetector()
        self.flow_optimizer = NarrativeFlowOptimizer()
        self.arc_analyzer = EmotionalArcAnalyzer()
        self.face_cropper = SmartFaceCropper()
        self.caption_generator = AnimatedCaptionGenerator()
        self.audio_normalizer = EBUAudioNormalizer()
        self.title_generator = MultiPlatformOptimizer()
        self.emoji_system = IntelligentEmojiSystem()
        self.emoji_renderer = EmojiRenderer()
        
        # Resource monitoring
        if enable_metrics:
            self.monitor = get_resource_monitor()
            self.monitor.start_monitoring()
        else:
            self.monitor = None
            
        logger.info("Intelligent Video Pipeline initialized")
        
    async def process_video(
        self,
        input_video: str,
        output_dir: str,
        config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Process video with all intelligent features
        
        Args:
            input_video: Path to input video
            output_dir: Output directory for clips
            config: Pipeline configuration
            
        Returns:
            Processing results with clip paths and metadata
        """
        # Default config
        if config is None:
            config = {
                "target_clips": 3,
                "clip_duration": 60,
                "platforms": ["tiktok", "instagram", "youtube"],
                "style": "viral",
                "enable_all_features": True
            }
            
        results = {
            "input": input_video,
            "clips": [],
            "metadata": {},
            "errors": []
        }
        
        try:
            # Create output directory
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            
            # Step 1: Transcription with Whisper
            logger.info("Step 1: Transcribing with Whisper...")
            transcript = await self.transcriber.transcribe(input_video)
            results["metadata"]["transcript_words"] = len(transcript.get("segments", []))
            
            # Step 2: Speaker Diarization
            logger.info("Step 2: Speaker diarization...")
            speaker_segments = await self.diarizer.diarize(input_video)
            results["metadata"]["speakers"] = len(set(s["speaker"] for s in speaker_segments))
            
            # Step 3: AI Highlight Selection
            logger.info("Step 3: AI highlight selection...")
            highlights = await self.highlight_selector.select_highlights(
                transcript,
                self._get_video_duration(input_video),
                target_clips=config["target_clips"]
            )
            
            # Step 4: Story Beat Detection
            logger.info("Step 4: Detecting story beats...")
            story_beats = self.story_detector.detect_beats(transcript)
            
            # Step 5: Narrative Flow Optimization
            logger.info("Step 5: Optimizing narrative flow...")
            # Convert highlights to segments for reordering
            segments = [
                {
                    "start": h.start,
                    "end": h.end,
                    "text": h.title,
                    "score": h.score
                }
                for h in highlights
            ]
            optimized_segments = self.flow_optimizer.optimize_flow(segments)
            
            # Process each highlight clip
            for i, segment in enumerate(optimized_segments):
                clip_result = await self._process_clip(
                    input_video,
                    segment,
                    i,
                    output_dir,
                    transcript,
                    speaker_segments,
                    config
                )
                results["clips"].append(clip_result)
                
            # Generate platform-specific content
            logger.info("Generating platform-specific content...")
            for clip in results["clips"]:
                platform_content = await self.title_generator.generate_multi_platform_content(
                    {"text": clip["transcript_text"]}
                )
                clip["platform_content"] = platform_content
                
            results["success"] = True
            
        except Exception as e:
            logger.error(f"Pipeline error: {e}")
            results["errors"].append(str(e))
            results["success"] = False
            
        finally:
            # Stop metrics server
            if self.monitor:
                self.monitor.stop_monitoring()
                
        return results
        
    async def _process_clip(
        self,
        input_video: str,
        segment: Dict,
        index: int,
        output_dir: str,
        transcript: Dict,
        speaker_segments: List[Dict],
        config: Dict
    ) -> Dict[str, Any]:
        """Process individual clip with all features"""
        clip_name = f"clip_{index:02d}"
        temp_dir = tempfile.mkdtemp()
        
        try:
            # Extract clip
            clip_path = os.path.join(temp_dir, f"{clip_name}_raw.mp4")
            await self._extract_clip(input_video, clip_path, segment["start"], segment["end"])
            
            # Step 6: Smart Face Crop
            logger.info(f"Processing clip {index}: Smart crop...")
            cropped_path = os.path.join(temp_dir, f"{clip_name}_cropped.mp4")
            self.face_cropper.process_video(clip_path, cropped_path)
            
            # Step 7: Audio Normalization
            logger.info(f"Processing clip {index}: Audio normalization...")
            normalized_path = os.path.join(temp_dir, f"{clip_name}_normalized.mp4")
            self.audio_normalizer.normalize(cropped_path, normalized_path)
            
            # Step 8: Animated Captions
            logger.info(f"Processing clip {index}: Animated captions...")
            captioned_path = os.path.join(temp_dir, f"{clip_name}_captioned.mp4")
            clip_transcript = self._extract_clip_transcript(transcript, segment)
            await self.caption_generator.generate_animated_captions(
                clip_transcript,
                normalized_path,
                captioned_path,
                animation="karaoke"
            )
            
            # Step 9: Emoji Overlays
            logger.info(f"Processing clip {index}: Emoji overlays...")
            emotions = self._analyze_emotions(clip_transcript)
            emoji_overlays = self.emoji_system.analyze_and_place_emojis(
                clip_transcript,
                emotions,
                (1080, 1920)
            )
            
            final_path = os.path.join(output_dir, f"{clip_name}_final.mp4")
            if emoji_overlays:
                self.emoji_renderer.apply_emoji_overlays(
                    captioned_path,
                    final_path,
                    emoji_overlays
                )
            else:
                shutil.copy(captioned_path, final_path)
                
            # Extract metadata
            clip_result = {
                "filename": os.path.basename(final_path),
                "path": final_path,
                "start_time": segment["start"],
                "end_time": segment["end"],
                "duration": segment["end"] - segment["start"],
                "transcript_text": self._get_segment_text(clip_transcript),
                "speakers": self._get_clip_speakers(speaker_segments, segment),
                "emojis_used": [e.emoji for e in emoji_overlays],
                "features_applied": [
                    "whisper_transcription",
                    "speaker_diarization",
                    "ai_highlights",
                    "narrative_flow",
                    "smart_crop",
                    "audio_normalization",
                    "animated_captions",
                    "emoji_overlays"
                ]
            }
            
            return clip_result
            
        finally:
            # Cleanup temp files
            shutil.rmtree(temp_dir, ignore_errors=True)
            
    async def _extract_clip(self, input_video: str, output_path: str, 
                          start: float, end: float):
        """Extract video clip"""
        duration = end - start
        cmd = [
            "ffmpeg",
            "-i", input_video,
            "-ss", str(start),
            "-t", str(duration),
            "-c", "copy",
            output_path,
            "-y"
        ]
        
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL
        )
        await process.wait()
        
    def _get_video_duration(self, video_path: str) -> float:
        """Get video duration"""
        import subprocess
        cmd = [
            "ffprobe",
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            video_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        return float(result.stdout.strip())
        
    def _extract_clip_transcript(self, full_transcript: Dict, 
                               segment: Dict) -> Dict[str, Any]:
        """Extract transcript for clip time range"""
        clip_segments = []
        
        for trans_segment in full_transcript.get("segments", []):
            if (trans_segment["start"] >= segment["start"] and 
                trans_segment["end"] <= segment["end"]):
                clip_segments.append(trans_segment)
                
        return {
            "segments": clip_segments,
            "text": " ".join(s["text"] for s in clip_segments)
        }
        
    def _get_segment_text(self, transcript: Dict) -> str:
        """Get text from transcript"""
        return transcript.get("text", "")
        
    def _get_clip_speakers(self, speaker_segments: List[Dict], 
                          segment: Dict) -> List[str]:
        """Get speakers in clip"""
        speakers = set()
        
        for speaker_seg in speaker_segments:
            if (speaker_seg["start"] >= segment["start"] and 
                speaker_seg["end"] <= segment["end"]):
                speakers.add(speaker_seg["speaker"])
                
        return list(speakers)
        
    def _analyze_emotions(self, transcript: Dict) -> List[Dict]:
        """Simple emotion analysis"""
        # In production, use proper emotion detection
        emotions = []
        
        emotion_keywords = {
            "happy": ["happy", "joy", "excited", "amazing", "wonderful"],
            "sad": ["sad", "cry", "miss", "sorry", "unfortunately"],
            "surprised": ["wow", "amazing", "unbelievable", "shocked"],
            "angry": ["angry", "mad", "frustrated", "annoyed"]
        }
        
        for segment in transcript.get("segments", []):
            text_lower = segment["text"].lower()
            
            for emotion, keywords in emotion_keywords.items():
                if any(keyword in text_lower for keyword in keywords):
                    emotions.append({
                        "emotion": emotion,
                        "start": segment["start"],
                        "end": segment["end"],
                        "intensity": 0.8
                    })
                    break
                    
        return emotions


async def main():
    """Example usage"""
    pipeline = IntelligentVideoPipeline()
    
    results = await pipeline.process_video(
        input_video="/path/to/video.mp4",
        output_dir="/path/to/output",
        config={
            "target_clips": 3,
            "platforms": ["tiktok", "instagram"],
            "style": "viral"
        }
    )
    
    print(f"Generated {len(results['clips'])} clips")
    for clip in results["clips"]:
        print(f"- {clip['filename']}: {clip['duration']:.1f}s")
        

if __name__ == "__main__":
    asyncio.run(main())