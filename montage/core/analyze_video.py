import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict
import logging
from .real_whisper_transcriber import RealWhisperTranscriber
from .speaker_diarizer import SpeakerDiarizer
from .ai_highlight_selector import AIHighlightSelector, StoryBeatDetector
import asyncio

logger = logging.getLogger(__name__)

SCENE_THRESHOLD = 30.0  # adjust per your content

def extract_frames(video_path: str, interval: float = 1.0) -> List[np.ndarray]:
    if not Path(video_path).is_file():
        raise FileNotFoundError(f"Video not found: {video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_interval = max(int(fps * interval), 1)

    frames = []
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % frame_interval == 0:
            frames.append(frame)
        idx += 1
    cap.release()
    return frames

def detect_scene_changes(frames: List[np.ndarray]) -> List[int]:
    if not frames:
        return []
    scene_idxs = [0]
    prev = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
    for i, frame in enumerate(frames[1:], start=1):
        curr = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(prev, curr)
        if float(np.mean(diff)) > SCENE_THRESHOLD:
            scene_idxs.append(i)
        prev = curr
    return scene_idxs

def analyze_motion(frame1: np.ndarray, frame2: np.ndarray) -> float:
    g1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(
        g1, g2, None,
        pyr_scale=0.5, levels=3, winsize=15,
        iterations=3, poly_n=5, poly_sigma=1.2, flags=0
    )
    return float(np.mean(np.abs(flow)))

def analyze_video_content(video_path: str) -> Dict:
    frames = extract_frames(video_path)
    scenes = detect_scene_changes(frames)
    motions: List[float] = []
    for a, b in zip(frames, frames[1:]):
        motions.append(analyze_motion(a, b))
    avg_motion = float(np.mean(motions)) if motions else 0.0
    return {
        "frame_count": len(frames),
        "scene_changes": scenes,
        "average_motion": avg_motion
    }

def analyze_video(video_path: str, use_premium: bool = False) -> Dict:
    """Main video analysis with transcription and speaker diarization"""
    content = analyze_video_content(video_path)
    
    # Real transcription
    transcript = []
    words = []
    try:
        transcriber = RealWhisperTranscriber(model_size="base")
        transcript_result = transcriber.transcribe(video_path)
        
        # Extract transcript and words
        transcript = transcript_result.get("segments", [])
        words = []
        for segment in transcript:
            if "words" in segment:
                words.extend(segment["words"])
                
    except Exception as e:
        logger.warning(f"Transcription failed: {e}")
        
    # Speaker diarization
    speaker_segments = []
    speaker_turns = []
    try:
        diarizer = SpeakerDiarizer()
        speaker_segments = diarizer.diarize_video(video_path)
        
        # Merge with transcript if available
        if transcript:
            transcript = diarizer.merge_with_transcript(speaker_segments, transcript)
            
        # Generate speaker turns (consecutive segments by same speaker)
        if speaker_segments:
            current_turn = None
            for seg in speaker_segments:
                if current_turn is None or current_turn["speaker"] != seg["speaker"]:
                    if current_turn:
                        speaker_turns.append(current_turn)
                    current_turn = {
                        "speaker": seg["speaker"],
                        "start": seg["start"],
                        "end": seg["end"]
                    }
                else:
                    current_turn["end"] = seg["end"]
            if current_turn:
                speaker_turns.append(current_turn)
                
    except Exception as e:
        logger.warning(f"Speaker diarization failed: {e}")
        
    # Generate smart highlights using AI if available
    highlights = []
    story_beats = []
    
    if transcript and use_premium:
        try:
            # Use AI for intelligent highlight selection
            ai_selector = AIHighlightSelector()
            
            # Get video duration
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            video_duration = frame_count / fps if fps > 0 else 0
            cap.release()
            
            # Run AI selection (async)
            async def run_ai_selection():
                return await ai_selector.select_highlights(
                    {"segments": transcript},
                    video_duration,
                    target_clips=5
                )
            
            # Execute async function
            ai_highlights = asyncio.run(run_ai_selection())
            
            # Convert to standard format
            for ai_hl in ai_highlights:
                highlight = {
                    "start": ai_hl.start,
                    "end": ai_hl.end,
                    "score": ai_hl.score,
                    "title": ai_hl.title,
                    "text": ai_hl.reason,
                    "type": "ai_selected",
                    "story_beat": ai_hl.story_beat,
                    "emotions": ai_hl.emotions,
                    "keywords": ai_hl.keywords
                }
                highlights.append(highlight)
                
            # Also detect story beats
            beat_detector = StoryBeatDetector()
            story_beats = beat_detector.detect_beats({"segments": transcript})
            
        except Exception as e:
            logger.warning(f"AI highlight selection failed: {e}, falling back to basic selection")
            # Fall back to basic transcript-based selection
            for i, segment in enumerate(transcript[:5]):
                if len(segment.get("text", "").split()) > 5:
                    highlight = {
                        "start": segment["start"],
                        "end": segment["end"],
                        "score": 0.9 - (i * 0.1),
                        "text": segment.get("text", "")[:100],
                        "type": "transcript_based"
                    }
                    if "speaker" in segment:
                        highlight["speaker"] = segment["speaker"]
                    highlights.append(highlight)
    
    elif transcript:
        # Basic transcript-based selection without AI
        for i, segment in enumerate(transcript[:5]):
            if len(segment.get("text", "").split()) > 5:
                highlight = {
                    "start": segment["start"],
                    "end": segment["end"],
                    "score": 0.9 - (i * 0.1),
                    "text": segment.get("text", "")[:100],
                    "type": "transcript_based"
                }
                if "speaker" in segment:
                    highlight["speaker"] = segment["speaker"]
                highlights.append(highlight)
    else:
        # Fallback to scene-based highlights
        for i, scene_idx in enumerate(content["scene_changes"][:5]):
            highlights.append({
                "start": float(scene_idx),
                "end": float(scene_idx + 5.0),
                "score": 0.8 - (i * 0.1),
                "type": "scene_change"
            })
    
    return {
        "highlights": highlights,
        "analysis": content,
        "transcript": transcript,
        "words": words,
        "speaker_segments": speaker_segments,
        "speaker_turns": speaker_turns,
        "story_beats": story_beats
    }
